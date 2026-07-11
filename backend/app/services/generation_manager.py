import asyncio
import logging
import json
from typing import Dict, List, Callable, Any, AsyncGenerator
from sqlalchemy import update
from app.core.database import AsyncSessionLocal
from app.models.dataset import Dataset, DatasetStatus

logger = logging.getLogger(__name__)

class GenerationTask:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.events: List[dict] = []
        self.listeners: List[asyncio.Queue] = []
        self.completed = False
        self.error = None
        self.task: asyncio.Task = None

    def add_event(self, event: dict):
        self.events.append(event)
        for queue in self.listeners:
            queue.put_nowait(event)

    def add_listener(self) -> asyncio.Queue:
        queue = asyncio.Queue()
        # Pre-fill queue with all events emitted so far
        for event in self.events:
            queue.put_nowait(event)
        self.listeners.append(queue)
        return queue

    def remove_listener(self, queue: asyncio.Queue):
        if queue in self.listeners:
            self.listeners.remove(queue)


class GenerationManager:
    def __init__(self):
        self._tasks: Dict[str, GenerationTask] = {}

    def get_task(self, dataset_id: str) -> GenerationTask:
        return self._tasks.get(dataset_id)

    def register_task(self, dataset_id: str) -> GenerationTask:
        if dataset_id not in self._tasks:
            self._tasks[dataset_id] = GenerationTask(dataset_id)
        return self._tasks[dataset_id]

    def start_generation(
        self,
        dataset_id: str,
        generator_func: Callable[[], AsyncGenerator[dict, None]],
        save_callback: Callable[[list[dict], dict, dict, dict, dict, Any], Any] = None
    ):
        task_info = self.register_task(dataset_id)
        
        async def run_task():
            all_rows = []
            final_audit = {}
            inferred_policy = {}
            fingerprint = {}
            coverage_plan = {}
            try:
                async for event in generator_func():
                    task_info.add_event(event)
                    if event.get("event") == "complete":
                        all_rows = event.get("rows", [])
                        final_audit = event.get("audit", {})
                        inferred_policy = event.get("policy", {})
                        fingerprint = event.get("fingerprint", {})
                        coverage_plan = event.get("coverage_plan", {})
                    elif event.get("event") == "error":
                        raise Exception(event.get("message", "Unknown error in event stream"))
                
                # Update DB to ready
                async with AsyncSessionLocal() as db:
                    if save_callback:
                        await save_callback(all_rows, final_audit, inferred_policy, fingerprint, coverage_plan, db)
                    else:
                        # Fallback simple update
                        await db.execute(
                            update(Dataset)
                            .where(Dataset.id == dataset_id)
                            .values(status=DatasetStatus.ready)
                        )
                        await db.commit()
                task_info.completed = True
            except asyncio.CancelledError:
                logger.info(f"Generation task {dataset_id} cancelled.")
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(Dataset)
                        .where(Dataset.id == dataset_id)
                        .values(status=DatasetStatus.failed)
                    )
                    await db.commit()
                task_info.error = "Cancelled"
                task_info.add_event({"event": "error", "message": "Generation task cancelled."})
            except Exception as e:
                logger.error(f"Error in background generation task {dataset_id}: {e}", exc_info=True)
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(Dataset)
                        .where(Dataset.id == dataset_id)
                        .values(status=DatasetStatus.failed)
                    )
                    await db.commit()
                task_info.error = str(e)
                task_info.add_event({"event": "error", "message": str(e)})
            finally:
                task_info.completed = True
                # Clean up memory after a small delay (e.g. 5 minutes) to allow reconnects to read complete state
                async def delayed_cleanup():
                    await asyncio.sleep(300)
                    self._tasks.pop(dataset_id, None)
                asyncio.create_task(delayed_cleanup())

        task_info.task = asyncio.create_task(run_task())

    async def event_generator(self, dataset_id: str) -> AsyncGenerator[str, None]:
        task_info = self.get_task(dataset_id)
        if not task_info:
            # Task not active in memory (either completed long ago or server restarted)
            # Query Database status
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                stmt = select(Dataset).where(Dataset.id == dataset_id)
                res = await db.execute(stmt)
                dataset = res.scalar_one_or_none()
                if not dataset:
                    yield f"data: {json.dumps({'event': 'error', 'message': 'Dataset not found'})}\n\n"
                    return
                
                if dataset.status == DatasetStatus.ready:
                    # Completed already, send a mock complete event with rows
                    audit_data = dataset.audit_results or {}
                    rows = audit_data.get("rows", [])
                    fingerprint = audit_data.get("fingerprint") or {}
                    coverage_plan = audit_data.get("coverage_plan") or {}
                    policy = audit_data.get("inferred_policy") or {}
                    columns = list(rows[0].keys()) if rows else []
                    yield f"data: {json.dumps({'event': 'complete', 'id': dataset.id, 'name': dataset.name, 'row_count': dataset.row_count, 'quality_score': dataset.quality_score, 'audit': audit_data, 'rows': rows, 'fingerprint': fingerprint, 'coverage_plan': coverage_plan, 'policy': policy, 'columns': columns, 'status': 'ready'})}\n\n"
                    return
                elif dataset.status == DatasetStatus.failed:
                    yield f"data: {json.dumps({'event': 'error', 'message': 'Generation failed previously.'})}\n\n"
                    return
                else:
                    # Database status is generating, but task not in memory!
                    # This means it was interrupted by a server restart.
                    dataset.status = DatasetStatus.failed
                    db.add(dataset)
                    await db.commit()
                    yield f"data: {json.dumps({'event': 'error', 'message': 'Generation interrupted (server restart).'})}\n\n"
                    return

        # Task is in memory, stream it
        queue = task_info.add_listener()
        try:
            while True:
                # If queue is empty and task is completed, we are done
                if queue.empty() and task_info.completed:
                    break
                
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    queue.task_done()
                    if event.get("event") in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    if task_info.completed:
                        break
        finally:
            task_info.remove_listener(queue)

# Global singleton manager
generation_manager = GenerationManager()
