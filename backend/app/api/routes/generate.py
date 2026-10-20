import json
import csv
import io
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import Optional
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.dataset import Dataset, DatasetStatus
from app.services.dataset_generator import generate_dataset_stream

router = APIRouter(prefix="/generate", tags=["generate"])

# ── Per-user concurrency gate (in-memory, upgradeable to Redis for multi-worker)
# Limits each user to 2 simultaneous generation requests. Prevents API abuse
# and runaway costs if a client retries aggressively.
_user_generation_locks: dict[str, asyncio.Semaphore] = {}


def _get_user_semaphore(user_id: str) -> asyncio.Semaphore:
    if user_id not in _user_generation_locks:
        _user_generation_locks[user_id] = asyncio.Semaphore(2)  # max 2 concurrent per user
    return _user_generation_locks[user_id]


async def sse_keepalive_generator(generator, interval: float = 15.0):
    iterator = generator.__aiter__()
    next_task = None
    try:
        while True:
            next_task = asyncio.create_task(iterator.__anext__())
            while True:
                done, pending = await asyncio.wait(
                    [next_task],
                    timeout=interval,
                    return_when=asyncio.FIRST_COMPLETED
                )
                if next_task in done:
                    yield next_task.result()
                    next_task = None
                    break
                else:
                    yield ": ping\n\n"
    except StopAsyncIteration:
        pass
    finally:
        if next_task and not next_task.done():
            next_task.cancel()


class GenerateRequest(BaseModel):
    topic: str
    task_type: str = "chat"
    style: str = "casual"
    total_rows: int = 100
    domain_context: str = ""
    dataset_name: str = ""
    save_to_account: bool = True
    # Phase 2 Multi-Turn: Enables generating nested OpenAI messages representation alongside flat input/outputs
    multi_turn: bool = False
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class PreviewRequest(BaseModel):
    topic: str
    task_type: str = "chat"
    style: str = "casual"
    domain_context: str = ""
    multi_turn: bool = False
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


@router.post("/preview")
async def preview_dataset(body: PreviewRequest, user_id: str = Depends(get_current_user_id)):
    rows_collected = []
    async for event in generate_dataset_stream(
        topic=body.topic, task_type=body.task_type,
        style=body.style, total_rows=5,
        domain_context=body.domain_context,
        multi_turn=body.multi_turn,
        openai_api_key=body.openai_api_key,
        deepseek_api_key=body.deepseek_api_key,
        groq_api_key=body.groq_api_key,
    ):
        if event["event"] == "complete":
            rows_collected = event.get("rows", [])
    return {"preview_rows": rows_collected[:5], "message": "Preview generated."}


@router.post("/dataset")
async def generate_and_save(
    request: Request,
    body: GenerateRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    has_custom_keys = bool(
        body.openai_api_key or 
        body.deepseek_api_key or 
        request.headers.get("x-key-configured") == "true"
    )
    if not has_custom_keys:
        result = await db.execute(
            select(Dataset).where(Dataset.user_id == user_id)
        )
        existing_count = len(result.scalars().all())
        if existing_count >= 1:
            raise HTTPException(
                400,
                "Free-tier limit reached: maximum 1 dataset allowed. Please configure your own API keys in Settings to remove this limit."
            )
        if body.total_rows > 20:
            raise HTTPException(
                400,
                "Free-tier limit reached: maximum 20 generated rows allowed. Please configure your own API keys in Settings to remove this limit."
            )

    if not 10 <= body.total_rows <= 5000:
        raise HTTPException(400, "total_rows must be between 10 and 5000")
    if body.task_type not in ["chat", "instruction", "qa", "classification", "summarization"]:
        raise HTTPException(400, "Invalid task_type")
    if body.style not in ["formal", "casual", "technical", "simple"]:
        raise HTTPException(400, "Invalid style")
    if len(body.topic.strip()) < 5:
        raise HTTPException(400, "Topic too short")

    all_rows = []
    audit_result = {}
    inferred_policy = {}
    fingerprint = {}
    coverage_plan = {}

    async for event in generate_dataset_stream(
        topic=body.topic, task_type=body.task_type,
        style=body.style, total_rows=body.total_rows,
        domain_context=body.domain_context,
        multi_turn=body.multi_turn,
        openai_api_key=body.openai_api_key,
        deepseek_api_key=body.deepseek_api_key,
    ):
        if event["event"] == "complete":
            all_rows = event.get("rows", [])
            audit_result = event.get("audit", {})
            inferred_policy = event.get("policy", {})
            fingerprint = event.get("fingerprint", {})
            coverage_plan = event.get("coverage_plan", {})

    if not all_rows:
        raise HTTPException(500, "Generation failed — no rows produced")

    # Store rows in audit_results so we can serve download later
    dataset_name = body.dataset_name or inferred_policy.get("dataset_name") or f"{body.topic[:40]} ({body.task_type})"
    dataset = Dataset(
        user_id=user_id,
        name=dataset_name,
        description=f"Synthetic | {body.topic} | {body.style} | {body.task_type}",
        file_size_bytes=len(json.dumps(all_rows).encode()),
        file_format="csv",
        row_count=len(all_rows),
        quality_score=audit_result.get("quality_score", 0),
        audit_results={
            **audit_result,
            "rows": all_rows,  # store rows for download
            "inferred_policy": inferred_policy,
            "fingerprint": fingerprint,
            "coverage_plan": coverage_plan,
            "generation_config": {
                "topic": body.topic, "task_type": body.task_type,
                "style": body.style, "total_rows": body.total_rows,
                "domain_context": body.domain_context, "multi_turn": body.multi_turn,
            }
        },
        status=DatasetStatus.ready,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "row_count": len(all_rows),
        "quality_score": audit_result.get("quality_score"),
        "audit": audit_result,
        "fingerprint": fingerprint,
        "coverage_plan": coverage_plan,
        "status": "ready",
        "message": f"Generated {len(all_rows)} high-quality rows successfully",
        "sample_rows": all_rows[:3],
        "rows": all_rows,  # return all rows to frontend for immediate download
    }


@router.post("/dataset/stream")
async def generate_and_save_stream(
    request: Request,
    body: GenerateRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    has_custom_keys = bool(
        body.openai_api_key or
        body.deepseek_api_key or
        body.groq_api_key or
        request.headers.get("x-key-configured") == "true"
    )
    if not has_custom_keys:
        result = await db.execute(
            select(Dataset).where(Dataset.user_id == user_id)
        )
        existing_count = len(result.scalars().all())
        if existing_count >= 1:
            raise HTTPException(
                400,
                "Free-tier limit reached: maximum 1 dataset allowed. Please configure your own API keys in Settings to remove this limit."
            )
        if body.total_rows > 20:
            raise HTTPException(
                400,
                "Free-tier limit reached: maximum 20 generated rows allowed. Please configure your own API keys in Settings to remove this limit."
            )

    if not 10 <= body.total_rows <= 5000:
        raise HTTPException(400, "total_rows must be between 10 and 5000")
    if body.task_type not in ["chat", "instruction", "qa", "classification", "summarization"]:
        raise HTTPException(400, "Invalid task_type")
    if body.style not in ["formal", "casual", "technical", "simple"]:
        raise HTTPException(400, "Invalid style")
    if len(body.topic.strip()) < 5:
        raise HTTPException(400, "Topic too short")

    # Rate limit: reject if user already has 2 concurrent generation jobs running
    sem = _get_user_semaphore(user_id)
    if sem._value == 0:  # no slots left
        raise HTTPException(
            429,
            "You already have 2 generation jobs running. Please wait for one to finish."
        )

    # Wrap the generator in semaphore gate to enforce concurrency limits
    async def gated_generator_func():
        async with _get_user_semaphore(user_id):
            all_rows = []
            audit_result = {}
            inferred_policy = {}
            fingerprint = {}
            coverage_plan = {}
            try:
                async for event in generate_dataset_stream(
                    topic=body.topic, task_type=body.task_type,
                    style=body.style, total_rows=body.total_rows,
                    domain_context=body.domain_context,
                    multi_turn=body.multi_turn,
                    openai_api_key=body.openai_api_key,
                    deepseek_api_key=body.deepseek_api_key,
                    groq_api_key=body.groq_api_key,
                ):
                    if event.get("event") == "complete":
                        all_rows = event.get("rows", [])
                        audit_result = event.get("audit", {})
                        inferred_policy = event.get("policy", {})
                        fingerprint = event.get("fingerprint", {})
                        coverage_plan = event.get("coverage_plan", {})

                        # Save to database
                        ds_name = body.dataset_name or inferred_policy.get("dataset_name") or f"{body.topic[:40]} ({body.task_type})"
                        dataset = Dataset(
                            user_id=user_id,
                            name=ds_name,
                            description=f"Synthetic | {body.topic} | {body.style} | {body.task_type}",
                            file_size_bytes=len(json.dumps(all_rows).encode()),
                            file_format="csv",
                            row_count=len(all_rows),
                            quality_score=audit_result.get("quality_score", 0),
                            audit_results={
                                **audit_result,
                                "rows": all_rows,
                                "inferred_policy": inferred_policy,
                                "fingerprint": fingerprint,
                                "coverage_plan": coverage_plan,
                                "generation_config": {
                                    "topic": body.topic, "task_type": body.task_type,
                                    "style": body.style, "total_rows": body.total_rows,
                                    "domain_context": body.domain_context, "multi_turn": body.multi_turn,
                                }
                            },
                            status=DatasetStatus.ready,
                        )
                        db.add(dataset)
                        await db.commit()
                        await db.refresh(dataset)

                        event["id"] = dataset.id
                        event["name"] = dataset.name
                        event["row_count"] = body.total_rows
                        event["status"] = "ready"
                        event["message"] = f"Generated {body.total_rows} high-quality rows successfully"
                        event["sample_rows"] = all_rows[:3]
                        event["quality_score"] = audit_result.get("quality_score", 0)

                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Error in generation: {e}", exc_info=True)
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        sse_keepalive_generator(gated_generator_func()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/download/{dataset_id}")
async def download_dataset(
    dataset_id: str,
    format: str = "csv",
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    rows = dataset.audit_results.get("rows", []) if dataset.audit_results else []

    if format == "jsonl":
        content = "\n".join(json.dumps(row) for row in rows)
        return Response(
            content=content,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{dataset.name}.jsonl"'}
        )
    else:
        # CSV
        if rows:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            content = buf.getvalue()
        else:
            content = ""
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{dataset.name}.csv"'}
        )
