"""
Training worker — orchestrator for local/RunPod GPU training.
Manages provider selection, limits validation, status checking, and teardown.
"""
import os
import time
import asyncio
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "finetune_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one job at a time per worker
)

def run_async(coro):
    """
    Helper to run an async function in a sync Celery worker environment.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Fallback if loop is already running
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

@celery_app.task(bind=True, name="run_training_job")
def run_training_job(self, job_id: str):
    """
    Orchestrate the fine-tuning job using the configured provider.
    Enforces cost protection and ensures pod/process auto-termination.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.models.job import TrainingJob, JobStatus
    from app.models.dataset import Dataset
    from app.services.trainer import get_trainer
    from datetime import datetime, timezone

    # Use sync engine for Celery database operations
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)

    def update_job(session, **kwargs):
        session.query(TrainingJob).filter(TrainingJob.id == job_id).update(kwargs)
        session.commit()

    trainer = get_trainer()

    with Session(engine) as session:
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            print(f"[Worker] Job {job_id} not found in database.")
            return {"status": "failed", "error": "Job not found"}

        try:
            # 1. Cost Protection / Hard Limits Validation
            dataset = session.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if not dataset:
                raise ValueError("Associated dataset not found.")

            row_count = dataset.row_count or 0
            training_config = job.training_config or {}
            epochs = training_config.get("epochs", 3)
            max_seq_length = training_config.get("max_seq_length", 512)
            batch_size = training_config.get("batch_size", 4)

            # Hard limits checks
            if row_count > 5000:
                raise ValueError(f"Dataset row count ({row_count}) exceeds limit of 5000 rows.")
            if epochs > 3:
                raise ValueError(f"Epoch count ({epochs}) exceeds limit of 3 epochs.")
            if max_seq_length > 2048:
                raise ValueError(f"Max sequence length ({max_seq_length}) exceeds limit of 2048.")

            # Calculate estimated cost
            provider_name = os.getenv("TRAINING_PROVIDER", "local").strip().lower()
            if provider_name == "runpod":
                # RTX 4090 rate: ~0.44 USD/hour. Let's assume a training speed of 0.5s per step.
                steps = (row_count * epochs) // batch_size
                estimated_seconds = max(60, steps * 0.5)
                estimated_cost = round((estimated_seconds / 3600.0) * 0.44, 4)
            else:
                estimated_cost = 0.0

            if estimated_cost > 1.00:
                raise ValueError(f"Estimated training cost (${estimated_cost:.2f}) exceeds budget of $1.00.")

            # Update DB with validation details and transition state
            update_job(
                session,
                status=JobStatus.provisioning,
                estimated_cost_usd=estimated_cost,
                started_at=datetime.now(timezone.utc)
            )
            self.update_state(state="PROVISIONING", meta={"job_id": job_id})

            # 2. Launch training job
            start_res = run_async(trainer.start_job(job_id, {
                **training_config,
                "base_model": job.base_model,
                "dataset_id": job.dataset_id
            }))

            pod_id = start_res.get("pod_id")
            update_job(session, runpod_pod_id=pod_id)

            # 3. Monitor training loop
            max_duration = 3600 * 2  # Hard limit: 2 hours maximum
            start_time = time.time()

            while True:
                # Timeout safeguard
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    raise TimeoutError("Training exceeded maximum allowed runtime (2 hours).")

                # Refresh job from database
                session.expire_all()
                db_job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if not db_job:
                    raise ValueError("Job deleted during training.")

                # If job was cancelled or marked failed/completed by callback
                if db_job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
                    print(f"[Worker] Job {job_id} reached terminal state: {db_job.status}")
                    break

                # Query status from provider
                status_res = run_async(trainer.get_status(job_id))
                provider_status = status_res.get("status")

                if provider_status == "finished":
                    # Let the callbacks process, sleep a little
                    time.sleep(5)
                    session.expire_all()
                    db_job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                    if db_job.status not in [JobStatus.completed, JobStatus.failed]:
                        # Fallback if callback missed
                        raise RuntimeError("Provider completed training, but callback was not received.")
                    break
                elif provider_status == "failed":
                    raise RuntimeError(status_res.get("error", "Provider training job failed."))

                # Update Celery task metadata
                self.update_state(
                    state="TRAINING" if db_job.status == JobStatus.training else db_job.status.value.upper(),
                    meta={
                        "job_id": job_id,
                        "step": db_job.current_step,
                        "loss": db_job.train_loss,
                        "epoch": db_job.current_epoch
                    }
                )

                time.sleep(10)

            return {"status": db_job.status.value, "job_id": job_id}

        except Exception as e:
            print(f"[Worker] Exception in training job {job_id}: {e}")
            update_job(session, status=JobStatus.failed, error_message=str(e))
            raise

        finally:
            # Enforce Auto-Termination and Billing Protection
            print(f"[Worker] Triggering auto-termination cleanup for job {job_id}...")
            try:
                run_async(trainer.stop_job(job_id))
            except Exception as cleanup_err:
                print(f"[Worker] Cleanup failed for job {job_id}: {cleanup_err}")


@celery_app.task(bind=True, name="run_evaluation_job")
def run_evaluation_job(self, evaluation_job_id: str):
    """
    Run evaluation job asynchronously.
    """
    from app.services.evaluator import evaluate_model
    self.update_state(state="RUNNING", meta={"evaluation_job_id": evaluation_job_id})
    try:
        res = run_async(evaluate_model(evaluation_job_id))
        return res
    except Exception as e:
        print(f"[Worker] Evaluation job {evaluation_job_id} failed: {e}")
        raise
