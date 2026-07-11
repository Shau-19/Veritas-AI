from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.evaluation import EvaluationJob, EvaluationStatus
from app.models.job import TrainingJob, JobStatus
from app.models.dataset import Dataset, DatasetStatus

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class CreateEvaluationRequest(BaseModel):
    training_job_id: str
    dataset_id: str


class EvaluationResponse(BaseModel):
    id: str
    training_job_id: str
    dataset_id: str
    status: EvaluationStatus
    metrics: Optional[Dict[str, Any]] = None
    comparison_data: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str


@router.post("/", status_code=201)
async def create_evaluation(
    body: CreateEvaluationRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    # Validate training job belongs to user and is completed
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.id == body.training_job_id, TrainingJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Training job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(400, f"Training job is not completed. Current status: {job.status}")

    # Validate dataset belongs to user and is ready
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == body.dataset_id, Dataset.user_id == user_id)
    )
    dataset = ds_result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    if dataset.status != DatasetStatus.ready:
        raise HTTPException(400, "Dataset is not ready")

    eval_job = EvaluationJob(
        user_id=user_id,
        training_job_id=body.training_job_id,
        dataset_id=body.dataset_id,
        status=EvaluationStatus.queued,
    )
    db.add(eval_job)
    await db.commit()
    await db.refresh(eval_job)

    # Dispatch to Celery task only if not a client-driven colab job
    is_colab = False
    if job.training_config and job.training_config.get("provider") == "colab":
        is_colab = True

    if not is_colab:
        try:
            from app.workers.training_worker import run_evaluation_job
            run_evaluation_job.delay(eval_job.id)
        except Exception:
            # Celery dispatcher failure fallback (e.g. redis offline)
            pass

    return {
        "id": eval_job.id,
        "status": eval_job.status,
        "message": "Evaluation job queued successfully",
    }


@router.get("/", response_model=List[EvaluationResponse])
async def list_evaluations(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EvaluationJob)
        .where(EvaluationJob.user_id == user_id)
        .order_by(EvaluationJob.created_at.desc())
    )
    jobs = result.scalars().all()
    
    return [
        EvaluationResponse(
            id=job.id,
            training_job_id=job.training_job_id,
            dataset_id=job.dataset_id,
            status=job.status,
            metrics=job.metrics,
            comparison_data=job.comparison_data,
            error_message=job.error_message,
            completed_at=str(job.completed_at) if job.completed_at else None,
            created_at=str(job.created_at),
        )
        for job in jobs
    ]


@router.get("/{eval_id}", response_model=EvaluationResponse)
async def get_evaluation(
    eval_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EvaluationJob).where(EvaluationJob.id == eval_id, EvaluationJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Evaluation job not found")
        
    return EvaluationResponse(
        id=job.id,
        training_job_id=job.training_job_id,
        dataset_id=job.dataset_id,
        status=job.status,
        metrics=job.metrics,
        comparison_data=job.comparison_data,
        error_message=job.error_message,
        completed_at=str(job.completed_at) if job.completed_at else None,
        created_at=str(job.created_at),
    )


@router.get("/queued/{job_id}")
async def get_queued_evaluation(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Allows Colab notebook to fetch the oldest queued evaluation job and its prompts.
    """
    # Fetch the oldest queued evaluation job for this training job
    result = await db.execute(
        select(EvaluationJob)
        .where(
            EvaluationJob.training_job_id == job_id,
            EvaluationJob.status == EvaluationStatus.queued
        )
        .order_by(EvaluationJob.created_at.asc())
    )
    eval_job = result.scalar_one_or_none()
    if not eval_job:
        return {"eval_job_id": None}

    # Get the test dataset to fetch prompts
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == eval_job.dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Test dataset not found")

    rows = dataset.audit_results.get("rows", []) if dataset.audit_results else []
    # Limit evaluation to a maximum of 15 rows for fast execution
    eval_rows = rows[:15]

    prompts = []
    for row in eval_rows:
        if "messages" in row and isinstance(row["messages"], list):
            user_msg = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
            prompts.append(user_msg)
        else:
            prompt = row.get("input", "") or str({k: v for k, v in row.items() if k != "target"})
            prompts.append(prompt)

    return {
        "eval_job_id": eval_job.id,
        "dataset_id": eval_job.dataset_id,
        "prompts": prompts
    }


class SubmitResponsesRequest(BaseModel):
    responses: List[str]
    latencies: Optional[List[int]] = None


@router.post("/{eval_id}/submit-responses")
async def submit_responses(
    eval_id: str,
    body: SubmitResponsesRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Enables Colab notebook to submit generated model responses and latencies back to the backend
    to compute metrics and complete the evaluation job.
    """
    # Verify evaluation job exists and belongs to user
    result = await db.execute(
        select(EvaluationJob).where(EvaluationJob.id == eval_id, EvaluationJob.user_id == user_id)
    )
    eval_job = result.scalar_one_or_none()
    if not eval_job:
        raise HTTPException(404, "Evaluation job not found")

    if eval_job.status == EvaluationStatus.completed:
        return {"status": "completed", "message": "Evaluation already completed", "metrics": eval_job.metrics}

    from app.services.evaluator import evaluate_model_with_responses
    try:
        res = await evaluate_model_with_responses(eval_id, body.responses, body.latencies)
        return res
    except Exception as e:
        raise HTTPException(500, f"Failed to run evaluation: {e}")
