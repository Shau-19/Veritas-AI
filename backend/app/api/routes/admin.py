from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any, List
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.user import User
from app.models.dataset import Dataset
from app.models.job import TrainingJob, JobStatus
from app.models.evaluation import EvaluationJob

router = APIRouter(prefix="/admin", tags=["admin"])


async def get_current_admin(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required."
        )
    return user


@router.get("/stats")
async def get_admin_stats(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    # 1. Total summary stats
    users_res = await db.execute(select(func.count(User.id)))
    total_users = users_res.scalar_one()

    datasets_res = await db.execute(select(func.count(Dataset.id)))
    total_datasets = datasets_res.scalar_one()

    jobs_res = await db.execute(select(TrainingJob))
    all_jobs = jobs_res.scalars().all()

    total_jobs = len(all_jobs)
    total_cost = sum(j.actual_cost_usd or 0.0 for j in all_jobs)
    total_gpu_sec = sum(j.gpu_seconds_used or 0 for j in all_jobs)
    active_jobs = sum(1 for j in all_jobs if j.status in [JobStatus.training, JobStatus.provisioning, JobStatus.uploading])

    # 2. Detailed user breakdowns
    user_list_res = await db.execute(select(User))
    all_users = user_list_res.scalars().all()

    user_breakdowns = []
    for u in all_users:
        u_datasets_res = await db.execute(select(func.count(Dataset.id)).where(Dataset.user_id == u.id))
        u_datasets = u_datasets_res.scalar_one()

        u_jobs_res = await db.execute(select(TrainingJob).where(TrainingJob.user_id == u.id))
        u_jobs = u_jobs_res.scalars().all()

        u_cost = sum(j.actual_cost_usd or 0.0 for j in u_jobs)
        u_gpu_sec = sum(j.gpu_seconds_used or 0 for j in u_jobs)

        user_breakdowns.append({
            "id": u.id,
            "email": u.email,
            "full_name": u.full_name,
            "is_active": u.is_active,
            "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "datasets_count": u_datasets,
            "jobs_count": len(u_jobs),
            "total_spent_usd": u_cost,
            "total_gpu_hours": round(u_gpu_sec / 3600.0, 2),
        })

    return {
        "summary": {
            "total_users": total_users,
            "total_datasets": total_datasets,
            "total_jobs": total_jobs,
            "total_cost_usd": round(total_cost, 4),
            "total_gpu_hours": round(total_gpu_sec / 3600.0, 2),
            "active_jobs_count": active_jobs,
        },
        "users": user_breakdowns,
    }
