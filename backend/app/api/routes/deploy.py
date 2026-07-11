from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List
import uuid
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.deployment import Deployment, DeploymentStatus
from app.models.job import TrainingJob, JobStatus
from app.services import sagemaker_service

router = APIRouter(prefix="/deploy", tags=["deploy"])


class DeployRequest(BaseModel):
    job_id: Optional[str] = None  # None if hosting a raw base model
    model_id: Optional[str] = None  # e.g. "microsoft/phi-3-mini-4k-instruct"
    instance_type: str = "ml.g5.xlarge"
    region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_role_arn: Optional[str] = None


class InvokeRequest(BaseModel):
    prompt: str


class DeploymentResponse(BaseModel):
    id: str
    job_id: Optional[str]
    model_id: str
    endpoint_name: str
    instance_type: str
    region: str
    status: str
    error_message: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


async def run_async_deploy(
    deployment_id: str,
    model_id: str,
    hf_token: Optional[str],
    req: DeployRequest,
    db_session_maker,
):
    """Background task to provision the SageMaker endpoint and track status."""
    endpoint_name = None
    try:
        # Resolve base model and lora adapter if it's a job
        base_model_id = model_id
        lora_adapter = None
        
        if req.job_id:
            async with db_session_maker() as db:
                result = await db.execute(select(TrainingJob).where(TrainingJob.id == req.job_id))
                job = result.scalar_one_or_none()
                if job:
                    base_model_id = job.base_model
                    lora_adapter = model_id

        # Resolve pre-generated endpoint name from DB record
        async with db_session_maker() as db:
            result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
            dep = result.scalar_one_or_none()
            if dep:
                endpoint_name = dep.endpoint_name

        # Call sagemaker_service to deploy
        endpoint_name = sagemaker_service.deploy_sagemaker_endpoint(
            model_id=base_model_id,
            hf_token=hf_token,
            access_key=req.aws_access_key,
            secret_key=req.aws_secret_key,
            region=req.region,
            role_arn=req.aws_role_arn,
            instance_type=req.instance_type,
            lora_adapter=lora_adapter,
            endpoint_name=endpoint_name,
        )

        # Update db (ensuring active state polling is aware of any backend changes)
        async with db_session_maker() as db:
            result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
            dep = result.scalar_one_or_none()
            if dep:
                dep.endpoint_name = endpoint_name
                await db.commit()

        # Poll status until active or failed
        import asyncio
        for _ in range(60):  # 10 minutes maximum (10s intervals)
            await asyncio.sleep(10)
            status = sagemaker_service.check_endpoint_status(
                endpoint_name=endpoint_name,
                access_key=req.aws_access_key,
                secret_key=req.aws_secret_key,
                region=req.region,
            )
            
            async with db_session_maker() as db:
                result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
                dep = result.scalar_one_or_none()
                if not dep:
                    break
                
                if status == "InService":
                    dep.status = DeploymentStatus.active
                    await db.commit()
                    break
                elif status in ["Failed", "OutOfService"]:
                    dep.status = DeploymentStatus.failed
                    dep.error_message = f"SageMaker endpoint status check returned: {status}"
                    await db.commit()
                    break

    except Exception as e:
        # If creation fails, update DB status to failed
        async with db_session_maker() as db:
            result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
            dep = result.scalar_one_or_none()
            if dep:
                dep.status = DeploymentStatus.failed
                dep.error_message = str(e)
                if endpoint_name:
                    dep.endpoint_name = endpoint_name
                await db.commit()


@router.post("/", response_model=DeploymentResponse, status_code=201)
async def create_deployment(
    body: DeployRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    target_model_id = ""
    hf_token = None

    if body.job_id:
        # Deploy fine-tuned model
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == body.job_id, TrainingJob.user_id == user_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(404, "Training job not found")
        if job.status != JobStatus.completed:
            raise HTTPException(400, "Cannot deploy a training job that is not completed")
        if not job.hf_model_id:
            raise HTTPException(400, "Completed job is missing a Hugging Face model repository ID")
        
        target_model_id = job.hf_model_id
        # Resolve target model ID and append latest Hugging Face commit SHA to bypass caching
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.get(f"https://huggingface.co/api/models/{job.hf_model_id}", timeout=5.0)
                if res.status_code == 200:
                    sha = res.json().get("sha")
                    if sha:
                        target_model_id = f"{job.hf_model_id}@{sha}"
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch latest Hugging Face SHA for {job.hf_model_id}: {e}")
    elif body.model_id:
        # Deploy standard base model
        target_model_id = body.model_id
    else:
        raise HTTPException(400, "Either 'job_id' or 'model_id' is required to deploy")

    # Pre-generate real SageMaker resource endpoint name
    from app.services.sagemaker_service import get_clean_aws_name
    short_id = target_model_id.split("/")[-1]
    aws_endpoint_name = get_clean_aws_name("ep", short_id)

    deployment = Deployment(
        user_id=user_id,
        job_id=body.job_id,
        model_id=target_model_id,
        endpoint_name=aws_endpoint_name,
        instance_type=body.instance_type,
        region=body.region,
        status=DeploymentStatus.deploying,
    )
    db.add(deployment)
    await db.commit()
    await db.refresh(deployment)

    # Dispatch SageMaker creation to background tasks
    from app.core.database import AsyncSessionLocal
    background_tasks.add_task(
        run_async_deploy,
        deployment_id=deployment.id,
        model_id=target_model_id,
        hf_token=hf_token,
        req=body,
        db_session_maker=AsyncSessionLocal,
    )

    return DeploymentResponse(
        id=deployment.id,
        job_id=deployment.job_id,
        model_id=deployment.model_id,
        endpoint_name=deployment.endpoint_name,
        instance_type=deployment.instance_type,
        region=deployment.region,
        status=deployment.status,
        error_message=deployment.error_message,
        created_at=str(deployment.created_at),
    )


@router.get("/", response_model=List[DeploymentResponse])
async def list_deployments(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Deployment)
        .where(Deployment.user_id == user_id)
        .order_by(Deployment.created_at.desc())
    )
    deps = result.scalars().all()
    return [
        DeploymentResponse(
            id=d.id,
            job_id=d.job_id,
            model_id=d.model_id,
            endpoint_name=d.endpoint_name,
            instance_type=d.instance_type,
            region=d.region,
            status=d.status,
            error_message=d.error_message,
            created_at=str(d.created_at),
        )
        for d in deps
    ]


@router.get("/{dep_id}", response_model=DeploymentResponse)
async def get_deployment(
    dep_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Deployment).where(Deployment.id == dep_id, Deployment.user_id == user_id)
    )
    dep = result.scalar_one_or_none()
    if not dep:
        raise HTTPException(404, "Deployment not found")
        

            
    return DeploymentResponse(
        id=dep.id,
        job_id=dep.job_id,
        model_id=dep.model_id,
        endpoint_name=dep.endpoint_name,
        instance_type=dep.instance_type,
        region=dep.region,
        status=dep.status,
        error_message=dep.error_message,
        created_at=str(dep.created_at),
    )


class TerminateRequest(BaseModel):
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None


@router.post("/{dep_id}/terminate")
async def terminate_deployment(
    dep_id: str,
    body: TerminateRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Deployment).where(Deployment.id == dep_id, Deployment.user_id == user_id)
    )
    dep = result.scalar_one_or_none()
    if not dep:
        raise HTTPException(404, "Deployment not found")

    if dep.status == DeploymentStatus.terminated:
        return {"message": "Endpoint is already terminated"}

    # Run sagemaker deletion
    if not dep.endpoint_name.startswith("provisioning"):
        try:
            sagemaker_service.terminate_sagemaker_endpoint(
                endpoint_name=dep.endpoint_name,
                access_key=body.aws_access_key,
                secret_key=body.aws_secret_key,
                region=dep.region,
            )
        except Exception as e:
            raise HTTPException(500, f"AWS deletion failed: {e}")

    from datetime import datetime, timezone
    dep.status = DeploymentStatus.terminated
    dep.terminated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"message": "SageMaker endpoint successfully deleted."}


class InvokeDeployRequest(BaseModel):
    prompt: str
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None


@router.post("/{dep_id}/invoke")
async def invoke_deployment(
    dep_id: str,
    body: InvokeDeployRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Deployment).where(Deployment.id == dep_id, Deployment.user_id == user_id)
    )
    dep = result.scalar_one_or_none()
    if not dep:
        raise HTTPException(404, "Deployment not found")

    if dep.status != DeploymentStatus.active:
        raise HTTPException(400, "Inference can only be run on active deployments.")

    try:
        adapter_id = dep.model_id if dep.job_id else None
        response_text = sagemaker_service.invoke_sagemaker_endpoint(
            endpoint_name=dep.endpoint_name,
            prompt=body.prompt,
            access_key=body.aws_access_key,
            secret_key=body.aws_secret_key,
            region=dep.region,
            adapter_id=adapter_id,
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(500, f"Inference invoke failed: {e}")


@router.delete("/{dep_id}")
async def delete_deployment(
    dep_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Deployment).where(Deployment.id == dep_id, Deployment.user_id == user_id)
    )
    dep = result.scalar_one_or_none()
    if not dep:
        raise HTTPException(404, "Deployment not found")

    # If active or deploying (and has a valid name), try to delete SageMaker resources
    if dep.status not in [DeploymentStatus.terminated, DeploymentStatus.failed] and not dep.endpoint_name.startswith("provisioning"):
        try:
            sagemaker_service.terminate_sagemaker_endpoint(
                endpoint_name=dep.endpoint_name,
                region=dep.region
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to delete SageMaker endpoint during deployment record deletion: {e}")

    await db.delete(dep)
    await db.commit()
    return {"message": "Deployment record successfully deleted"}
