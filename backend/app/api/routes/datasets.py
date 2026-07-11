from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.dataset import Dataset, DatasetStatus
from app.services.dataset_service import audit_dataset, estimate_training_cost

router = APIRouter(prefix="/datasets", tags=["datasets"])

ALLOWED_FORMATS = {"csv", "jsonl"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    row_count: Optional[int]
    file_format: Optional[str]
    file_size_bytes: Optional[int]
    quality_score: Optional[float]
    audit_results: Optional[dict]
    status: str
    created_at: str

    class Config:
        from_attributes = True


@router.post("/upload", status_code=201)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    # Validate format
    suffix = file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format. Use: {ALLOWED_FORMATS}")

    # Check free-tier dataset count limit if keyless
    x_key_configured = request.headers.get("x-key-configured")
    if x_key_configured != "true":
        result = await db.execute(
            select(Dataset).where(Dataset.user_id == user_id)
        )
        existing_count = len(result.scalars().all())
        if existing_count >= 1:
            raise HTTPException(
                400,
                "Free-tier limit reached: maximum 1 dataset allowed. Please configure your own API keys in Settings to remove this limit."
            )

    # Read content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Max 100 MB.")

    # Run quality audit (CPU, no GPU needed)
    audit = audit_dataset(content, suffix)

    # Parse rows to save them for download/preview
    import pandas as pd
    import io
    import json
    df = None
    try:
        if suffix == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif suffix == "jsonl":
            lines = content.decode("utf-8").strip().split("\n")
            records = [json.loads(line) for line in lines if line.strip()]
            df = pd.DataFrame(records)
    except Exception:
        pass

    rows_list = []
    if df is not None:
        rows_list = df.to_dict(orient="records")

    # Create DB record
    dataset = Dataset(
        user_id=user_id,
        name=name,
        description=description,
        file_size_bytes=len(content),
        file_format=suffix,
        row_count=audit.get("row_count"),
        quality_score=audit.get("quality_score"),
        audit_results={
            **audit,
            "rows": rows_list
        },
        status=DatasetStatus.ready if audit.get("quality_score", 0) > 0 else DatasetStatus.failed,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "status": dataset.status,
        "quality_score": dataset.quality_score,
        "audit_results": audit,
        "message": "Dataset uploaded and audited successfully"
    }
@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            row_count=d.row_count,
            file_format=d.file_format,
            file_size_bytes=d.file_size_bytes,
            quality_score=d.quality_score,
            audit_results=d.audit_results,
            status=d.status,
            created_at=str(d.created_at),
        )
        for d in datasets
    ]


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.post("/{dataset_id}/estimate")
async def estimate_cost(
    dataset_id: str,
    base_model: str,
    epochs: int = 3,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    estimate = estimate_training_cost(
        row_count=dataset.row_count or 0,
        base_model=base_model,
        epochs=epochs,
    )
    return estimate


@router.delete("/{dataset_id}", status_code=200)
async def delete_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    await db.delete(dataset)
    await db.commit()
    return {"message": "Dataset deleted successfully", "id": dataset_id}


@router.get("/{dataset_id}/raw")
async def get_dataset_raw(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Exposes a public download route for raw dataset rows (used by training pods)."""
    import json
    from fastapi.responses import PlainTextResponse
    
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
        
    rows = dataset.audit_results.get("rows", [])
    content = "\n".join(json.dumps(r) for r in rows)
    return PlainTextResponse(content, media_type="application/jsonl")


class UpdateDatasetRequest(BaseModel):
    name: Optional[str] = None
    rows: Optional[list[dict]] = None


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    body: UpdateDatasetRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    if body.name is not None:
        dataset.name = body.name

    if body.rows is not None:
        dataset.row_count = len(body.rows)
        # Update audit_results["rows"]
        audit_results = dict(dataset.audit_results or {})
        audit_results["rows"] = body.rows
        dataset.audit_results = audit_results
        
        # Calculate new size of dataset
        import json
        dataset.file_size_bytes = len(json.dumps(body.rows).encode("utf-8"))

    await db.commit()
    await db.refresh(dataset)
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        row_count=dataset.row_count,
        file_format=dataset.file_format,
        file_size_bytes=dataset.file_size_bytes,
        quality_score=dataset.quality_score,
        audit_results=dataset.audit_results,
        status=dataset.status,
        created_at=str(dataset.created_at),
    )

