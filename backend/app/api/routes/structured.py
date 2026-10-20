"""
Structured Dataset Generation Routes
POST /api/v1/structured/generate  — generate multi-column tabular dataset
POST /api/v1/structured/schema    — AI suggests schema from description
POST /api/v1/generate/repair      — repair existing dataset
"""
import json
import csv
import io
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import Optional
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.dataset import Dataset, DatasetStatus
from app.services.structured_generator import generate_structured_stream, audit_structured_batch
from app.services.repair_engine import repair_dataset, compute_entropy_metrics, detect_weak_rows

router = APIRouter(prefix="/structured", tags=["structured"])


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


class ColumnSpec(BaseModel):
    name: str
    type: str = "string"         # string | int | float | boolean | category
    description: str = ""
    range: Optional[str] = None  # e.g. "1-100" or "0.0-1.0"
    values: Optional[list] = None  # for categorical columns
    # Phase 2 Tabular Prior: Allows user to specify column statistical distributions for synthetic variance
    distribution_prior: Optional[str] = None  # normal | log_normal | poisson | exponential | uniform


class StructuredGenerateRequest(BaseModel):
    description: str             # "House price dataset for Mumbai real estate"
    columns: list[ColumnSpec]
    total_rows: int = 200
    domain_context: str = ""
    dataset_name: str = ""
    # Phase 2 Noise Simulation: Ratio of simulated null inputs and labels to corrupt/mimic real-world missingness and labeling noise
    missingness_rate: float = 0.0
    label_noise_rate: float = 0.0
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class SchemaRequest(BaseModel):
    description: str             # User describes what they want
    num_columns: int = 8


class RepairRequest(BaseModel):
    dataset_rows: list[dict]
    topic: str
    task_type: str = "chat"
    style: str = "casual"
    quality_threshold: float = 90.0
    max_iterations: int = 3


@router.post("/schema")
async def suggest_schema(
    body: SchemaRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    AI suggests a schema (columns + types) from a plain description.
    User can then edit it before generating.
    """
    from app.services.llm_client import get_llm_client
    client, synthesis_model = get_llm_client()

    prompt = f"""A user wants to create a synthetic ML dataset.
Description: "{body.description}"
Suggest exactly {body.num_columns} columns for this dataset.

Return a JSON object:
{{
  "dataset_name": "suggested name",
  "description": "one line description",
  "ml_task": "regression|classification|clustering|nlp",
  "target_column": "which column is the prediction target",
  "columns": [
    {{
      "name": "column_name",
      "type": "string|int|float|boolean|category",
      "description": "what this column represents",
      "range": "min-max for numeric (optional)",
      "values": ["val1","val2"] for categorical (optional),
      "distribution_prior": "normal|log_normal|poisson|exponential|uniform (optional)"
    }}
  ]
}}

Make columns realistic and useful for ML. Include the target column last."""

    response = await client.chat.completions.create(
        model=synthesis_model,
        messages=[
            {"role": "system", "content": "You are a data scientist. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    try:
        client._client.cookies.clear()
    except Exception:
        pass

    schema = json.loads(response.choices[0].message.content)
    return schema


@router.post("/generate")
async def generate_structured(
    request: Request,
    body: StructuredGenerateRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Generate a multi-column structured/tabular dataset."""
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
    if not 1 <= len(body.columns) <= 20:
        raise HTTPException(400, "columns must be between 1 and 20")

    columns_dicts = [c.model_dump() for c in body.columns]
    all_rows = []
    audit_result = {}

    async for event in generate_structured_stream(
        description=body.description,
        columns=columns_dicts,
        total_rows=body.total_rows,
        domain_context=body.domain_context,
        missingness_rate=body.missingness_rate,
        label_noise_rate=body.label_noise_rate,
        openai_api_key=body.openai_api_key,
        deepseek_api_key=body.deepseek_api_key,
        groq_api_key=body.groq_api_key,
    ):
        if event["event"] == "complete":
            all_rows = event.get("rows", [])
            audit_result = event.get("audit", {})

    if not all_rows:
        raise HTTPException(500, "Generation failed — no rows produced")

    # Convert to CSV
    if all_rows:
        col_names = list(all_rows[0].keys())
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(all_rows)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
    else:
        csv_bytes = b""

    dataset_name = body.dataset_name or body.description[:50]
    col_names = [c.name for c in body.columns]

    dataset = Dataset(
        user_id=user_id,
        name=dataset_name,
        description=f"Structured | {body.description} | {len(body.columns)} columns",
        file_size_bytes=len(csv_bytes),
        file_format="csv",
        row_count=len(all_rows),
        quality_score=audit_result.get("quality_score", 0),
        audit_results={
            **audit_result,
            "rows": all_rows,  # store rows for download
            "dataset_type": "structured",
            "columns": col_names,
            "generation_config": {
                "description": body.description,
                "total_rows": body.total_rows,
                "domain_context": body.domain_context,
            }
        },
        status=DatasetStatus.ready,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset_name,
        "row_count": len(all_rows),
        "columns": col_names,
        "quality_score": audit_result.get("quality_score"),
        "audit": audit_result,
        "sample_rows": all_rows[:5],
        "rows": all_rows,  # return all rows to frontend for immediate download
        "message": f"Generated {len(all_rows)} rows with {len(col_names)} columns",
    }


@router.post("/generate/stream")
async def generate_structured_stream_route(
    request: Request,
    body: StructuredGenerateRequest,
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
    if not 1 <= len(body.columns) <= 20:
        raise HTTPException(400, "columns must be between 1 and 20")

    columns_dicts = [c.model_dump() for c in body.columns]

    async def gated_generator_func():
        all_rows = []
        audit_result = {}
        try:
            async for event in generate_structured_stream(
                description=body.description,
                columns=columns_dicts,
                total_rows=body.total_rows,
                domain_context=body.domain_context,
                missingness_rate=body.missingness_rate,
                label_noise_rate=body.label_noise_rate,
                openai_api_key=body.openai_api_key,
                deepseek_api_key=body.deepseek_api_key,
                groq_api_key=body.groq_api_key,
            ):
                if event.get("event") == "complete":
                    all_rows = event.get("rows", [])
                    audit_result = event.get("audit", {})

                    # Convert to CSV
                    col_names = [c.name for c in body.columns]
                    csv_buffer = io.StringIO()
                    if all_rows:
                        writer = csv.DictWriter(csv_buffer, fieldnames=list(all_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(all_rows)
                    csv_bytes = csv_buffer.getvalue().encode("utf-8")

                    dataset_name = body.dataset_name or "Tabular Dataset"
                    dataset = Dataset(
                        user_id=user_id,
                        name=dataset_name,
                        description=f"Structured | {body.description} | {len(body.columns)} columns",
                        file_size_bytes=len(csv_bytes),
                        file_format="csv",
                        row_count=len(all_rows),
                        quality_score=audit_result.get("synthetic_realism_score", 0),
                        audit_results={
                            **audit_result,
                            "rows": all_rows,
                            "dataset_type": "structured",
                            "columns": col_names,
                            "generation_config": {
                                "description": body.description,
                                "total_rows": body.total_rows,
                                "domain_context": body.domain_context,
                            }
                        },
                        status=DatasetStatus.ready,
                    )
                    db.add(dataset)
                    await db.commit()
                    await db.refresh(dataset)

                    event["id"] = dataset.id
                    event["name"] = dataset_name
                    event["row_count"] = body.total_rows
                    event["status"] = "ready"
                    event["message"] = f"Generated {body.total_rows} tabular rows successfully"
                    event["quality_score"] = audit_result.get("synthetic_realism_score", 0)

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
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


@router.post("/repair")
async def repair_endpoint(
    body: RepairRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Repair a dataset iteratively.
    Detects weak rows, regenerates them, repeats until quality threshold met.
    Returns the repaired dataset + entropy metrics.
    """
    if not body.dataset_rows:
        raise HTTPException(400, "dataset_rows cannot be empty")
    if len(body.dataset_rows) > 5000:
        raise HTTPException(400, "Max 5000 rows for repair")

    final_rows = []
    final_entropy = {}
    iterations_done = 0

    async for event in repair_dataset(
        rows=body.dataset_rows,
        topic=body.topic,
        task_type=body.task_type,
        style=body.style,
        quality_threshold=body.quality_threshold,
        max_iterations=body.max_iterations,
    ):
        if event["event"] == "repair_complete":
            final_rows = event.get("rows", [])
            final_entropy = event.get("entropy", {})
            iterations_done = event.get("iterations", 0)

    # Final weak row count
    remaining_weak = detect_weak_rows(final_rows, body.task_type)

    return {
        "repaired_rows": final_rows,
        "original_count": len(body.dataset_rows),
        "final_count": len(final_rows),
        "iterations_performed": iterations_done,
        "weak_rows_remaining": len(remaining_weak),
        "entropy_metrics": final_entropy,
        "sample_rows": final_rows[:3],
        "message": f"Repair complete after {iterations_done} iterations. {len(remaining_weak)} weak rows remaining.",
    }


@router.post("/entropy")
async def check_entropy(
    body: dict,
    user_id: str = Depends(get_current_user_id),
):
    """Check entropy/diversity metrics for any dataset rows."""
    rows = body.get("rows", [])
    if not rows:
        raise HTTPException(400, "rows required")
    return compute_entropy_metrics(rows)
