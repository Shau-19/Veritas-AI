"""
Ingest Router
Three new endpoints for Dataset Studio Phase 2:
  POST /ingest/scrape  — Web URL scraping → dataset
  POST /ingest/pdf     — PDF file upload → dataset
  POST /ingest/mcp     — MCP server resource → dataset
  GET  /ingest/mcp/resources — List available MCP resources

All three pipelines:
1. Extract raw text (via scraper / pdf / mcp service)
2. Restructure chunks to {input, output} pairs (restructurer_service)
3. Save to Dataset DB with source metadata
4. Stream progress events via SSE
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, HttpUrl
from typing import Optional

from app.core.database import get_db, AsyncSessionLocal
from app.core.security import get_current_user_id
from app.core.url_safety import assert_public_url
from app.models.dataset import Dataset, DatasetStatus
from app.services.restructurer_service import restructure_chunks_to_pairs, extract_tabular_from_chunks
from app.services.scraper_service import scrape_url
from app.services.pdf_service import extract_text_from_pdf
from app.services.mcp_client_service import (
    list_mcp_resources,
    read_mcp_resources,
    MCPClientError,
)
from app.api.routes.structured import ColumnSpec
from app.services.structured_generator import audit_structured_batch

logger = logging.getLogger(__name__)


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
router = APIRouter(prefix="/ingest", tags=["ingest"])

VALID_TASK_TYPES = ["chat", "instruction", "qa", "classification", "summarization"]
VALID_STYLES = ["formal", "casual", "technical", "simple"]


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str
    task_type: str = "qa"
    style: str = "formal"
    max_pages: int = 10
    pairs_per_chunk: int = 3
    dataset_name: str = ""


class MCPIngestRequest(BaseModel):
    server_url: str
    auth_token: Optional[str] = None
    resource_uris: list[str]
    task_type: str = "qa"
    style: str = "formal"
    pairs_per_chunk: int = 3
    dataset_name: str = ""


class MCPResourcesRequest(BaseModel):
    server_url: str
    auth_token: Optional[str] = None


# ─── Helper: Save Dataset to DB ──────────────────────────────────────────────

async def _save_dataset(
    user_id: str,
    name: str,
    description: str,
    rows: list[dict],
    source_meta: dict,
) -> Dataset:
    async with AsyncSessionLocal() as db:
        dataset = Dataset(
            user_id=user_id,
            name=name,
            description=description,
            file_size_bytes=len(json.dumps(rows).encode()),
            file_format="jsonl",
            row_count=len(rows),
            quality_score=None,  # No quality score for real-data ingestion (different metrics)
            audit_results={
                "rows": rows,
                "source": source_meta,
                "generation_config": source_meta,
            },
            status=DatasetStatus.ready,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset


class ScrapeTabularRequest(BaseModel):
    url: str
    columns: list[ColumnSpec]
    max_pages: int = 10
    dataset_name: Optional[str] = None
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


async def _save_structured_dataset(
    user_id: str,
    name: str,
    description: str,
    rows: list[dict],
    columns: list[str],
    audit_result: dict,
    source_meta: dict,
) -> Dataset:
    import csv
    import io
    csv_bytes = b""
    if rows and columns:
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            clean_row = {k: v for k, v in r.items() if k != "_provenance"}
            writer.writerow(clean_row)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

    async with AsyncSessionLocal() as db:
        dataset = Dataset(
            user_id=user_id,
            name=name,
            description=description,
            file_size_bytes=len(csv_bytes),
            file_format="csv",
            row_count=len(rows),
            quality_score=audit_result.get("quality_score", 0.0),
            audit_results={
                **audit_result,
                "rows": rows,
                "dataset_type": "structured",
                "columns": columns,
                "generation_config": source_meta,
            },
            status=DatasetStatus.ready,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset


# ─── Endpoint 1: Web Scraping ─────────────────────────────────────────────────

@router.post("/scrape")
async def ingest_from_url(
    body: ScrapeRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Scrape a URL and convert extracted text to fine-tuning pairs.
    """
    # Parse URLs (separated by newlines or commas)
    raw_urls = body.url.replace(",", "\n").split("\n")
    urls = [u.strip() for u in raw_urls if u.strip()]

    if not urls:
        raise HTTPException(400, "At least one valid URL must be provided")

    for url in urls:
        assert_public_url(url)

    if body.task_type not in VALID_TASK_TYPES:
        raise HTTPException(400, f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}")
    if body.style not in VALID_STYLES:
        raise HTTPException(400, f"Invalid style. Must be one of: {VALID_STYLES}")
    if not 1 <= body.max_pages <= 50:
        raise HTTPException(400, "max_pages must be between 1 and 50")
    if not 1 <= body.pairs_per_chunk <= 10:
        raise HTTPException(400, "pairs_per_chunk must be between 1 and 10")

    dataset_name = body.dataset_name or f"Scraped: {body.url[:50]}"
    description = f"Web scrape | {body.url} | {body.task_type} | {body.style}"

    async def event_stream():
        all_chunks = []

        yield {"event": "start", "message": f"Starting scrape of {len(urls)} URLs..."}

        # Phase 1: Scraping
        async for scrape_event in scrape_url(urls, max_pages=body.max_pages):
            if scrape_event["event"] == "page_scraped":
                all_chunks = []  # will be rebuilt on complete event
                yield {"event": "page_scraped", "url": scrape_event['url'], 'chunks_found': scrape_event['chunks'], 'total_chunks': scrape_event['total_chunks'], 'pages_done': scrape_event['pages_done']}

            elif scrape_event["event"] == "complete":
                all_chunks = scrape_event["chunks"]
                yield {"event": "scrape_complete", "total_chunks": len(all_chunks), "pages_scraped": scrape_event['pages_scraped']}

            elif scrape_event["event"] == "error":
                yield {"event": "error", "message": scrape_event['message']}
                return

        if not all_chunks:
            yield {"event": "error", "message": "No content could be extracted from the URL."}
            return

        # Phase 2: Restructuring
        yield {"event": "restructuring", "message": f"Converting {len(all_chunks)} text chunks into training pairs..."}

        progress_queue = asyncio.Queue()

        async def on_progress(chunk_idx: int, completed_chunks: int, num_pairs: int, total_pairs_so_far: int, reasoning: str, elapsed: float):
            await progress_queue.put({
                "event": "restructure_progress",
                "chunk_idx": chunk_idx,
                "completed_chunks": completed_chunks,
                "total_chunks": len(all_chunks),
                "pairs_generated": num_pairs,
                "total_pairs": total_pairs_so_far,
                "reasoning": reasoning,
                "elapsed": elapsed,
            })

        # Start restructuring task in background
        restructure_task = asyncio.create_task(
            restructure_chunks_to_pairs(
                chunks=all_chunks,
                task_type=body.task_type,
                style=body.style,
                pairs_per_chunk=body.pairs_per_chunk,
                on_progress=on_progress
            )
        )

        # Poll progress queue and yield updates
        while not restructure_task.done() or not progress_queue.empty():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                yield event
                progress_queue.task_done()
            except asyncio.TimeoutError:
                continue

        try:
            pairs = await restructure_task
        except Exception as e:
            yield {"event": "error", "message": f"Restructuring failed: {str(e)}"}
            return

        if not pairs:
            yield {"event": "error", "message": "No training pairs could be generated from the scraped content."}
            return

        clean_pairs = [{"input": p["input"], "output": p["output"]} for p in pairs]
        yield {"event": "complete", "row_count": len(clean_pairs), "sample_rows": clean_pairs[:3], "rows": clean_pairs, "message": f"Generated {len(clean_pairs)} training pairs from {body.url}"}

    async def gated_event_stream():
        try:
            async for event in event_stream():
                if event.get("event") == "complete":
                    clean_pairs = event.get("rows", [])
                    async with AsyncSessionLocal() as db_session:
                        dataset = Dataset(
                            user_id=user_id,
                            name=dataset_name,
                            description=description,
                            file_size_bytes=len(json.dumps(clean_pairs).encode()),
                            file_format="jsonl",
                            row_count=len(clean_pairs),
                            quality_score=None,
                            audit_results={
                                "rows": clean_pairs,
                                "source": {
                                    "source_type": "web_scrape",
                                    "url": body.url,
                                    "task_type": body.task_type,
                                    "style": body.style,
                                    "max_pages": body.max_pages,
                                    "pairs_per_chunk": body.pairs_per_chunk,
                                },
                                "generation_config": {
                                    "source_type": "web_scrape",
                                    "url": body.url,
                                    "task_type": body.task_type,
                                    "style": body.style,
                                    "max_pages": body.max_pages,
                                    "pairs_per_chunk": body.pairs_per_chunk,
                                },
                            },
                            status=DatasetStatus.ready,
                        )
                        db_session.add(dataset)
                        await db_session.commit()
                        await db_session.refresh(dataset)

                        event["id"] = dataset.id
                        event["name"] = dataset.name

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Error in scraper stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        sse_keepalive_generator(gated_event_stream()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ─── Endpoint 2: PDF Upload ────────────────────────────────────────────────────

@router.post("/pdf")
async def ingest_from_pdf(
    task_type: str = Form("qa"),
    style: str = Form("formal"),
    pairs_per_chunk: int = Form(3),
    dataset_name: str = Form(""),
    pdf_file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload a PDF and convert extracted text to fine-tuning pairs.
    Returns JSON (non-streaming) since PDF processing is fast.
    """
    if task_type not in VALID_TASK_TYPES:
        raise HTTPException(400, f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}")
    if style not in VALID_STYLES:
        raise HTTPException(400, f"Invalid style. Must be one of: {VALID_STYLES}")
    if not 1 <= pairs_per_chunk <= 10:
        raise HTTPException(400, "pairs_per_chunk must be between 1 and 10")

    # Validate file type
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    pdf_bytes = await pdf_file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    # Phase 1: Extract text
    try:
        extraction = await extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    if not extraction["chunks"]:
        raise HTTPException(422, "No readable text could be extracted from the PDF. It may be scanned/image-only.")

    # Phase 2: Restructure
    try:
        pairs = await restructure_chunks_to_pairs(
            chunks=extraction["chunks"],
            task_type=task_type,
            style=style,
            pairs_per_chunk=pairs_per_chunk,
        )
    except Exception as e:
        raise HTTPException(500, f"Restructuring failed: {str(e)}")

    if not pairs:
        raise HTTPException(422, "No training pairs could be generated from the PDF content.")

    # Phase 3: Save
    clean_pairs = [{"input": p["input"], "output": p["output"]} for p in pairs]
    name = dataset_name or f"PDF: {pdf_file.filename[:50]}"
    description = f"PDF ingestion | {pdf_file.filename} | {task_type} | {style}"

    try:
        dataset = await _save_dataset(
            user_id=user_id,
            name=name,
            description=description,
            rows=clean_pairs,
            source_meta={
                "source_type": "pdf",
                "filename": pdf_file.filename,
                "total_pages": extraction["total_pages"],
                "extracted_pages": extraction["extracted_pages"],
                "skipped_pages": extraction["skipped_pages"],
                "task_type": task_type,
                "style": style,
                "pairs_per_chunk": pairs_per_chunk,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Database save failed: {str(e)}")

    return {
        "id": dataset.id,
        "name": name,
        "row_count": len(clean_pairs),
        "pdf_pages": extraction["total_pages"],
        "extracted_pages": extraction["extracted_pages"],
        "total_chunks": len(extraction["chunks"]),
        "sample_rows": clean_pairs[:3],
        "rows": clean_pairs,
        "status": "ready",
        "message": f"Generated {len(clean_pairs)} training pairs from {extraction['extracted_pages']} PDF pages",
    }


# ─── Endpoint 3a: List MCP Resources ─────────────────────────────────────────

@router.post("/mcp/resources")
async def get_mcp_resources(
    body: MCPResourcesRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Connect to an MCP server and return its available resources.
    Use this to browse what's available before ingesting.
    """
    assert_public_url(body.server_url)

    try:
        resources = await list_mcp_resources(
            server_url=body.server_url,
            auth_token=body.auth_token,
        )
    except MCPClientError as e:
        raise HTTPException(502, f"Could not connect to MCP server: {str(e)}")

    return {
        "server_url": body.server_url,
        "resource_count": len(resources),
        "resources": resources,
    }


# ─── Endpoint 3b: Ingest from MCP ─────────────────────────────────────────────

@router.post("/mcp/ingest")
async def ingest_from_mcp(
    body: MCPIngestRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Read selected resources from an MCP server and convert to fine-tuning pairs.
    """
    if body.task_type not in VALID_TASK_TYPES:
        raise HTTPException(400, f"Invalid task_type. Must be one of: {VALID_TASK_TYPES}")
    if body.style not in VALID_STYLES:
        raise HTTPException(400, f"Invalid style. Must be one of: {VALID_STYLES}")
    if not body.resource_uris:
        raise HTTPException(400, "At least one resource_uri must be provided")
    if len(body.resource_uris) > 50:
        raise HTTPException(400, "Cannot ingest more than 50 resources at once")
    assert_public_url(body.server_url)

    # Phase 1: Read resources
    try:
        texts = await read_mcp_resources(
            server_url=body.server_url,
            resource_uris=body.resource_uris,
            auth_token=body.auth_token,
        )
    except MCPClientError as e:
        raise HTTPException(502, f"MCP server error: {str(e)}")

    if not texts:
        raise HTTPException(422, "No readable text content found in selected MCP resources.")

    # Phase 2: Restructure
    try:
        pairs = await restructure_chunks_to_pairs(
            chunks=texts,
            task_type=body.task_type,
            style=body.style,
            pairs_per_chunk=body.pairs_per_chunk,
        )
    except Exception as e:
        raise HTTPException(500, f"Restructuring failed: {str(e)}")

    if not pairs:
        raise HTTPException(422, "No training pairs could be generated from the MCP resources.")

    # Phase 3: Save
    clean_pairs = [{"input": p["input"], "output": p["output"]} for p in pairs]
    name = body.dataset_name or f"MCP: {body.server_url[:40]}"
    description = f"MCP ingestion | {body.server_url} | {body.task_type} | {body.style} | {len(body.resource_uris)} resources"

    try:
        dataset = await _save_dataset(
            user_id=user_id,
            name=name,
            description=description,
            rows=clean_pairs,
            source_meta={
                "source_type": "mcp",
                "server_url": body.server_url,
                "resource_count": len(body.resource_uris),
                "resource_uris": body.resource_uris,
                "task_type": body.task_type,
                "style": body.style,
                "pairs_per_chunk": body.pairs_per_chunk,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Database save failed: {str(e)}")

    return {
        "id": dataset.id,
        "name": name,
        "row_count": len(clean_pairs),
        "resources_read": len(body.resource_uris),
        "text_segments": len(texts),
        "sample_rows": clean_pairs[:3],
        "rows": clean_pairs,
        "status": "ready",
        "message": f"Generated {len(clean_pairs)} training pairs from {len(body.resource_uris)} MCP resources",
    }


@router.post("/scrape/tabular")
async def ingest_tabular_from_url(
    body: ScrapeTabularRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Scrape a URL and extract structured tabular data based on a specified list of columns.
    """
    raw_urls = body.url.replace(",", "\n").split("\n")
    urls = [u.strip() for u in raw_urls if u.strip()]

    if not urls:
        raise HTTPException(400, "At least one valid URL must be provided")

    for url in urls:
        assert_public_url(url)

    if not 1 <= body.max_pages <= 50:
        raise HTTPException(400, "max_pages must be between 1 and 50")

    if not body.columns:
        raise HTTPException(400, "At least one column spec must be provided")

    dataset_name = body.dataset_name or f"Tabular: {body.url[:50]}"
    col_names = [c.name for c in body.columns]
    description = f"Scrape-to-Table | {body.url} | {len(body.columns)} columns"

    # Enforce free-tier limit if BYOK keys are absent
    has_custom_keys = bool(
        body.openai_api_key or
        body.deepseek_api_key or
        body.groq_api_key
    )

    async def event_stream():
        all_chunks = []

        yield {"event": "start", "message": f"Starting scrape of {len(urls)} URLs..."}

        # Phase 1: Scraping
        async for scrape_event in scrape_url(urls, max_pages=body.max_pages):
            if scrape_event["event"] == "page_scraped":
                all_chunks = []  # will be rebuilt on complete event
                yield {
                    "event": "page_scraped",
                    "url": scrape_event['url'],
                    'chunks_found': scrape_event['chunks'],
                    'total_chunks': scrape_event['total_chunks'],
                    'pages_done': scrape_event['pages_done']
                }
            elif scrape_event["event"] == "complete":
                all_chunks = scrape_event["chunks"]
                yield {
                    "event": "scrape_complete",
                    "total_chunks": len(all_chunks),
                    "pages_scraped": scrape_event['pages_scraped']
                }
            elif scrape_event["event"] == "error":
                yield {"event": "error", "message": scrape_event['message']}
                return

        if not all_chunks:
            yield {"event": "error", "message": "No content could be extracted from the URL."}
            return

        # Phase 2: Tabular Extraction
        yield {"event": "restructuring", "message": f"Extracting tabular data from {len(all_chunks)} text chunks..."}

        progress_queue = asyncio.Queue()

        async def on_progress(chunk_idx: int, completed_chunks: int, num_rows: int, total_rows_so_far: int, elapsed: float):
            await progress_queue.put({
                "event": "extract_progress",
                "chunk_idx": chunk_idx,
                "completed_chunks": completed_chunks,
                "total_chunks": len(all_chunks),
                "rows_extracted": num_rows,
                "total_rows": total_rows_so_far,
                "elapsed": elapsed,
            })

        # Columns schema representation
        columns_schema = [c.model_dump() for c in body.columns]

        # Start extraction task in background
        extract_task = asyncio.create_task(
            extract_tabular_from_chunks(
                chunks=all_chunks,
                columns=columns_schema,
                openai_api_key=body.openai_api_key,
                deepseek_api_key=body.deepseek_api_key,
                groq_api_key=body.groq_api_key,
                on_progress=on_progress
            )
        )

        # Poll progress queue and yield updates
        while not extract_task.done() or not progress_queue.empty():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                yield event
                progress_queue.task_done()
            except asyncio.TimeoutError:
                continue

        try:
            extracted_rows = await extract_task
        except Exception as e:
            yield {"event": "error", "message": f"Extraction failed: {str(e)}"}
            return

        if not extracted_rows:
            yield {"event": "error", "message": "No matching tabular data could be extracted from the content."}
            return

        # Phase 3: Auditing
        yield {"event": "auditing", "message": "Auditing quality and structure of extracted rows..."}
        audit_result = audit_structured_batch(extracted_rows, columns_schema)

        yield {
            "event": "complete",
            "row_count": len(extracted_rows),
            "sample_rows": extracted_rows[:3],
            "rows": extracted_rows,
            "audit": audit_result,
            "message": f"Successfully extracted {len(extracted_rows)} structured rows from {body.url}"
        }

    async def gated_event_stream():
        try:
            async for event in event_stream():
                if event.get("event") == "complete":
                    clean_rows = event.get("rows", [])
                    audit_res = event.get("audit", {})
                    columns_schema = [c.model_dump() for c in body.columns]
                    
                    async with AsyncSessionLocal() as db_session:
                        # Apply free-tier count limit check if not BYOK
                        if not has_custom_keys:
                            result = await db_session.execute(
                                select(Dataset).where(Dataset.user_id == user_id)
                            )
                            existing_count = len(result.scalars().all())
                            if existing_count >= 1:
                                yield {"event": "error", "message": "Free-tier limit reached: maximum 1 dataset allowed. Configure API keys in Settings to remove limit."}
                                return
                            if len(clean_rows) > 20:
                                yield {"event": "error", "message": "Free-tier limit reached: maximum 20 extracted rows allowed on free tier. Configure API keys in Settings to remove limit."}
                                return

                        # Save dataset
                        dataset = await _save_structured_dataset(
                            user_id=user_id,
                            name=dataset_name,
                            description=description,
                            rows=clean_rows,
                            columns=col_names,
                            audit_result=audit_res,
                            source_meta={
                                "source_type": "scrape_tabular",
                                "url": body.url,
                                "max_pages": body.max_pages,
                                "columns": columns_schema,
                            }
                        )
                        # Inject saved dataset ID into the complete event
                        event["id"] = dataset.id

                # Send standard SSE string
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as err:
            logger.error(f"[TabularIngest] Streaming failed: {err}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(err)})}\n\n"

    return StreamingResponse(
        gated_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )
