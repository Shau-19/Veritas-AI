"""
PDF Service
Extracts text from uploaded PDF files and chunks it for fine-tuning dataset generation.

Design principles:
- Page-by-page extraction with pdfplumber (handles tables + text layout well)
- Sentence-boundary chunking to keep semantic context intact
- Image-only page detection and skipping
- Async-friendly: runs blocking PDF work in executor thread pool
"""
import re
import logging
import asyncio
from io import BytesIO

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 80
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:]
            if len(chunk) >= MIN_CHUNK_SIZE:
                chunks.append(chunk.strip())
            break

        boundary = end
        for sep in [". ", "? ", "! ", "\n\n", "\n"]:
            idx = text.rfind(sep, start, end)
            if idx != -1 and idx > start + chunk_size // 2:
                boundary = idx + len(sep)
                break

        chunk = text[start:boundary].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)
        start = boundary

    return chunks


def _extract_text_sync(pdf_bytes: bytes) -> dict:
    """
    Synchronous PDF text extraction (runs in thread pool).
    Returns dict with pages list and metadata.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

    pages_text = []
    total_chars = 0
    skipped_pages = 0

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if not text or len(text.strip()) < MIN_CHUNK_SIZE:
                    skipped_pages += 1
                    continue

                # Normalize whitespace
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]{2,}", " ", text)
                text = text.strip()

                pages_text.append({
                    "page": page_num + 1,
                    "text": text,
                    "chars": len(text),
                })
                total_chars += len(text)
            except Exception as e:
                logger.warning(f"[PDF] Error on page {page_num + 1}: {e}")
                skipped_pages += 1

    return {
        "total_pages": total_pages,
        "extracted_pages": len(pages_text),
        "skipped_pages": skipped_pages,
        "total_chars": total_chars,
        "pages": pages_text,
    }


async def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Async wrapper — runs pdfplumber in a thread pool to avoid blocking the event loop.

    Returns:
        {
            "total_pages": int,
            "extracted_pages": int,
            "skipped_pages": int,
            "total_chars": int,
            "chunks": list[str],
        }
    """
    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"PDF exceeds maximum size of 20MB ({len(pdf_bytes) // 1024 // 1024}MB)")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _extract_text_sync, pdf_bytes)

    # Build flat chunk list from all pages
    all_chunks: list[str] = []
    for page_data in result["pages"]:
        page_chunks = _chunk_text(page_data["text"])
        all_chunks.extend(page_chunks)

    return {
        "total_pages": result["total_pages"],
        "extracted_pages": result["extracted_pages"],
        "skipped_pages": result["skipped_pages"],
        "total_chars": result["total_chars"],
        "chunks": all_chunks,
    }
