"""
Scraper Service
Fetches and extracts clean text from web URLs for fine-tuning dataset generation.

Design principles:
- Server-rendered HTML only (no JS execution — keeps dependencies minimal)
- Smart content extraction: targets <article>, <main>, <p>, <section>
- Strips navigation, footer, scripts, ads, and UI chrome
- Chunked output (~1200 chars per chunk) for optimal restructuring
- Sitemap support for multi-page crawling
- Per-request timeout to prevent hanging on slow sites
"""
import re
import logging
import asyncio
from urllib.parse import urljoin, urlparse
from typing import AsyncGenerator
import httpx

logger = logging.getLogger(__name__)

# Tags whose content we always discard
DISCARD_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "input", "select", "textarea",
    "iframe", "meta", "link", "svg", "img", "figure",
    "advertisement", "banner", "cookie", "menu",
}

# Priority content tags (extracted first, in order)
CONTENT_TAGS = ["article", "main", "section", ".content", ".post", ".entry"]

CHUNK_SIZE = 1200   # chars per chunk — good balance for GPT prompts
MIN_CHUNK_SIZE = 80  # ignore chunks shorter than this
REQUEST_TIMEOUT = 20.0  # seconds per HTTP request


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from raw HTML without JS.
    Uses regex + tag stripping (no BS4 required as primary path,
    but uses BS4 if available for better accuracy).
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Remove all discard tags
        for tag in soup(list(DISCARD_TAGS)):
            tag.decompose()

        # Try to find primary content area
        content = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id=re.compile(r"content|main|article|post", re.I)) or
            soup.find(class_=re.compile(r"content|main|article|post|entry", re.I)) or
            soup.find("body") or
            soup
        )

        text = content.get_text(separator="\n", strip=True)
    except ImportError:
        # Fallback: naive regex HTML stripper
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-z]+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks at sentence boundaries.
    Prefers splitting on '. ', '? ', '! ', '\n\n'.
    """
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

        # Walk backwards to find a sentence boundary
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


async def _fetch_url(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch a single URL and return HTML content, or None on failure."""
    try:
        response = await client.get(
            url,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "html" in content_type or "text" in content_type:
                return response.text
            else:
                logger.warning(f"[Scraper] Non-HTML content at {url}: {content_type}")
        else:
            logger.warning(f"[Scraper] HTTP {response.status_code} at {url}")
    except httpx.TimeoutException:
        logger.warning(f"[Scraper] Timeout fetching {url}")
    except Exception as e:
        logger.warning(f"[Scraper] Error fetching {url}: {e}")
    return None


async def _parse_sitemap(client: httpx.AsyncClient, sitemap_url: str, max_pages: int) -> list[str]:
    """Parse XML sitemap and return up to max_pages page URLs."""
    html = await _fetch_url(client, sitemap_url)
    if not html:
        return []

    # Simple regex sitemap parser — handles standard XML sitemaps
    urls = re.findall(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", html)
    # Filter out sitemap index entries (they end in sitemap.xml)
    page_urls = [u for u in urls if not u.endswith("sitemap.xml") and not u.endswith("sitemap_index.xml")]
    logger.info(f"[Scraper] Sitemap found {len(page_urls)} pages at {sitemap_url}")
    return page_urls[:max_pages]


async def scrape_url(
    urls: list[str],
    max_pages: int = 10,
) -> AsyncGenerator[dict, None]:
    """
    Scrape a list of URLs (or sitemaps) and yield progress events + final chunks.

    Yields dicts with events:
        {"event": "page_scraped", "url": str, "chunks": int, "total_chunks": int}
        {"event": "complete", "chunks": list[str], "pages_scraped": int}
        {"event": "error", "message": str}
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; VeritasAI-Scraper/1.0; +https://veritasai.dev)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    all_chunks: list[str] = []
    pages_scraped = 0
    urls_to_scrape: list[str] = []
    seen_urls: set[str] = set()

    async with httpx.AsyncClient(headers=headers) as client:
        for url in urls:
            url = url.strip()
            if not url:
                continue
            # Detect if URL is a sitemap
            is_sitemap = url.endswith("sitemap.xml") or "sitemap" in url.lower()

            if is_sitemap:
                sitemap_urls = await _parse_sitemap(client, url, max_pages)
                for u in sitemap_urls:
                    if u not in seen_urls:
                        seen_urls.add(u)
                        urls_to_scrape.append(u)
            else:
                if url not in seen_urls:
                    seen_urls.add(url)
                    urls_to_scrape.append(url)
                # Also try to discover sitemap for this seed URL
                base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                sitemap_candidates = [
                    f"{base}/sitemap.xml",
                    f"{url.rstrip('/')}/sitemap.xml",
                ]
                for candidate in sitemap_candidates:
                    extra_pages = await _parse_sitemap(client, candidate, max_pages)
                    if extra_pages:
                        # Add discovered pages
                        for p in extra_pages:
                            if p != url and p not in seen_urls:
                                seen_urls.add(p)
                                urls_to_scrape.append(p)
                        logger.info(f"[Scraper] Auto-discovered {len(extra_pages)} pages from sitemap of {url}")
                        break

        if not urls_to_scrape:
            yield {"event": "error", "message": "No valid URLs found to scrape."}
            return

        logger.info(f"[Scraper] Scraping {len(urls_to_scrape)} pages")

        for page_url in urls_to_scrape[:max_pages]:
            html = await _fetch_url(client, page_url)
            if not html:
                continue

            text = _extract_text_from_html(html)
            if not text or len(text) < MIN_CHUNK_SIZE:
                logger.debug(f"[Scraper] No usable text at {page_url}")
                continue

            page_chunks = _chunk_text(text)
            all_chunks.extend(page_chunks)
            pages_scraped += 1

            yield {
                "event": "page_scraped",
                "url": page_url,
                "chunks": len(page_chunks),
                "total_chunks": len(all_chunks),
                "pages_done": pages_scraped,
            }

            # Small delay between pages to be polite
            await asyncio.sleep(0.5)

    if not all_chunks:
        yield {"event": "error", "message": "No readable text could be extracted from the provided URL(s)."}
        return

    yield {
        "event": "complete",
        "chunks": all_chunks,
        "pages_scraped": pages_scraped,
        "total_chunks": len(all_chunks),
    }
