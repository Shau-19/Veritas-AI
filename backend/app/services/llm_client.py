import httpx
import logging
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.local_embedder import LocalEmbedder

logger = logging.getLogger(__name__)

# Groq's free-tier models (fastest to slowest, best quality first)
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def is_empty_key(k: str) -> bool:
    return not k or "placeholder" in k or k.strip() in ("", "your-deepseek-api-key", "your-openai-api-key", "your-groq-api-key")

def _make_client(api_key: str, base_url: str = None) -> AsyncOpenAI:
    """Helper to create an AsyncOpenAI-compatible client."""
    kwargs = dict(
        api_key=api_key,
        http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
    )
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)

def get_llm_client(
    openai_key: str = None,
    deepseek_key: str = None,
    groq_key: str = None,
) -> tuple[AsyncOpenAI, str]:
    """
    Returns the appropriate AsyncOpenAI-compatible client and model name.

    Priority order (BYOK mode — user-provided keys):
      1. Groq     (free tier, llama-3.3-70b-versatile)
      2. DeepSeek (paid, deepseek-chat)
      3. OpenAI   (paid, gpt-4o-mini)

    If no user keys are given, falls back to server-side .env keys
    in the same priority order.
    """
    has_user_groq     = groq_key and not is_empty_key(groq_key)
    has_user_deepseek = deepseek_key and not is_empty_key(deepseek_key)
    has_user_openai   = openai_key and not is_empty_key(openai_key)

    # ── BYOK Mode ──────────────────────────────────────────────────────────
    if has_user_groq or has_user_deepseek or has_user_openai:
        if has_user_groq:
            logger.info("[LLMClient] BYOK: Using user-provided Groq API (free tier).")
            return _make_client(groq_key, GROQ_BASE_URL), GROQ_MODEL
        if has_user_deepseek:
            logger.info("[LLMClient] BYOK: Using user-provided DeepSeek API.")
            return _make_client(deepseek_key, "https://api.deepseek.com/v1"), "deepseek-chat"
        logger.info("[LLMClient] BYOK: Using user-provided OpenAI API.")
        return _make_client(openai_key), "gpt-4o-mini"

    # ── Server-Side Key Mode ────────────────────────────────────────────────
    srv_groq = settings.groq_api_key
    if srv_groq and not is_empty_key(srv_groq):
        logger.info("[LLMClient] Server Key: Using configured Groq API (free tier).")
        return _make_client(srv_groq, GROQ_BASE_URL), GROQ_MODEL

    srv_deepseek = settings.deepseek_api_key
    if srv_deepseek and not is_empty_key(srv_deepseek):
        logger.info("[LLMClient] Server Key: Using configured DeepSeek API.")
        return _make_client(srv_deepseek, "https://api.deepseek.com/v1"), "deepseek-chat"

    logger.info("[LLMClient] Server Key: Falling back to configured OpenAI API.")
    return _make_client(settings.openai_api_key), "gpt-4o-mini"


def get_openai_client(openai_key: str = None) -> AsyncOpenAI:
    """Returns an OpenAI-specific client as explicit fallback."""
    return _make_client(openai_key or settings.openai_api_key)


def get_groq_client(groq_key: str = None) -> AsyncOpenAI:
    """Returns a Groq client (OpenAI-compatible) for free-tier usage."""
    resolved = groq_key or settings.groq_api_key
    return _make_client(resolved, GROQ_BASE_URL)


def get_embedding_client() -> LocalEmbedder:
    """Returns a LocalEmbedder instance for zero-cost CPU embeddings."""
    return LocalEmbedder()
