import httpx
import logging
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.local_embedder import LocalEmbedder

logger = logging.getLogger(__name__)

def is_empty_key(k: str) -> bool:
    return not k or "placeholder" in k or k.strip() in ("", "your-deepseek-api-key", "your-openai-api-key")

def get_llm_client(openai_key: str = None, deepseek_key: str = None) -> tuple[AsyncOpenAI, str]:
    """
    Returns the appropriate AsyncOpenAI client and model name.
    If the user has provided their own keys (BYOK), we strictly use them:
      - If deepseek_key is provided, use it.
      - If openai_key is provided, use it.
    If no user keys are provided, we fallback to server-side configured keys:
      - Prefer server-side DeepSeek key if configured.
      - Otherwise fallback to server-side OpenAI key.
    """
    has_user_deepseek = deepseek_key and not is_empty_key(deepseek_key)
    has_user_openai = openai_key and not is_empty_key(openai_key)

    if has_user_deepseek or has_user_openai:
        # BYOK Mode: Use user-provided keys strictly. Do NOT fallback to server keys.
        if has_user_deepseek:
            logger.info("[LLMClient] BYOK: Directing requests to user-provided DeepSeek API.")
            client = AsyncOpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com/v1",
                http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
            )
            return client, "deepseek-chat"
        else:
            logger.info("[LLMClient] BYOK: Falling back to user-provided OpenAI API.")
            client = AsyncOpenAI(
                api_key=openai_key,
                http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
            )
            return client, "gpt-4o-mini"

    # Server-Side Key Mode: Use server .env configured keys.
    resolved_deepseek = settings.deepseek_api_key
    if resolved_deepseek and not is_empty_key(resolved_deepseek):
        logger.info("[LLMClient] Server Key: Directing requests to configured DeepSeek API.")
        client = AsyncOpenAI(
            api_key=resolved_deepseek,
            base_url="https://api.deepseek.com/v1",
            http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
        )
        return client, "deepseek-chat"
    
    resolved_openai = settings.openai_api_key
    logger.info("[LLMClient] Server Key: Falling back to configured OpenAI API.")
    client = AsyncOpenAI(
        api_key=resolved_openai,
        http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
    )
    return client, "gpt-4o-mini"

def get_openai_client(openai_key: str = None) -> AsyncOpenAI:
    """
    Returns an OpenAI-specific client as fallback.
    """
    resolved_openai = openai_key or settings.openai_api_key
    return AsyncOpenAI(
        api_key=resolved_openai,
        http_client=httpx.AsyncClient(cookies=None, timeout=90.0)
    )

def get_embedding_client() -> LocalEmbedder:
    """
    Returns a LocalEmbedder instance strictly for zero-cost CPU embeddings.
    """
    return LocalEmbedder()

