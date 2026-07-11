import asyncio
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_model = None

def get_local_embedder():
    global _model
    if _model is None:
        logger.info("[LocalEmbedder] Loading sentence-transformers model 'all-MiniLM-L6-v2' (CPU-friendly singleton)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("[LocalEmbedder] Model loaded successfully.")
    return _model

class LocalEmbedder:
    def __init__(self):
        self.model = get_local_embedder()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        logger.info(f"[LocalEmbedder] Encoding {len(texts)} texts asynchronously on CPU...")
        embs = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, normalize_embeddings=True).tolist()
        )
        logger.info(f"[LocalEmbedder] Encoding finished. Generated {len(embs)} vectors.")
        return embs
