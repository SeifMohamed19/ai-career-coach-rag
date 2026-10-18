from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger("EmbeddingEngine")

class LocalEmbeddingEngine:
    """
    Local embedding generator using HuggingFace sentence-transformers.
    Runs 100% locally on CPU/GPU without cost or API rate limits.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEmbeddingEngine, cls).__new__(cls)
            logger.info(f"Loading local embedding model: '{EMBEDDING_MODEL_NAME}'...")
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Local embedding model loaded successfully.")
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of document strings."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates a dense vector embedding for a single search query."""
        embedding = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.tolist()
