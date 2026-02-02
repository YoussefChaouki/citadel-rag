"""
Vector Embedding Service

Local embedding generation using sentence-transformers.
Model: all-MiniLM-L6-v2 (384 dimensions, ~22M parameters).

Design choices:
    - Singleton pattern: model loaded once, reused across requests.
    - Lazy loading: model downloaded/loaded on first embed call.
    - asyncio.to_thread: model inference is CPU-bound and must not
      block the FastAPI event loop.

Pre-download the model for production:
    python -c "from sentence_transformers import SentenceTransformer; \\
               SentenceTransformer('all-MiniLM-L6-v2')"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384


class VectorService:
    """
    Async embedding service backed by a local sentence-transformers model.

    The model is loaded lazily on first use and cached as a class-level
    singleton. All inference runs in a thread pool to keep the event
    loop responsive.

    Usage::

        vectors = await VectorService.embed_chunks(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
    """

    _model: ClassVar[Any] = None

    @classmethod
    def _get_model(cls) -> Any:
        """
        Get or lazily initialize the sentence-transformers model.

        The import is deferred so that ``sentence_transformers`` is not
        required at module-import time (keeps test collection fast).
        """
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s ...", MODEL_NAME)
            cls._model = SentenceTransformer(MODEL_NAME)
            logger.info("Model loaded (dim=%d)", EMBEDDING_DIMENSION)
        return cls._model

    @classmethod
    def _encode_sync(cls, texts: list[str]) -> list[list[float]]:
        """
        Synchronous batch encoding.

        Always call via ``asyncio.to_thread`` — this is CPU-bound
        and will block the calling thread for the duration of inference.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dimensional float vectors (L2-normalized).
        """
        model = cls._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        # numpy ndarray → native Python lists for pgvector compatibility
        result: list[list[float]] = embeddings.tolist()
        return result

    @classmethod
    async def embed_chunks(cls, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of text chunks.

        Offloads the blocking model inference to a thread pool
        via ``asyncio.to_thread``.

        Args:
            texts: List of chunk texts to embed.

        Returns:
            List of 384-dimensional embedding vectors.

        Raises:
            RuntimeError: If model loading fails.
        """
        if not texts:
            return []
        return await asyncio.to_thread(cls._encode_sync, texts)

    @classmethod
    async def embed_query(cls, query: str) -> list[float]:
        """
        Generate a single embedding for a search query.

        Convenience wrapper around ``embed_chunks`` for single-text use.

        Args:
            query: Search query string.

        Returns:
            384-dimensional embedding vector.
        """
        results = await cls.embed_chunks([query])
        return results[0]

    @classmethod
    def reset(cls) -> None:
        """
        Release the model from memory.

        Useful for testing or when switching models at runtime.
        """
        cls._model = None
        logger.info("VectorService model released")
