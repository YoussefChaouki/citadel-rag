"""Models package — Pydantic schemas and SQLAlchemy ORM for the CITADEL pipeline."""

from app.models.orm import EMBEDDING_DIMENSION, ChunkRecord, DocumentRecord
from app.models.schemas import Chunk, Document, DocumentMetadata

__all__ = [
    # Pydantic schemas (ingestion pipeline)
    "Chunk",
    "Document",
    "DocumentMetadata",
    # SQLAlchemy ORM (persistence layer)
    "ChunkRecord",
    "DocumentRecord",
    "EMBEDDING_DIMENSION",
]
