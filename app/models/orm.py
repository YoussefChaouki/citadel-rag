"""
CITADEL RAG Database Models

SQLAlchemy 2.0 ORM models for the document and chunk storage layer.
Uses pgvector for vector similarity search on chunk embeddings.

Tables:
    documents — Ingested files with content hash for deduplication.
    chunks    — Document segments with 384-dim embeddings (MiniLM).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION: int = 384


class DocumentRecord(Base):
    """
    Persistent storage for ingested documents.

    Each record represents a fully processed file. The ``file_hash``
    column enforces uniqueness at the database level to prevent
    duplicate ingestion of the same file content.

    Attributes:
        id: UUID primary key (generated Python-side).
        filename: Original filename with extension.
        file_hash: SHA-256 hex digest (64 chars), unique index.
        file_metadata: JSONB blob for extensible metadata.
        created_at: Insertion timestamp (server-side default).
        chunks: Related ChunkRecord instances (cascade delete).
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
    )
    file_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Relationship — cascade ensures chunks are deleted with the document
    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkRecord.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<DocumentRecord(id={self.id!s:.8}, filename='{self.filename}')>"


class ChunkRecord(Base):
    """
    Persistent storage for document chunks with vector embeddings.

    Each chunk is a segment of a parent document with a 384-dimensional
    embedding vector for cosine similarity search via pgvector.

    Attributes:
        id: UUID primary key.
        document_id: Foreign key to parent document (CASCADE delete).
        chunk_index: Zero-based position within the parent document.
        content: Full text content of the chunk.
        embedding: 384-dim vector (nullable until embedding is generated).
        document: Back-reference to parent DocumentRecord.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True,
    )

    # Relationship
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<ChunkRecord(id={self.id!s:.8}, "
            f"doc={self.document_id!s:.8}, idx={self.chunk_index})>"
        )
