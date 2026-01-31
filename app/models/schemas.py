"""
CITADEL Document Schemas

Pydantic models for the ingestion pipeline.
Defines the core data structures for documents and chunks
flowing through the RAG system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted during file processing."""

    filename: str = Field(description="Original filename with extension")
    file_size: int = Field(ge=0, description="File size in bytes")
    page_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of pages (PDF only, None for other formats)",
    )
    file_type: str = Field(
        description="Format identifier: 'pdf', 'markdown'",
    )


class Document(BaseModel):
    """
    Processed document ready for chunking and embedding.

    Produced by FileProcessor after text extraction.
    The file_hash field enables idempotent processing —
    documents with the same hash are considered duplicates.

    Attributes:
        id: Unique identifier (auto-generated UUID4).
        content: Extracted text content.
        file_hash: SHA-256 hex digest of raw file bytes.
        metadata: File-level metadata (name, size, page count).
        created_at: UTC timestamp of processing.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1, description="Extracted text content")
    file_hash: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest for deduplication",
    )
    metadata: DocumentMetadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class Chunk(BaseModel):
    """
    A segment of a document for embedding and vector retrieval.

    Chunks are produced by splitting a Document's content into
    overlapping windows suitable for semantic search.

    Attributes:
        id: Unique chunk identifier.
        document_id: Reference to the parent Document.
        content: Chunk text.
        chunk_index: Zero-based position within the parent document.
        metadata: Chunk-specific context (page number, char offsets, etc.).
    """

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID = Field(description="Parent document reference")
    content: str = Field(min_length=1, description="Chunk text content")
    chunk_index: int = Field(ge=0, description="Position in document (0-based)")
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Chunk-level metadata (page_number, char_start, etc.)",
    )
