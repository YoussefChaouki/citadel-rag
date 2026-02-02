"""
RAG API Schemas

Pydantic models for the CITADEL RAG endpoint request/response cycle.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language search query",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    )


class SearchResult(BaseModel):
    """Single search result returned to the client."""

    content: str = Field(description="Chunk text content")
    score: float = Field(description="Cosine similarity score (higher = more relevant)")
    source: str = Field(description="Source filename")
    chunk_index: int = Field(description="Position within source document (0-based)")
    document_id: UUID = Field(description="Parent document identifier")


class IngestResponse(BaseModel):
    """Response for the file ingestion endpoint."""

    document_id: UUID | None = Field(
        default=None,
        description="Document UUID (set for duplicates, None while processing)",
    )
    filename: str = Field(description="Original filename")
    chunks_count: int = Field(
        default=0,
        description="Number of chunks created (0 while processing)",
    )
    status: str = Field(
        description="Processing status: 'processing', 'duplicate', 'completed'",
    )
    message: str = Field(description="Human-readable status message")
