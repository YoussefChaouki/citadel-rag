"""
Note Schemas

Pydantic models for Note API request/response validation.
Separates concerns: NoteCreate (input), NoteResponse (output), NoteUpdate (partial).
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NoteBase(BaseModel):
    """Base schema with shared validation rules for Note fields."""

    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Note title (3-200 chars)",
    )
    content: str = Field(
        ...,
        min_length=5,
        description="Note content",
    )
    is_active: bool = Field(default=True)


class NoteCreate(NoteBase):
    """Request schema for POST /notes (inherits all NoteBase validations)."""

    pass


class NoteUpdate(BaseModel):
    """
    Request schema for PATCH /notes/{id}.

    All fields optional to support partial updates.
    Does not inherit NoteBase to avoid required field conflicts.
    """

    title: str | None = Field(None, min_length=3, max_length=200)
    content: str | None = Field(None, min_length=5)
    is_active: bool | None = None


class NoteRead(NoteBase):
    """Full Note representation including all timestamps."""

    id: int
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)  # Enables ORM model conversion


class NoteResponse(NoteBase):
    """
    Lightweight response schema for list endpoints.

    Excludes updated_at to reduce payload size on bulk queries.
    """

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)  # Enables ORM model conversion


class NoteSearchRequest(BaseModel):
    """Request schema for semantic search endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search query text",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    )
