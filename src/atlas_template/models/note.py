"""
Note Model

Core entity for storing notes with vector embeddings for semantic search.
Uses pgvector extension for efficient similarity queries.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from atlas_template.models.base import Base, TimestampMixin


class Note(Base, TimestampMixin):
    """
    Note entity with vector embedding support.

    Attributes:
        id: Primary key.
        title: Note title (max 200 chars), indexed for fast lookups.
        content: Full note content, no length limit.
        is_active: Soft delete flag.
        embedding: 1536-dim vector for semantic search (nullable until processed).
    """

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    content: Mapped[str] = mapped_column(String)
    is_active: Mapped[bool] = mapped_column(default=True)
    # Nullable: embedding is generated async after note creation
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    def __repr__(self) -> str:
        return f"<Note(id={self.id}, title='{self.title[:20]}...')>"
