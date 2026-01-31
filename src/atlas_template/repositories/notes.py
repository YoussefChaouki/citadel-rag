"""
Note Repository

Data access layer for Note entities with semantic search capabilities.
Extends BaseRepository with pgvector-specific query methods.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_template.models import Note
from atlas_template.repositories.base import BaseRepository
from atlas_template.schemas.notes import NoteCreate


class NoteRepository(BaseRepository[Note]):
    """
    Repository for Note entities with vector search support.

    Inherits standard CRUD from BaseRepository and adds:
        - search_similar: Semantic search via pgvector cosine distance
        - update_embedding: Targeted embedding updates for background tasks
    """

    def __init__(self) -> None:
        super().__init__(Note)

    async def search_similar(
        self,
        session: AsyncSession,
        embedding: list[float],
        limit: int = 5,
    ) -> Sequence[Note]:
        """
        Search notes by vector similarity using cosine distance (pgvector).

        Args:
            session: Database session.
            embedding: Query vector (1536 dimensions).
            limit: Maximum number of results.

        Returns:
            List of notes ordered by semantic similarity (closest first).
        """
        stmt = (
            select(Note)
            .where(Note.embedding.isnot(None))  # Exclude notes pending embedding
            .order_by(Note.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    async def update_embedding(
        self,
        session: AsyncSession,
        note_id: int,
        embedding: list[float],
    ) -> None:
        """
        Update only the embedding field of a note.

        Used by background tasks to avoid full entity reload.
        Uses bulk UPDATE for efficiency (no SELECT required).

        Args:
            session: Database session.
            note_id: Note ID to update.
            embedding: New embedding vector (1536 dimensions).
        """
        stmt = update(Note).where(Note.id == note_id).values(embedding=embedding)
        await session.execute(stmt)
        await session.commit()


# Module-level instance for function-based API
note_repository = NoteRepository()


# ============================================================================
# Function-based API (delegates to repository instance)
# Provides a simpler import pattern: `from repositories import notes as repo`
# ============================================================================


async def create(session: AsyncSession, note_in: NoteCreate | dict[str, Any]) -> Note:
    """Create a new note."""
    return await note_repository.create(session, note_in)


async def get_by_id(session: AsyncSession, note_id: int) -> Note | None:
    """Get a note by ID."""
    return await note_repository.get_by_id(session, note_id)


async def get_all(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[Note]:
    """Get all notes with pagination."""
    return await note_repository.get_all(session, skip, limit)


async def search_similar_notes(
    session: AsyncSession,
    embedding: list[float],
    limit: int = 5,
) -> Sequence[Note]:
    """Search notes by semantic similarity."""
    return await note_repository.search_similar(session, embedding, limit)


async def update_embedding(
    session: AsyncSession,
    note_id: int,
    embedding: list[float],
) -> None:
    """Update note embedding (used by background tasks)."""
    await note_repository.update_embedding(session, note_id, embedding)
