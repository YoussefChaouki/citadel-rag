"""
Base Repository

Generic repository pattern implementation for async SQLAlchemy CRUD operations.
Provides type-safe database access with consistent session handling.
"""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_template.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic repository providing common CRUD operations.

    Implements the Repository pattern with async SQLAlchemy. All methods
    expect an externally managed session (injected via FastAPI dependency).

    Usage:
        class NoteRepository(BaseRepository[Note]):
            def __init__(self):
                super().__init__(Note)
    """

    def __init__(self, model: type[ModelType]):
        self.model = model

    async def create(self, session: AsyncSession, obj_in: Any) -> ModelType:
        """
        Create a new record.

        Args:
            session: Active database session.
            obj_in: Pydantic schema or dict with entity data.

        Returns:
            The created entity with database-generated fields populated.
        """
        data = obj_in.model_dump() if hasattr(obj_in, "model_dump") else obj_in
        db_obj = self.model(**data)
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)  # Load DB-generated fields (id, created_at)
        return db_obj

    async def get_by_id(self, session: AsyncSession, id: int) -> ModelType | None:
        """Get a record by primary key. Returns None if not found."""
        result = await session.execute(
            select(self.model).where(self.model.id == id)  # type: ignore[attr-defined]
        )
        return result.scalars().first()

    async def get_all(
        self,
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[ModelType]:
        """Get all records with offset-based pagination."""
        result = await session.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()

    async def update(
        self,
        session: AsyncSession,
        db_obj: ModelType,
        obj_in: Any,
    ) -> ModelType:
        """
        Update a record with partial data.

        Args:
            session: Active database session.
            db_obj: Existing entity to update.
            obj_in: Pydantic schema or dict (only provided fields are updated).
        """
        update_data = (
            obj_in.model_dump(exclude_unset=True)  # Partial update support
            if hasattr(obj_in, "model_dump")
            else obj_in
        )
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj

    async def delete(self, session: AsyncSession, db_obj: ModelType) -> None:
        """Delete a record. Hard delete - consider soft delete for production."""
        await session.delete(db_obj)
        await session.commit()
