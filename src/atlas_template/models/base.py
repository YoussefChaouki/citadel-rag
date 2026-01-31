"""
SQLAlchemy Base Models

Provides the declarative base and reusable mixins for all ORM models.
"""

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Declarative base class for all SQLAlchemy ORM models."""

    pass


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at timestamp fields.

    Behavior:
        - created_at: Set by database on INSERT (server_default)
        - updated_at: Set by database on UPDATE (onupdate), NULL on insert

    Note:
        Uses timezone-aware timestamps for proper UTC handling.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # Database-side default, not Python-side
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),  # Automatically set on any UPDATE
        nullable=True,
    )
