"""Models package - re-exports all models for convenient imports."""

from atlas_template.models.base import Base, TimestampMixin
from atlas_template.models.note import Note

__all__ = [
    "Base",
    "TimestampMixin",
    "Note",
]
