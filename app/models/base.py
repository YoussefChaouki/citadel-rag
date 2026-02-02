"""
Shared SQLAlchemy base for CITADEL models.

Re-exports from atlas_template to ensure a single DeclarativeBase
across both legacy and new model packages. This guarantees that
Alembic sees all tables in one metadata registry.
"""

from atlas_template.models.base import Base, TimestampMixin

__all__ = ["Base", "TimestampMixin"]
