"""
Database Configuration

Async SQLAlchemy 2.0 setup with connection pooling and session management.
Uses asyncpg as the PostgreSQL driver for non-blocking I/O.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atlas_template.core.config import settings
from atlas_template.models.base import Base

# Async engine with connection pooling (default pool_size=5, max_overflow=10)
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# expire_on_commit=False: prevents implicit I/O after commit when accessing attributes
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Yields:
        AsyncSession: Scoped to the request lifecycle. Automatically closed
        after the request completes (including on exceptions).
    """
    async with AsyncSessionLocal() as session:
        yield session


# Re-export Base for Alembic migrations compatibility
__all__ = ["Base", "engine", "AsyncSessionLocal", "get_db"]
