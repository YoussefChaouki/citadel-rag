"""
CITADEL Database Layer

Autonomous async database setup for the CITADEL RAG pipeline.
Reads connection parameters from environment variables directly
(no dependency on atlas_template.core.config) to keep the ``app/``
package self-contained.

Design:
    - Lazy initialization: engine created on first use, not at import.
    - get_session_factory: returns a reusable async session maker.
    - get_db: FastAPI dependency that yields a request-scoped session.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Module-level singletons (lazy)
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_database_url() -> str:
    """Build async PostgreSQL URL from environment variables."""
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> AsyncEngine:  # noqa: ANN201
    """Get or create the async SQLAlchemy engine (singleton)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        url = _build_database_url()
        _engine = create_async_engine(url, echo=False, pool_size=5)
        logger.info(
            "Database engine created: %s@%s",
            os.environ.get("POSTGRES_USER", "?"),
            os.environ.get("POSTGRES_HOST", "?"),
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory (singleton)."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a request-scoped database session.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine at application shutdown."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed")
