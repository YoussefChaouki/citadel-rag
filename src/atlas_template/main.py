"""
Atlas Backend Application

FastAPI application entrypoint with async lifespan management.
Handles startup checks (database, Redis) and graceful shutdown.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from atlas_template.api.v1.notes import router as notes_router
from atlas_template.core.config import settings
from atlas_template.core.logging import setup_logging

# Initialize logging before any log statements
setup_logging()
logger = logging.getLogger(__name__)


async def wait_for_db(retries: int = 10, delay: int = 1) -> bool:
    """
    Wait for PostgreSQL to become available.

    Useful in containerized environments where the database may start
    after the application. Implements retry logic with linear delay.

    Args:
        retries: Maximum connection attempts.
        delay: Seconds between attempts.

    Returns:
        True if connection established, False if all retries exhausted.
    """
    engine = create_async_engine(settings.DATABASE_URL)
    for i in range(retries):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("Postgres connection established")
                await engine.dispose()
                return True
        except Exception as e:
            logger.warning(f"Waiting for Postgres ({i + 1}/{retries})... Error: {e}")
            await asyncio.sleep(delay)

    await engine.dispose()
    return False


async def check_redis() -> bool:
    """
    Verify Redis connectivity.

    Non-blocking check - application continues if Redis is unavailable.
    """
    try:
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await r.ping()
        logger.info(f"Redis connection established ({settings.REDIS_HOST})")
        await r.close()
        return True
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler (FastAPI 0.109+ pattern).

    Startup:
        - Validates database connectivity (required, blocks startup on failure)
        - Checks Redis connectivity (optional, logs warning if unavailable)

    Shutdown:
        - Logs shutdown event for observability
    """
    logger.info("Starting Atlas Platform...")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")

    if not await wait_for_db():
        logger.critical("Could not connect to Postgres. Shutting down.")
        raise RuntimeError("Database connection failed")

    if not await check_redis():
        logger.warning("Redis not reachable - continuing without cache")

    yield  # Application runs here

    logger.info("Shutting down Atlas Platform...")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(notes_router, prefix="/api/v1/notes", tags=["Notes"])


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.

    Returns:
        Static health status. For deep health checks, consider
        adding live database/Redis probes.
    """
    return {
        "status": "ok",
        "service": "atlas-template",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "db": "connected",
        "redis": "connected",
    }
