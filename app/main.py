"""
CITADEL RAG Pipeline — Application Entry Point

FastAPI application for the CITADEL retrieval-augmented generation system.
Runs as a standalone service alongside the legacy Atlas API.

Start locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

Start in Docker:
    docker compose up rag-api
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.rag import router as rag_router
from app.core.database import dispose_engine, get_engine
from app.services.vector import VectorService

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
        1. Validate database connectivity.
        2. Pre-load the embedding model (avoids cold-start on first request).

    Shutdown:
        1. Dispose database engine.
        2. Release embedding model from memory.
    """
    logger.info("Starting CITADEL RAG Pipeline...")

    # Warm up database connection
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1"),
            )
        logger.info("Database connection verified")
    except Exception:
        logger.exception("Database connection failed")
        raise

    # Pre-load embedding model in a background thread
    logger.info("Pre-loading embedding model...")
    await asyncio.to_thread(VectorService._get_model)
    logger.info("Embedding model ready")

    yield

    # Shutdown
    VectorService.reset()
    await dispose_engine()
    logger.info("CITADEL shutdown complete")


app = FastAPI(
    title="CITADEL RAG Pipeline",
    description="Document ingestion, chunking, embedding, and semantic retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check for load balancers and orchestrators."""
    return {
        "status": "ok",
        "service": "citadel-rag",
        "environment": os.getenv("ENVIRONMENT", "local"),
    }
