"""
RAG API Router

HTTP endpoints for the CITADEL retrieval-augmented generation pipeline.

Endpoints:
    POST /ingest  — Upload a file for async ingestion (returns 202).
    POST /search  — Semantic search across ingested documents.
    POST /ask     — Full RAG: retrieve context and generate answer.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Response,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, get_session_factory
from app.repositories.rag import RAGRepository
from app.schemas.rag import (
    AskRequest,
    AskResponse,
    IngestResponse,
    SearchRequest,
    SearchResult,
)
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_pipeline() -> RAGPipeline:
    """FastAPI dependency — returns a RAGPipeline instance."""
    return RAGPipeline()


def _get_repository() -> RAGRepository:
    """FastAPI dependency — returns a RAGRepository instance."""
    return RAGRepository()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _run_ingest(filename: str, file_bytes: bytes) -> None:
    """
    Background task that runs the full ingestion pipeline.

    Creates its own database session because FastAPI background tasks
    execute after the HTTP response is sent — the request-scoped
    session is already closed by then.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            pipeline = RAGPipeline()
            result = await pipeline.ingest_file(session, filename, file_bytes)
            logger.info(
                "Ingestion complete: '%s' → %d chunks (dup=%s)",
                filename,
                result.chunks_count,
                result.is_duplicate,
            )
        except Exception:
            logger.exception("Ingestion failed for '%s'", filename)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload a file for ingestion",
    responses={
        200: {"description": "File already ingested (duplicate)"},
        202: {"description": "File accepted for background processing"},
    },
)
async def ingest_file(
    file: UploadFile,
    response: Response,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    repo: RAGRepository = Depends(_get_repository),
) -> IngestResponse:
    """
    Upload a PDF or Markdown file for ingestion into the RAG pipeline.

    The file content is hashed (SHA-256) for deduplication. If the file
    has already been ingested, returns 200 with the existing document info.
    Otherwise, the processing pipeline runs in the background and the
    endpoint returns 202 immediately.
    """
    raw = await file.read()
    filename = file.filename or "unknown"

    if not filename.lower().endswith((".pdf", ".md")):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: '{filename}'. Accepted: .pdf, .md",
        )

    # --- Fast dedup check (hash + indexed lookup) ---
    file_hash = hashlib.sha256(raw).hexdigest()
    existing = await repo.get_document_by_hash(db, file_hash)

    if existing is not None:
        chunks = await repo.get_chunks_by_document(db, existing.id)
        response.status_code = 200
        return IngestResponse(
            document_id=existing.id,
            filename=filename,
            chunks_count=len(chunks),
            status="duplicate",
            message=f"File already ingested as '{existing.filename}'.",
        )

    # --- New file → background processing ---
    background_tasks.add_task(_run_ingest, filename, raw)

    response.status_code = 202
    return IngestResponse(
        filename=filename,
        status="processing",
        message=f"'{filename}' accepted for processing.",
    )


@router.post(
    "/search",
    response_model=list[SearchResult],
    summary="Semantic search across documents",
)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    pipeline: RAGPipeline = Depends(_get_pipeline),
) -> list[SearchResult]:
    """
    Search ingested documents by semantic similarity.

    Embeds the query using the local MiniLM model, then performs
    cosine similarity search via pgvector on stored chunk embeddings.
    """
    return await pipeline.search(db, request.query, request.k)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question using RAG",
    responses={
        200: {
            "description": "Answer generated successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "normal": {
                            "summary": "Normal response with Ollama",
                            "value": {
                                "answer": "Based on the context...",
                                "sources": [
                                    {
                                        "filename": "doc.pdf",
                                        "chunk_index": 0,
                                        "score": 0.85,
                                        "preview": "First 100 chars...",
                                    }
                                ],
                                "is_mocked": False,
                                "query": "What is quantum computing?",
                            },
                        },
                        "mocked": {
                            "summary": "Mock response (Ollama unavailable)",
                            "value": {
                                "answer": "⚠️ Note: AI Service unavailable...",
                                "sources": [],
                                "is_mocked": True,
                                "query": "What is quantum computing?",
                            },
                        },
                    }
                }
            },
        },
    },
)
async def ask(
    request: AskRequest,
    db: AsyncSession = Depends(get_db),
    pipeline: RAGPipeline = Depends(_get_pipeline),
) -> AskResponse:
    """
    Answer a question using the full RAG pipeline.

    Process:
        1. Embed the query using the local MiniLM model.
        2. Retrieve the k most relevant document chunks via pgvector.
        3. Generate an answer using Ollama (local LLM).

    Graceful Degradation:
        If Ollama is unavailable (not running), the endpoint returns
        a mock response with ``is_mocked=True``. The retrieval step
        still works, so you can see which documents would be used.

    To enable full responses, start Ollama:
        ```
        ollama serve
        ollama pull mistral
        ```
    """
    logger.info("RAG /ask request: query='%s', k=%d", request.query[:50], request.k)

    response = await pipeline.ask(db, request.query, request.k)

    if response.is_mocked:
        logger.warning(
            "Returning mocked response for query '%s' (Ollama unavailable)",
            request.query[:50],
        )

    return response
