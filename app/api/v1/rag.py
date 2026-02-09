"""
RAG API Router

HTTP endpoints for the CITADEL retrieval-augmented generation pipeline.

Endpoints:
    POST /ingest      — Upload a file for async ingestion (returns 202).
    POST /search      — Semantic search across ingested documents.
    POST /ask         — Full RAG: retrieve context and generate answer.
    GET /documents    — List all ingested documents.
    DELETE /documents/{filename} — Delete a document and its chunks.
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
    DeleteResponse,
    DocumentInfo,
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

    Performs a fast deduplication check using SHA-256 content hashing before
    accepting the file. If the hash already exists in the database, returns
    200 OK with the existing document metadata. Otherwise, queues the
    file for background processing and returns 202 Accepted immediately.

    Background Processing:
        The ingestion pipeline (text extraction → chunking → embedding →
        persistence) runs as a FastAPI BackgroundTask with its own database
        session. This decouples upload latency from processing time, which
        can be significant for large PDFs (10+ seconds for embedding).

    Args:
        file: Uploaded file (multipart/form-data). Must be .pdf or .md.
        response: FastAPI Response object for status code override.
        background_tasks: FastAPI background task scheduler.
        db: Request-scoped async database session (injected).
        repo: RAGRepository instance (injected).

    Returns:
        IngestResponse with document_id, filename, chunks_count, status,
        and a human-readable message.

    Raises:
        HTTPException 422: If file extension is not .pdf or .md.

    Status Codes:
        200: File already ingested (duplicate detected via SHA-256).
        202: File accepted for background processing.
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

    Embeds the query using the same MiniLM-L6-v2 model used during
    ingestion, then performs cosine similarity search via pgvector on
    stored chunk embeddings. Results are ordered by descending similarity
    score (1.0 = identical, 0.0 = orthogonal).

    The search uses an HNSW index (m=16, ef_construction=64) for
    sub-linear approximate nearest neighbor lookup, achieving <100ms
    latency on typical corpus sizes.

    Args:
        request: SearchRequest with query (str) and k (int, 1-50).
        db: Request-scoped async database session (injected).
        pipeline: RAGPipeline instance (injected).

    Returns:
        List of SearchResult DTOs with chunk content, similarity score,
        source filename, chunk index, and parent document UUID.
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

    Orchestrates the retrieve-then-generate flow:
        1. Embed the query using the local MiniLM-L6-v2 model.
        2. Retrieve the k most semantically similar document chunks
           via pgvector's HNSW cosine distance index.
        3. Assemble retrieved chunks into a context window.
        4. Generate an answer using Ollama (local Mistral model).

    Graceful Degradation:
        If Ollama is unavailable (not running or unreachable), the endpoint
        returns a structured mock response with is_mocked=True. The
        retrieval step still executes, so sources are populated with
        real chunk references and relevance scores. This allows:
            - Evaluating retrieval quality independently of generation.
            - Running in CI/CD without a GPU or LLM service.
            - Demonstrating the system in environments without Ollama.

    Args:
        request: AskRequest with query (str, 1-2000 chars) and k (int, 1-20).
        db: Request-scoped async database session (injected).
        pipeline: RAGPipeline instance (injected).

    Returns:
        AskResponse with answer text, source references, mock status,
        and the original query.
    """
    logger.info("RAG /ask request: query='%s', k=%d", request.query[:50], request.k)

    response = await pipeline.ask(db, request.query, request.k)

    if response.is_mocked:
        logger.warning(
            "Returning mocked response for query '%s' (Ollama unavailable)",
            request.query[:50],
        )

    return response


@router.get(
    "/documents",
    response_model=list[DocumentInfo],
    summary="List all ingested documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    repo: RAGRepository = Depends(_get_repository),
) -> list[DocumentInfo]:
    """
    Retrieve all documents currently in the RAG system.

    Returns document metadata including chunk counts for each ingested
    file. Ordered by creation date (newest first).

    Use cases:
        - UI document listing and management.
        - Inventory checks before evaluation runs.
        - Monitoring ingestion pipeline health.

    Args:
        db: Request-scoped async database session (injected).
        repo: RAGRepository instance (injected).

    Returns:
        List of DocumentInfo DTOs with document_id, filename,
        chunks_count, and ISO-formatted creation timestamp.
    """

    documents = await repo.get_all_documents(db)

    result: list[DocumentInfo] = []
    for doc in documents:
        chunks = await repo.get_chunks_by_document(db, doc.id)
        result.append(
            DocumentInfo(
                document_id=str(doc.id),
                filename=doc.filename,
                chunks_count=len(chunks),
                created_at=doc.created_at.isoformat(),
            )
        )

    return result


@router.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    summary="Delete a document",
    responses={
        200: {"description": "Document successfully deleted"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    filename: str,
    db: AsyncSession = Depends(get_db),
    repo: RAGRepository = Depends(_get_repository),
) -> DeleteResponse:
    """
    Delete a document and all associated data from the RAG system.

    Performs a cascading delete: removing the document record automatically
    removes all linked ChunkRecords (and their embeddings) via the
    ON DELETE CASCADE foreign key constraint.

    This operation is permanent and cannot be undone.

    Args:
        filename: Exact filename of the document to delete
            (e.g., "research_paper.pdf"). Case-sensitive.
        db: Request-scoped async database session (injected).
        repo: RAGRepository instance (injected).

    Returns:
        DeleteResponse with success flag, filename, chunks_deleted count,
        and a human-readable message.

    Raises:
        HTTPException 404: If no document with this filename exists.
    """
    logger.info("Delete request for document: '%s'", filename)

    found, chunks_deleted = await repo.delete_document_by_filename(db, filename)

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: '{filename}'",
        )

    logger.info("Successfully deleted '%s' (%d chunks)", filename, chunks_deleted)

    return DeleteResponse(
        success=True,
        filename=filename,
        chunks_deleted=chunks_deleted,
        message=f"'{filename}' deleted successfully ({chunks_deleted} chunks removed).",
    )
