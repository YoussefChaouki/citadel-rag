"""
RAG Pipeline Orchestrator

Coordinates the full document lifecycle: ingestion → chunking →
embedding → vector storage. Also handles semantic search queries.

This is the single entry point for the API layer. It composes the
individual services (FileProcessor, TextChunker, VectorService,
RAGRepository) into cohesive workflows.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import ChunkRecord, DocumentRecord
from app.models.schemas import Document
from app.repositories.rag import RAGRepository
from app.schemas.rag import SearchResult
from app.services.chunking import TextChunker
from app.services.ingestion import FileProcessor
from app.services.vector import VectorService

logger = logging.getLogger(__name__)


class IngestResult(NamedTuple):
    """Return value of a successful ingestion."""

    document_id: UUID
    chunks_count: int
    is_duplicate: bool


class RAGPipeline:
    """
    Orchestrates the full RAG document lifecycle.

    Composes the individual CITADEL services into two workflows:

    **Ingestion** (``ingest_file``):
        UploadFile bytes → FileProcessor → TextChunker →
        VectorService → RAGRepository

    **Search** (``search``):
        Query string → VectorService → RAGRepository → SearchResults

    All methods are async-safe. CPU-bound work (PDF parsing,
    embedding inference) is offloaded to thread pools by the
    underlying services.

    Usage::

        pipeline = RAGPipeline()
        async with session_factory() as session:
            result = await pipeline.ingest_file(session, "doc.pdf", raw)
            hits = await pipeline.search(session, "quantum computing", k=5)
    """

    def __init__(self) -> None:
        self._processor = FileProcessor()
        self._chunker = TextChunker()
        self._repository = RAGRepository()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        session: AsyncSession,
        filename: str,
        file_bytes: bytes,
    ) -> IngestResult:
        """
        Process a file through the full ingestion pipeline.

        Steps:
            1. Dedup check via SHA-256 hash (fast, avoids redundant work).
            2. Extract text content (FileProcessor).
            3. Split into chunks (TextChunker).
            4. Generate embeddings (VectorService, CPU-bound in thread).
            5. Persist document + chunks atomically (RAGRepository).

        Args:
            session: Active async database session.
            filename: Original filename with extension.
            file_bytes: Raw file content.

        Returns:
            IngestResult with document_id, chunks_count, and duplicate flag.
        """
        # --- Step 1: Fast dedup check ---
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = await self._repository.get_document_by_hash(session, file_hash)
        if existing is not None:
            existing_chunks = await self._repository.get_chunks_by_document(
                session,
                existing.id,
            )
            logger.info("Duplicate detected: '%s' (hash=%s)", filename, file_hash[:12])
            return IngestResult(
                document_id=existing.id,
                chunks_count=len(existing_chunks),
                is_duplicate=True,
            )

        # --- Step 2: Extract text via temp file ---
        document = await self._extract_document(filename, file_bytes)

        # --- Step 3: Chunk ---
        chunks = self._chunker.split(document)
        logger.info(
            "Chunked '%s': %d chunks from %d chars",
            filename,
            len(chunks),
            len(document.content),
        )

        # --- Step 4: Embed ---
        texts = [c.content for c in chunks]
        embeddings = await VectorService.embed_chunks(texts)
        logger.info("Generated %d embeddings for '%s'", len(embeddings), filename)

        # --- Step 5: Persist ---
        doc_record = DocumentRecord(
            id=document.id,
            filename=filename,
            file_hash=document.file_hash,
            file_metadata=document.metadata.model_dump(),
        )
        chunk_records = [
            ChunkRecord(
                id=chunk.id,
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embeddings[i],
            )
            for i, chunk in enumerate(chunks)
        ]

        await self._repository.save_document_with_chunks(
            session,
            document=doc_record,
            chunks=chunk_records,
        )

        return IngestResult(
            document_id=document.id,
            chunks_count=len(chunk_records),
            is_duplicate=False,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        session: AsyncSession,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        """
        Perform semantic search against the chunk vector store.

        Args:
            session: Active async database session.
            query: Natural language search query.
            k: Maximum number of results.

        Returns:
            List of SearchResult DTOs ordered by relevance.
        """
        query_embedding = await VectorService.embed_query(query)

        hits = await self._repository.search_similar(
            session,
            query_embedding,
            limit=k,
        )

        results: list[SearchResult] = []
        for chunk, score in hits:
            # Resolve source filename from parent document
            doc = await self._repository.get_document_by_id(session, chunk.document_id)
            source = doc.filename if doc else "unknown"

            results.append(
                SearchResult(
                    content=chunk.content,
                    score=score,
                    source=source,
                    chunk_index=chunk.chunk_index,
                    document_id=chunk.document_id,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_document(
        self,
        filename: str,
        file_bytes: bytes,
    ) -> Document:
        """
        Extract a Document via FileProcessor using a temp file.

        Writes bytes to a temp file with the correct extension,
        processes it, then patches the metadata with the original
        filename (temp files have names like ``tmp1a2b3c.pdf``).
        """
        suffix = Path(filename).suffix.lower()
        tmp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            document = await self._processor.process(tmp_path)

            # Fix metadata: replace temp filename with original
            fixed_metadata = document.metadata.model_copy(
                update={"filename": filename},
            )
            return document.model_copy(update={"metadata": fixed_metadata})

        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
