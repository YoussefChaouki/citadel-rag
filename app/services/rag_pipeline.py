"""
RAG Pipeline Orchestrator

Coordinates the full document lifecycle: ingestion → chunking →
embedding → vector storage. Also handles semantic search queries
and RAG-based question answering.

This is the single entry point for the API layer. It composes the
individual services (FileProcessor, TextChunker, VectorService,
RAGRepository, LLMService) into cohesive workflows.
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
from app.schemas.rag import AskResponse, SearchResult, SourceReference
from app.services.chunking import TextChunker
from app.services.ingestion import FileProcessor
from app.services.llm import LLMService
from app.services.vector import VectorService

logger = logging.getLogger(__name__)


class IngestResult(NamedTuple):
    """Return value of a successful ingestion."""

    document_id: UUID
    chunks_count: int
    is_duplicate: bool


class RAGPipeline:
    """
    RAG Pipeline Orchestrator.

    Central coordinator for the CITADEL document lifecycle. Composes five
    independent services into three cohesive workflows:

    Ingestion (ingest_file):
        Raw bytes → FileProcessor (text extraction) → TextChunker (recursive
        splitting) → VectorService (MiniLM-L6-v2 embedding) → RAGRepository
        (atomic persist with SHA-256 dedup).

    Search (search):
        Query string → VectorService (embed) → RAGRepository (pgvector cosine
        similarity via HNSW index) → ranked SearchResult DTOs.

    Ask (ask):
        Query string → Search pipeline → LLMService (Ollama generation with
        context grounding) → AskResponse with source references.

    Design Principles:
        - Single entry point for the API layer — endpoints never call
          individual services directly.
        - All CPU-bound work (PDF parsing, embedding inference) is offloaded
          to thread pools by the underlying services.
        - Graceful degradation: if Ollama is unreachable, ask returns
          a mock response with is_mocked=True while retrieval still works.

    Example:
        >>> pipeline = RAGPipeline()
        >>> async with session_factory() as session:
        ...     result = await pipeline.ingest_file(session, "doc.pdf", raw_bytes)
        ...     hits = await pipeline.search(session, "quantum computing", k=5)
        ...     answer = await pipeline.ask(session, "What is entanglement?")
    """

    def __init__(self) -> None:
        self._processor = FileProcessor()
        self._chunker = TextChunker()
        self._repository = RAGRepository()
        self._llm = LLMService()

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

        Executes a five-stage pipeline with an early-exit dedup check:

        1. Dedup — SHA-256 hash lookup against ``documents.file_hash``
           unique index. Returns immediately if content already exists.
        2. Extract — FileProcessor writes bytes to a temp file, extracts
           text via PyMuPDF (PDF) or UTF-8 decode (Markdown).
        3. Chunk — RecursiveCharacterTextSplitter with 500-char windows
           and 100-char overlap, tuned for MiniLM's 256-token context.
        4. Embed — Batch encoding via sentence-transformers in a thread
           pool (CPU-bound, non-blocking to the event loop).
        5. Persist — Atomic INSERT of DocumentRecord + all ChunkRecords
           in a single transaction with rollback on failure.

        Args:
            session: Active SQLAlchemy AsyncSession (caller-managed lifecycle).
            filename: Original filename with extension (e.g., "report.pdf").
                Used for display and dedup reporting, not for format detection.
            file_bytes: Raw file content as bytes.

        Returns:
            IngestResult: Named tuple containing:
                - document_id (UUID): Persisted document identifier.
                - chunks_count (int): Number of chunks created.
                - is_duplicate (bool): True if content hash already existed.

        Raises:
            ValueError: If file extension is not .pdf or .md.
            SQLAlchemyError: On database constraint violations or connectivity
                issues.
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

        Embeds the query using the same MiniLM-L6-v2 model used during ingestion,
        then executes a pgvector cosine distance query against the HNSW index on
        the chunks.embedding column.

        Similarity scores are computed as 1 - cosine_distance and range from
        -1 (opposite) to 1 (identical), with typical relevant results scoring
        above 0.5.

        Args:
            session: Active SQLAlchemy AsyncSession.
            query: Natural language search query (1–2000 chars).
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of SearchResult DTOs ordered by descending similarity score.
            Each result includes: chunk content, score, source filename,
            chunk index, and parent document UUID.
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
    # Ask (Full RAG)
    # ------------------------------------------------------------------

    async def ask(
        self,
        session: AsyncSession,
        query: str,
        k: int = 5,
    ) -> AskResponse:
        """
        Answer a question using the full retrieve-then-generate pipeline.

        Execution flow:
            1. Embed query via VectorService (MiniLM-L6-v2).
            2. Retrieve top-k chunks via pgvector cosine similarity.
            3. Assemble context string from retrieved chunks.
            4. Send context + query to LLMService (Ollama/Mistral).
            5. Package response with source references and mock status.

        Graceful Degradation:
            If Ollama is unreachable (ConnectError/TimeoutException), the
            LLMService returns a mock response with is_mocked=True. The
            retrieval step still executes normally, so source references are
            always populated when relevant documents exist.

        Args:
            session: Active SQLAlchemy AsyncSession.
            query: Natural language question (1–2000 chars).
            k: Number of context chunks to retrieve. Defaults to 5.
                Higher values provide more context but may introduce noise.

        Returns:
            AskResponse containing:
                - answer (str): Generated text or mock fallback.
                - sources (list[SourceReference]): Chunk references with scores.
                - is_mocked (bool): True if LLM was unavailable.
                - query (str): Original query for client-side reference.
        """

        logger.info("Processing RAG query: '%s' (k=%d)", query[:50], k)

        # --- Step 1 & 2: Retrieve relevant chunks ---
        query_embedding = await VectorService.embed_query(query)
        hits = await self._repository.search_similar(
            session,
            query_embedding,
            limit=k,
        )

        if not hits:
            logger.warning("No relevant chunks found for query: '%s'", query[:50])
            return AskResponse(
                answer="I found no relevant documents to answer this question.",
                sources=[],
                is_mocked=False,
                query=query,
            )

        # --- Step 3: Build context and source references ---
        context_chunks: list[str] = []
        sources: list[SourceReference] = []

        for chunk, score in hits:
            context_chunks.append(chunk.content)

            # Resolve source filename
            doc = await self._repository.get_document_by_id(session, chunk.document_id)
            filename = doc.filename if doc else "unknown"

            # Create preview (first 100 chars)
            preview = chunk.content[:100].replace("\n", " ")
            if len(chunk.content) > 100:
                preview += "..."

            sources.append(
                SourceReference(
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    score=score,
                    preview=preview,
                )
            )

        logger.info(
            "Retrieved %d chunks for context (top score=%.3f)",
            len(context_chunks),
            hits[0][1] if hits else 0,
        )

        # --- Step 4: Generate answer via LLM ---
        llm_response = await self._llm.generate_response(
            query=query,
            context_chunks=context_chunks,
        )

        logger.info(
            "Generated response (mocked=%s, length=%d)",
            llm_response.is_mocked,
            len(llm_response.content),
        )

        # --- Step 5: Build response ---
        return AskResponse(
            answer=llm_response.content,
            sources=sources,
            is_mocked=llm_response.is_mocked,
            query=query,
        )

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
