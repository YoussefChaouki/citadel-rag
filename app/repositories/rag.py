"""
RAG Repository

Data access layer for the CITADEL retrieval pipeline.
Provides atomic persistence of documents with their chunks
and vector similarity search via pgvector cosine distance.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import ChunkRecord, DocumentRecord

logger = logging.getLogger(__name__)


class RAGRepository:
    """
    Repository for document and chunk persistence with vector search.

    All methods expect an externally managed ``AsyncSession``
    (injected via FastAPI dependency or created in a service layer).

    Key guarantees:
        - ``save_document_with_chunks``: atomic — either the document
          AND all chunks are persisted, or nothing is.
        - ``search_similar``: returns chunks with cosine similarity scores,
          ordered by relevance (highest score first).
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save_document_with_chunks(
        self,
        session: AsyncSession,
        *,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
    ) -> tuple[DocumentRecord, bool]:
        """
        Persist a document and its chunks atomically.

        Checks for duplicate ``file_hash`` before inserting. If a document
        with the same hash already exists, returns it without modification.

        Args:
            session: Active async database session.
            document: DocumentRecord to persist.
            chunks: ChunkRecords linked to the document.

        Returns:
            Tuple of (document, created).
            ``created=True`` if newly inserted, ``False`` if duplicate.
        """
        # Dedup check — SHA-256 hash uniqueness
        stmt = select(DocumentRecord).where(
            DocumentRecord.file_hash == document.file_hash
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()

        if existing is not None:
            logger.info(
                "Document already exists (hash=%s): %s",
                document.file_hash[:12],
                existing.filename,
            )
            return existing, False

        # Atomic insert: document + all chunks in one transaction
        session.add(document)
        session.add_all(chunks)
        await session.flush()
        await session.commit()
        await session.refresh(document)

        logger.info(
            "Saved document '%s' with %d chunks (hash=%s)",
            document.filename,
            len(chunks),
            document.file_hash[:12],
        )
        return document, True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_document_by_hash(
        self,
        session: AsyncSession,
        file_hash: str,
    ) -> DocumentRecord | None:
        """Look up a document by its SHA-256 content hash."""
        stmt = select(DocumentRecord).where(DocumentRecord.file_hash == file_hash)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def get_document_by_id(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> DocumentRecord | None:
        """Look up a document by its UUID."""
        stmt = select(DocumentRecord).where(DocumentRecord.id == document_id)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def search_similar(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[ChunkRecord, float]]:
        """
        Search chunks by cosine similarity against a query vector.

        Uses pgvector's ``cosine_distance`` operator, leveraging the
        HNSW index on ``chunks.embedding`` for sub-linear lookup.

        The cosine distance is converted to a similarity score:
        ``score = 1 - distance`` (range: [-1, 1], higher = more similar).

        Args:
            session: Active async database session.
            query_embedding: 384-dimensional query vector.
            limit: Maximum number of results to return.

        Returns:
            List of (ChunkRecord, similarity_score) tuples,
            ordered by similarity (highest first).
        """
        distance = ChunkRecord.embedding.cosine_distance(query_embedding).label(
            "distance"
        )

        stmt = (
            select(ChunkRecord, distance)
            .where(ChunkRecord.embedding.isnot(None))
            .order_by(distance)
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.all()

        # Convert cosine distance → similarity score
        return [(row[0], round(1.0 - float(row[1]), 4)) for row in rows]

    async def get_chunks_by_document(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> Sequence[ChunkRecord]:
        """Get all chunks for a document, ordered by index."""
        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.chunk_index)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# Module-level singleton for convenience imports
rag_repository = RAGRepository()
