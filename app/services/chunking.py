"""
Chunking Service

Splits ingested Documents into smaller Chunks suitable for
embedding and vector retrieval. Uses LangChain's
RecursiveCharacterTextSplitter for intelligent boundary detection.

Configuration tuned for all-MiniLM-L6-v2:
    - Max sequence length: 256 tokens (~1024 chars)
    - chunk_size=500 chars: safe margin within the model's window
    - chunk_overlap=100 chars: preserves context across boundaries
"""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import Chunk, Document

logger = logging.getLogger(__name__)

# Defaults tuned for all-MiniLM-L6-v2 (256-token context window)
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100


class TextChunker:
    """
    Splits a Document into overlapping text Chunks.

    Uses recursive character splitting with intelligent separators
    that prefer splitting at paragraph > sentence > word boundaries.

    Usage::

        chunker = TextChunker()
        chunks = chunker.split(document)
        # Each chunk has: document_id, content, chunk_index, metadata

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        """Maximum characters per chunk."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Characters shared between consecutive chunks."""
        return self._chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        """
        Split a Document's content into Chunks.

        Args:
            document: Ingested Document with extracted text.

        Returns:
            List of Chunks with sequential indices and parent reference.
            Returns a single chunk if content is shorter than chunk_size.
        """
        texts = self._splitter.split_text(document.content)

        chunks = [
            Chunk(
                document_id=document.id,
                content=text,
                chunk_index=i,
                metadata={
                    "source_filename": document.metadata.filename,
                    "chunk_size": len(text),
                },
            )
            for i, text in enumerate(texts)
        ]

        logger.info(
            "Split document '%s' into %d chunks (size=%d, overlap=%d)",
            document.metadata.filename,
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )

        return chunks
