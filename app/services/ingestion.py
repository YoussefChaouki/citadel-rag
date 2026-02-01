"""
Document Ingestion Service

File processing pipeline for the CITADEL RAG system.
Extracts text from PDF and Markdown files, computes SHA-256
hashes for idempotent processing, and collects file metadata.

Supported formats:
    - PDF (.pdf): Text extraction via PyMuPDF (fitz)
    - Markdown (.md): Raw UTF-8 text decoding
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Final
from uuid import uuid4

import fitz  # PyMuPDF

from app.models.schemas import Document, DocumentMetadata

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf", ".md"})


class FileProcessor:
    """
    Async file processor for document ingestion.

    Reads files from disk, computes a SHA-256 content hash for
    deduplication, and extracts text content with metadata.

    All public methods are async-safe. Blocking I/O (file reads,
    PDF parsing) is offloaded to a thread pool via asyncio.to_thread.

    Usage::

        processor = FileProcessor()
        doc = await processor.process(Path("report.pdf"))
        print(doc.file_hash, doc.metadata.page_count)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, file_path: Path) -> Document:
        """
        Process a file and return a structured Document.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            Document with extracted content, SHA-256 hash, and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Read raw bytes in a thread to avoid blocking the event loop
        raw = await asyncio.to_thread(file_path.read_bytes)
        file_hash = self._compute_hash(raw)

        if suffix == ".pdf":
            return await self._process_pdf(file_path, raw, file_hash)
        return await self._process_markdown(file_path, raw, file_hash)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute SHA-256 hex digest of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _extract_pdf_content(raw: bytes) -> tuple[str, int]:
        """
        Extract text and page count from PDF bytes.

        This is a *synchronous* helper — always call via
        ``asyncio.to_thread`` to keep the event loop free.
        """
        doc = fitz.open(stream=raw, filetype="pdf")
        try:
            pages: list[str] = [page.get_text() for page in doc]
            return "\n".join(pages), len(pages)
        finally:
            doc.close()

    async def _process_pdf(
        self,
        file_path: Path,
        raw: bytes,
        file_hash: str,
    ) -> Document:
        """Build a Document from a PDF file."""
        content, page_count = await asyncio.to_thread(self._extract_pdf_content, raw)

        logger.info(
            "Processed PDF: %s (%d pages, %d bytes)",
            file_path.name,
            page_count,
            len(raw),
        )

        return Document(
            id=uuid4(),
            content=content,
            file_hash=file_hash,
            metadata=DocumentMetadata(
                filename=file_path.name,
                file_size=len(raw),
                page_count=page_count,
                file_type="pdf",
            ),
        )

    async def _process_markdown(
        self,
        file_path: Path,
        raw: bytes,
        file_hash: str,
    ) -> Document:
        """Build a Document from a Markdown file."""
        content = raw.decode("utf-8")

        logger.info(
            "Processed Markdown: %s (%d bytes)",
            file_path.name,
            len(raw),
        )

        return Document(
            id=uuid4(),
            content=content,
            file_hash=file_hash,
            metadata=DocumentMetadata(
                filename=file_path.name,
                file_size=len(raw),
                page_count=None,
                file_type="markdown",
            ),
        )
