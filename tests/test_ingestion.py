"""
Ingestion Service Unit Tests

Verifies FileProcessor behaviour for PDF and Markdown extraction,
SHA-256 idempotency, error handling, and metadata correctness.

No external services required — runs entirely offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import fitz
import pytest

from app.models.schemas import Document
from app.services.ingestion import SUPPORTED_EXTENSIONS, FileProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor() -> FileProcessor:
    """Fresh FileProcessor instance."""
    return FileProcessor()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal single-page PDF with known text content."""
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello CITADEL")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def multipage_pdf(tmp_path: Path) -> Path:
    """Create a 3-page PDF for pagination tests."""
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """Create a minimal Markdown file."""
    path = tmp_path / "test.md"
    path.write_text(
        "# CITADEL\n\nRAG Pipeline documentation.\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------


class TestPDFProcessing:
    """Tests for PDF file extraction."""

    @pytest.mark.asyncio
    async def test_extracts_text(
        self, processor: FileProcessor, sample_pdf: Path
    ) -> None:
        doc = await processor.process(sample_pdf)

        assert isinstance(doc, Document)
        assert "Hello CITADEL" in doc.content

    @pytest.mark.asyncio
    async def test_metadata_fields(
        self, processor: FileProcessor, sample_pdf: Path
    ) -> None:
        doc = await processor.process(sample_pdf)

        assert doc.metadata.filename == "test.pdf"
        assert doc.metadata.file_type == "pdf"
        assert doc.metadata.page_count == 1
        assert doc.metadata.file_size > 0

    @pytest.mark.asyncio
    async def test_multipage_extraction(
        self, processor: FileProcessor, multipage_pdf: Path
    ) -> None:
        doc = await processor.process(multipage_pdf)

        assert doc.metadata.page_count == 3
        assert "Page 1 content" in doc.content
        assert "Page 2 content" in doc.content
        assert "Page 3 content" in doc.content

    @pytest.mark.asyncio
    async def test_returns_valid_uuid(
        self, processor: FileProcessor, sample_pdf: Path
    ) -> None:
        doc = await processor.process(sample_pdf)
        assert isinstance(doc.id, UUID)


# ---------------------------------------------------------------------------
# Markdown processing
# ---------------------------------------------------------------------------


class TestMarkdownProcessing:
    """Tests for Markdown file extraction."""

    @pytest.mark.asyncio
    async def test_extracts_text(
        self, processor: FileProcessor, sample_md: Path
    ) -> None:
        doc = await processor.process(sample_md)

        assert isinstance(doc, Document)
        assert "# CITADEL" in doc.content
        assert "RAG Pipeline documentation." in doc.content

    @pytest.mark.asyncio
    async def test_metadata_fields(
        self, processor: FileProcessor, sample_md: Path
    ) -> None:
        doc = await processor.process(sample_md)

        assert doc.metadata.filename == "test.md"
        assert doc.metadata.file_type == "markdown"
        assert doc.metadata.page_count is None
        assert doc.metadata.file_size > 0


# ---------------------------------------------------------------------------
# SHA-256 hashing & idempotency
# ---------------------------------------------------------------------------


class TestHashing:
    """Tests for content hashing and deduplication."""

    @pytest.mark.asyncio
    async def test_hash_is_valid_sha256(
        self, processor: FileProcessor, sample_md: Path
    ) -> None:
        doc = await processor.process(sample_md)

        assert len(doc.file_hash) == 64
        # Verify it's valid hex
        int(doc.file_hash, 16)

    @pytest.mark.asyncio
    async def test_hash_matches_manual_sha256(
        self, processor: FileProcessor, sample_md: Path
    ) -> None:
        expected = hashlib.sha256(sample_md.read_bytes()).hexdigest()
        doc = await processor.process(sample_md)

        assert doc.file_hash == expected

    @pytest.mark.asyncio
    async def test_idempotent_hash(
        self, processor: FileProcessor, sample_md: Path
    ) -> None:
        """Same file content → same hash, different UUIDs."""
        doc1 = await processor.process(sample_md)
        doc2 = await processor.process(sample_md)

        assert doc1.file_hash == doc2.file_hash
        assert doc1.id != doc2.id  # New UUID each call

    @pytest.mark.asyncio
    async def test_different_content_different_hash(
        self, processor: FileProcessor, tmp_path: Path
    ) -> None:
        file_a = tmp_path / "a.md"
        file_b = tmp_path / "b.md"
        file_a.write_text("Content A", encoding="utf-8")
        file_b.write_text("Content B", encoding="utf-8")

        doc_a = await processor.process(file_a)
        doc_b = await processor.process(file_b)

        assert doc_a.file_hash != doc_b.file_hash


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for invalid inputs and edge cases."""

    @pytest.mark.asyncio
    async def test_unsupported_extension(
        self, processor: FileProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.docx"
        path.write_bytes(b"fake content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.process(path)

    @pytest.mark.asyncio
    async def test_file_not_found(
        self, processor: FileProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "ghost.pdf"

        with pytest.raises(FileNotFoundError, match="File not found"):
            await processor.process(path)

    def test_supported_extensions_constant(self) -> None:
        """Guard against accidental changes to supported formats."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert len(SUPPORTED_EXTENSIONS) == 2
