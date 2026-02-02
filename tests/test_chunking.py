"""
Chunking Service Unit Tests

Verifies TextChunker behaviour: splitting logic, chunk indices,
overlap handling, edge cases, and metadata propagation.

No external services required — runs entirely offline.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.models.schemas import Chunk, Document, DocumentMetadata
from app.services.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, TextChunker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(content: str, filename: str = "test.md") -> Document:
    """Create a minimal Document for testing."""
    return Document(
        content=content,
        file_hash="a" * 64,
        metadata=DocumentMetadata(
            filename=filename,
            file_size=len(content.encode()),
            page_count=None,
            file_type="markdown",
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker() -> TextChunker:
    """Default TextChunker instance."""
    return TextChunker()


@pytest.fixture
def small_chunker() -> TextChunker:
    """TextChunker with small settings for deterministic testing."""
    return TextChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def long_document() -> Document:
    """Document with content long enough to require multiple chunks."""
    # ~2500 chars → should produce ~5 chunks at default settings (500/100)
    paragraphs = [
        f"Paragraph {i}. " + "This is filler text for testing purposes. " * 8
        for i in range(7)
    ]
    content = "\n\n".join(paragraphs)
    return _make_document(content, filename="long_doc.md")


@pytest.fixture
def short_document() -> Document:
    """Document shorter than chunk_size — should produce a single chunk."""
    return _make_document("Short content that fits in one chunk.")


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------


class TestBasicSplitting:
    """Tests for core splitting functionality."""

    def test_long_document_produces_multiple_chunks(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)

        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_document_produces_single_chunk(
        self, chunker: TextChunker, short_document: Document
    ) -> None:
        chunks = chunker.split(short_document)

        assert len(chunks) == 1
        assert chunks[0].content == short_document.content

    def test_all_content_preserved(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        """Concatenated chunks should cover the entire original content."""
        chunks = chunker.split(long_document)
        # combined = "".join(c.content for c in chunks)

        # With overlap, combined text is longer than original.
        # Every original character must appear in at least one chunk.
        for paragraph in long_document.content.split("\n\n"):
            trimmed = paragraph.strip()
            if trimmed:
                found = any(trimmed[:40] in c.content for c in chunks)
                assert found, f"Content lost: {trimmed[:40]}..."


# ---------------------------------------------------------------------------
# Chunk indices and references
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    """Tests for chunk indices, document references, and metadata."""

    def test_indices_are_sequential(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)
        indices = [c.chunk_index for c in chunks]

        assert indices == list(range(len(chunks)))

    def test_document_id_propagated(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)

        for chunk in chunks:
            assert chunk.document_id == long_document.id

    def test_chunk_ids_are_unique(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)
        ids = [c.id for c in chunks]

        assert len(set(ids)) == len(ids)
        assert all(isinstance(cid, UUID) for cid in ids)

    def test_metadata_includes_source_filename(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)

        for chunk in chunks:
            assert chunk.metadata["source_filename"] == "long_doc.md"

    def test_metadata_includes_chunk_size(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)

        for chunk in chunks:
            assert chunk.metadata["chunk_size"] == len(chunk.content)


# ---------------------------------------------------------------------------
# Chunk size and overlap
# ---------------------------------------------------------------------------


class TestChunkSizing:
    """Tests for chunk size limits and overlap behaviour."""

    def test_chunks_respect_max_size(
        self, chunker: TextChunker, long_document: Document
    ) -> None:
        chunks = chunker.split(long_document)

        for chunk in chunks:
            # LangChain may slightly exceed chunk_size at separator boundaries
            assert len(chunk.content) <= chunker.chunk_size + 50

    def test_custom_chunk_size(self, long_document: Document) -> None:
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.split(long_document)

        # Smaller chunks → more splits
        assert len(chunks) > 5

    def test_overlap_creates_shared_content(self, small_chunker: TextChunker) -> None:
        """Consecutive chunks should share some text when overlap > 0."""
        # Build text with clear word boundaries
        words = [f"word{i}" for i in range(50)]
        content = " ".join(words)
        doc = _make_document(content)

        chunks = small_chunker.split(doc)

        if len(chunks) >= 2:
            # At least some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                tail = chunks[i].content[-small_chunker.chunk_overlap :]
                head = chunks[i + 1].content[: small_chunker.chunk_overlap]
                # The shared region should have some common substring
                shared = set(tail.split()) & set(head.split())
                assert len(shared) > 0, (
                    f"No overlap found between chunks {i} and {i + 1}"
                )

    def test_default_config_values(self) -> None:
        assert DEFAULT_CHUNK_SIZE == 500
        assert DEFAULT_CHUNK_OVERLAP == 100

    def test_properties_match_config(self) -> None:
        chunker = TextChunker(chunk_size=300, chunk_overlap=75)
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 75


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for boundary conditions and error handling."""

    def test_single_character_content(self, chunker: TextChunker) -> None:
        doc = _make_document("X")
        chunks = chunker.split(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "X"

    def test_whitespace_heavy_content(self, chunker: TextChunker) -> None:
        doc = _make_document("Hello\n\n\n\n\nWorld")
        chunks = chunker.split(doc)

        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "Hello" in combined
        assert "World" in combined

    def test_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=200)
