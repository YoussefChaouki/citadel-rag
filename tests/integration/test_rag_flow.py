"""
RAG Pipeline Integration Tests

End-to-end tests for the CITADEL RAG pipeline:
    Ingest file → Wait for background processing → Search → Verify results.

Requires Docker stack running with rag-api service:
    docker compose up -d rag-api
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import pytest

CITADEL_URL = "http://localhost:8001"
RAG_BASE = f"{CITADEL_URL}/api/v1/rag"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wait_for_citadel() -> None:
    """Block until the CITADEL API is ready or timeout expires."""
    url = f"{CITADEL_URL}/health"
    timeout = 60  # Longer timeout: model download on first boot
    start = time.time()

    print("\n[Test] Waiting for CITADEL API...")
    while time.time() - start < timeout:
        try:
            res = httpx.get(url, timeout=2.0)
            if res.status_code == 200:
                print(f"CITADEL API ready ({time.time() - start:.1f}s)")
                return
        except httpx.RequestError:
            time.sleep(2)

    pytest.fail(
        f"CITADEL API unreachable at {CITADEL_URL} after {timeout}s. "
        "Is 'docker compose up rag-api' running?"
    )


# ---------------------------------------------------------------------------
# Test content
# ---------------------------------------------------------------------------

MARKDOWN_CONTENT = """\
# Quantum Computing Overview

Quantum computers leverage the principles of quantum mechanics to process
information in fundamentally new ways. Unlike classical bits that exist as
either 0 or 1, quantum bits (qubits) can exist in a superposition of both
states simultaneously.

## Key Concepts

Entanglement allows qubits to be correlated in ways that have no classical
equivalent. When two qubits are entangled, measuring one instantly determines
the state of the other, regardless of the distance between them.

## Applications

Quantum computing has promising applications in cryptography, drug discovery,
optimization problems, and machine learning. Companies like IBM, Google, and
various startups are racing to build practical quantum computers.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_returns_202(wait_for_citadel: None) -> None:
    """Uploading a new file should return 202 Accepted."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        unique_content = f"{MARKDOWN_CONTENT}\n\nRun ID: {uuid.uuid4()}"
        response = await client.post(
            f"{RAG_BASE}/ingest",
            files={
                "file": (
                    f"quantum_{uuid.uuid4()}.md",  #
                    unique_content.encode(),
                    "text/markdown",
                )
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_format(wait_for_citadel: None) -> None:
    """Uploading an unsupported file type should return 422."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{RAG_BASE}/ingest",
            files={"file": ("notes.docx", b"fake content", "application/octet-stream")},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_full_ingest_then_search(wait_for_citadel: None) -> None:
    """
    Full RAG flow integration test: Ingest -> Wait -> Search -> Verify.

    Strategy:
        To avoid "semantic noise" from previous test runs (where multiple
        documents might have similar content), this test injects a
        unique, random identifier (PROJECT_TITAN_xyz) into both the
        document content and the search query. This guarantees the
        target document has the highest semantic relevance.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Generate a unique identifier for this specific test run
        # utilizing a short UUID to ensure uniqueness across test executions.
        run_id = uuid.uuid4().hex[:8]
        unique_topic = f"PROJECT_TITAN_{run_id}"

        # 2. Inject this topic at the BEGINNING of the content.
        # Placing it at the start gives it high weight in the embedding vector.
        # The model will treat 'PROJECT_TITAN_xyz' as the primary subject.
        content = f"# Confidential Report: {unique_topic}\n\n{MARKDOWN_CONTENT}"
        unique_filename = f"quantum_{run_id}.md"

        # --- Ingest Step ---
        ingest_resp = await client.post(
            f"{RAG_BASE}/ingest",
            files={
                "file": (
                    unique_filename,
                    content.encode(),
                    "text/markdown",
                )
            },
        )
        # Expect 202 (Accepted/Processing) or 200 (Duplicate/Fast process)
        assert ingest_resp.status_code in (200, 202), ingest_resp.text

        # --- Search & Polling Step ---
        # 3. CRITICAL: Include the unique ID in the search query.
        # This forces the vector search to prioritize this specific document
        # over older, similar documents remaining in the database.
        search_payload = {
            "query": f"{unique_topic} quantum computing",
            "k": 10,  # Retrieve top 10 to ensure we catch it even if noisy
        }

        results: list[dict] = []

        # Poll the API for up to 60 seconds waiting for the async worker
        for _attempt in range(60):
            await asyncio.sleep(1)
            search_resp = await client.post(
                f"{RAG_BASE}/search",
                json=search_payload,
            )
            assert search_resp.status_code == 200
            results = search_resp.json()

            # Check if our specific filename appears in the result list
            if any(r["source"] == unique_filename for r in results):
                break

        # --- Verification Step ---
        # Filter to find exactly our document within the results
        my_results = [r for r in results if r["source"] == unique_filename]

        assert len(my_results) > 0, (
            f"Could not find document {unique_filename} when searching for unique topic '{unique_topic}'. "
            "Background task might have failed or timed out."
        )

        top_result = my_results[0]
        assert top_result["score"] > 0, "Similarity score should be positive"
        assert unique_filename in top_result["source"], "Source filename mismatch"


@pytest.mark.asyncio
async def test_duplicate_detection(wait_for_citadel: None) -> None:
    """
    Uploading the same file twice should return 200 with status='duplicate'.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        unique_content = (MARKDOWN_CONTENT + f"\n\nDup Test {uuid.uuid4()}").encode()

        await client.post(
            f"{RAG_BASE}/ingest",
            files={"file": ("quantum_dup.md", unique_content, "text/markdown")},
        )

        for _ in range(20):  # Max 20 secondes
            await asyncio.sleep(1)
            check = await client.post(
                f"{RAG_BASE}/search", json={"query": "quantum", "k": 1}
            )
            if check.status_code == 200 and len(check.json()) > 0:
                break

        response = await client.post(
            f"{RAG_BASE}/ingest",
            files={"file": ("quantum_dup_v2.md", unique_content, "text/markdown")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "duplicate"


@pytest.mark.asyncio
async def test_search_empty_corpus(wait_for_citadel: None) -> None:
    """Search for a topic not in any ingested document."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{RAG_BASE}/search",
            json={"query": "ancient egyptian pottery techniques", "k": 3},
        )

    assert response.status_code == 200
    # May return results with low scores if corpus has unrelated docs,
    # or empty list if corpus is empty. Both are valid.
    assert isinstance(response.json(), list)
