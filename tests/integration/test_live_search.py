"""
Semantic Search Integration Tests

Tests the /search endpoint against a running Docker stack.
Validates that the pgvector similarity search is operational.
"""

import httpx
import pytest


@pytest.mark.asyncio
async def test_search_endpoint_live(wait_for_api):
    """
    Verify semantic search endpoint returns valid response.

    Prerequisites:
        - Docker stack running (make up)
        - AI service in mock mode or with valid API key

    Note:
        Does not assert specific results since DB may be empty.
        Validates endpoint availability and response format.
    """
    base_url = "http://localhost:8000/api/v1/notes"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/search", json={"query": "strategy plan", "k": 3}
        )

        assert response.status_code == 200, f"Search failed: {response.text}"
        data = response.json()
        assert isinstance(data, list)  # Empty list is valid if no notes exist
