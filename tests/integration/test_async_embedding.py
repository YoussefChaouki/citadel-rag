"""
Async Embedding Integration Tests

Verifies that background embedding generation works end-to-end.
Requires Docker stack running (make up) with API and database accessible.
"""

import asyncio
import os
import uuid

import asyncpg
import httpx
import pytest


def _build_test_dsn() -> str:
    """Build PostgreSQL DSN from environment variables with fallback defaults."""
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'atlas')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'atlas_password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'atlas_db')}"
    )


@pytest.mark.asyncio
async def test_create_note_triggers_embedding(wait_for_api):
    """
    Verify that creating a note triggers async embedding generation.

    Test strategy:
        1. Create note via HTTP API
        2. Poll database directly to verify embedding was generated
        3. Uses mock AI service (no real OpenAI calls)

    Why direct DB check instead of /search endpoint:
        Vector similarity search may not return exact matches due to
        approximate nearest neighbor algorithms. Direct DB verification
        is deterministic.
    """
    base_url = "http://localhost:8000/api/v1/notes"
    unique_title = f"Async Test Note {uuid.uuid4()}"  # Avoid collisions between runs

    async with httpx.AsyncClient() as client:
        payload = {
            "title": unique_title,
            "content": "This note should be processed in background.",
            "is_active": True,
        }
        response = await client.post(f"{base_url}/", json=payload)
        assert response.status_code == 201, response.text
        note_id = response.json()["id"]

    # Polling config: 20 * 0.5s = 10s max wait for background task
    max_retries = 20
    poll_interval_s = 0.5

    dsn = (
        os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or _build_test_dsn()
    )
    # asyncpg requires standard postgresql:// scheme (not SQLAlchemy's +asyncpg)
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

    found = False
    conn = None

    try:
        conn = await asyncpg.connect(dsn)

        for _attempt in range(max_retries):
            await asyncio.sleep(poll_interval_s)

            row = await conn.fetchrow(
                "SELECT embedding IS NOT NULL AS has_embedding FROM notes WHERE id = $1",
                note_id,
            )

            if row and row["has_embedding"]:
                found = True
                break

    finally:
        if conn:
            await conn.close()

    assert found, (
        f"Note {note_id} embedding was not generated after "
        f"{max_retries * poll_interval_s}s"
    )
