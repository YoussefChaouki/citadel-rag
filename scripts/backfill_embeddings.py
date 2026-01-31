#!/usr/bin/env python3
"""
Backfill Embeddings Script

Seeds the database with sample notes and mock vector embeddings
for local development and testing of semantic search.

Usage:
    Requires Docker stack running (make up):
    $ python scripts/backfill_embeddings.py
"""

import asyncio
import os
import random
import sys

# Required for direct script execution without package installation
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas_template.models import Note

# Uses localhost (not 'db') since this script runs on the host, outside Docker network
DATABASE_URL = "postgresql+asyncpg://atlas:atlas_password@localhost:5432/atlas_db"

EMBEDDING_DIMENSION = 1536  # Must match OpenAI text-embedding-3-small output


def generate_mock_embedding() -> list[float]:
    """Generate a random 1536-dim vector simulating an OpenAI embedding."""
    return [random.random() for _ in range(EMBEDDING_DIMENSION)]


async def main() -> None:
    """
    Seed the notes table with sample data and mock embeddings.

    Warning:
        TRUNCATES existing data. Intended for dev/test environments only.
    """
    print("Starting Backfill...")
    engine = create_async_engine(DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        # TRUNCATE resets ID sequence; faster than DELETE for full table wipe
        await session.execute(text("TRUNCATE TABLE notes RESTART IDENTITY;"))

        notes_data = [
            ("Project Atlas Plan", "Strategy for 2026 AI Engineering path."),
            ("Grocery List", "Milk, eggs, bread, coffee."),
            ("Python Tips", "Use async/await for I/O bound tasks."),
        ]

        for title, content in notes_data:
            note = Note(
                title=title,
                content=content,
                embedding=generate_mock_embedding(),
            )
            session.add(note)

        await session.commit()
        print(f"Inserted {len(notes_data)} notes with embeddings.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
