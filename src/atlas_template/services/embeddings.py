"""
Embedding Service

Background task processing for note embeddings.
Runs outside the HTTP request lifecycle with its own database session.
"""

import asyncio
import logging

from atlas_template.core.database import AsyncSessionLocal
from atlas_template.repositories import notes as repo
from atlas_template.services import ai

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2  # Base delay, multiplied by attempt number (linear backoff)


async def process_note_embedding(
    note_id: int,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """
    Generate and store embedding for a note (background task).

    Creates its own database session since FastAPI background tasks run
    after the HTTP response is sent and the request session is closed.

    Args:
        note_id: ID of the note to process.
        max_retries: Maximum retry attempts for transient failures.

    Returns:
        True if embedding was successfully generated and stored.
    """
    for attempt in range(max_retries):
        # New session per attempt: previous session may be in failed state
        async with AsyncSessionLocal() as session:
            try:
                note = await repo.get_by_id(session, note_id)
                if not note:
                    logger.warning(f"Note {note_id} not found for embedding processing")
                    return False

                text_to_embed = f"{note.title} {note.content}"
                vector = await ai.get_embedding(text_to_embed)

                await repo.update_embedding(session, note_id, vector)
                logger.info(f"Embedding generated for note {note_id}")
                return True

            except Exception as e:
                logger.error(
                    f"Error processing embedding for note {note_id} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    # Linear backoff: 2s, 4s, 6s...
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                    continue

                logger.error(
                    f"Failed to process embedding for note {note_id} "
                    f"after {max_retries} attempts"
                )
                return False

    return False
