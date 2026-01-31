"""
AI Service

OpenAI integration for generating text embeddings.
Supports mock mode for local development without API costs.
"""

import os
import random

from openai import AsyncOpenAI

EMBEDDING_DIMENSION = 1536  # text-embedding-3-small output size


async def get_embedding(text: str) -> list[float]:
    """
    Generate a vector embedding for the given text.

    Uses OpenAI's text-embedding-3-small model in production.
    Falls back to random vectors when OPENAI_API_KEY is missing or set to 'mock'.

    Args:
        text: Input text to embed.

    Returns:
        1536-dimensional embedding vector.

    Raises:
        Exception: If OpenAI API call fails (in production mode).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # Mock mode: random vectors for dev/test (no API costs, no network dependency)
    if not api_key or api_key.lower() == "mock":
        return [random.random() for _ in range(EMBEDDING_DIMENSION)]

    client = AsyncOpenAI(api_key=api_key)
    text = text.replace("\n", " ")  # OpenAI recommends single-line input

    try:
        response = await client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"OpenAI Error: {e}")
        raise e
