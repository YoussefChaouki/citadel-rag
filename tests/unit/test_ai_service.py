"""
AI Service Unit Tests

Tests for the embedding generation service with mocked OpenAI client.
No external API calls - runs without network or API keys.
"""

from unittest.mock import AsyncMock, patch

import pytest

from atlas_template.services.ai import get_embedding


@pytest.mark.asyncio
async def test_get_embedding_mock():
    """
    Verify get_embedding calls OpenAI API correctly when API key is present.

    Uses mock to avoid real API calls and validate:
        - Correct model selection (text-embedding-3-small)
        - Proper input formatting
        - Response parsing
    """
    mock_vector = [0.1] * 1536
    # Dynamically create response object matching OpenAI SDK structure
    mock_response = type(
        "Response", (), {"data": [type("Item", (), {"embedding": mock_vector})]}
    )

    with patch("atlas_template.services.ai.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.embeddings.create = AsyncMock(return_value=mock_response)

        # Inject dummy API key to bypass mock mode in get_embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            vector = await get_embedding("Hello World")

        assert len(vector) == 1536
        assert vector[0] == 0.1
        mock_instance.embeddings.create.assert_called_once()
        args, kwargs = mock_instance.embeddings.create.call_args
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == ["Hello World"]
