"""
LLM Service

Local language model integration via Ollama API.
Provides generation capabilities for the RAG pipeline with
automatic fallback to mock responses when Ollama is unavailable.

Design:
    - Async HTTP calls via httpx (non-blocking).
    - Graceful degradation: returns mock response on connection failure.
    - Strict system prompt to ground responses in provided context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# System prompt enforcing context-grounded responses
SYSTEM_PROMPT: Final[
    str
] = """Tu es un assistant expert. Tu dois répondre aux questions de l'utilisateur en utilisant UNIQUEMENT le contexte fourni ci-dessous.

Règles strictes:
1. Base ta réponse UNIQUEMENT sur le contexte fourni.
2. Si le contexte ne contient pas l'information, dis-le clairement.
3. Ne fabrique jamais d'information.
4. Cite les sources quand c'est pertinent.
5. Réponds de manière concise et précise.

Contexte:
{context}
"""


@dataclass
class LLMResponse:
    """
    Response from the LLM service.

    Attributes:
        content: Generated text response.
        is_mocked: True if response is a fallback (Ollama unavailable).
    """

    content: str
    is_mocked: bool


class LLMService:
    """
    Async LLM service backed by Ollama with automatic fallback.

    Handles connection failures gracefully by returning a mock response
    instead of raising exceptions. This ensures the RAG pipeline remains
    functional even when Ollama is not running.

    Usage::

        service = LLMService()
        response = await service.generate_response(
            query="What is quantum computing?",
            context_chunks=["Quantum computers use qubits...", "..."]
        )
        if response.is_mocked:
            print("Warning: Using mock response")
        print(response.content)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            base_url: Ollama API base URL (default from config).
            model: Model name to use (default from config).
            timeout: Request timeout in seconds (default from config).
        """
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout

    async def generate_response(
        self,
        query: str,
        context_chunks: list[str],
    ) -> LLMResponse:
        """
        Generate a response using the LLM with provided context.

        Attempts to call Ollama API. On connection failure or timeout,
        returns a mock response with is_mocked=True.

        Args:
            query: User's question.
            context_chunks: Retrieved document chunks for context.

        Returns:
            LLMResponse with generated content and mock status.
        """
        # Format context from chunks
        context = self._format_context(context_chunks)
        prompt = self._build_prompt(query, context)

        try:
            return await self._call_ollama(prompt)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(
                "Ollama unreachable (%s), using mock response: %s",
                type(e).__name__,
                str(e),
            )
            return self._create_mock_response(context_chunks)
        except httpx.HTTPStatusError as e:
            logger.error("Ollama API error: %s", e.response.text)
            return self._create_mock_response(context_chunks)

    async def _call_ollama(self, prompt: str) -> LLMResponse:
        """
        Make the actual API call to Ollama.

        Args:
            prompt: Full prompt including system instructions and query.

        Returns:
            LLMResponse with generated content.

        Raises:
            httpx.ConnectError: If Ollama server is unreachable.
            httpx.TimeoutException: If request times out.
            httpx.HTTPStatusError: If API returns error status.
        """
        url = f"{self._base_url}/api/generate"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "")

            logger.info(
                "Ollama response generated (model=%s, length=%d)",
                self._model,
                len(content),
            )

            return LLMResponse(content=content, is_mocked=False)

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks into a single string."""
        if not chunks:
            return "Aucun contexte disponible."

        formatted_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            formatted_parts.append(f"[Source {i}]\n{chunk}")

        return "\n\n".join(formatted_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full prompt with system instructions."""
        system = SYSTEM_PROMPT.format(context=context)
        return f"{system}\n\nQuestion: {query}\n\nRéponse:"

    def _create_mock_response(self, context_chunks: list[str]) -> LLMResponse:
        """
        Create a fallback response when Ollama is unavailable.

        Includes a preview of the retrieved context to show the
        RAG retrieval is working even if generation is not.
        """
        if context_chunks:
            # Show first 50 chars of first chunk as preview
            preview = context_chunks[0][:50].replace("\n", " ")
            if len(context_chunks[0]) > 50:
                preview += "..."
            context_preview = f'"{preview}"'
        else:
            context_preview = "(no context retrieved)"

        content = (
            "⚠️ **Note: AI Service unavailable (Ollama not running).**\n\n"
            "Here is a simulated response based on the context found:\n\n"
            f"Retrieved context preview: {context_preview}\n\n"
            f"Total chunks retrieved: {len(context_chunks)}\n\n"
            "To enable full AI responses, please start Ollama with:\n"
            "```\nollama serve\n```"
        )

        return LLMResponse(content=content, is_mocked=True)

    async def health_check(self) -> bool:
        """
        Check if Ollama is reachable.

        Returns:
            True if Ollama API responds, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


# Module-level singleton for convenience
llm_service = LLMService()
