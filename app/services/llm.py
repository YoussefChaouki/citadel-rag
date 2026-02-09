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
] = """You are an expert assistant. You must answer the user's questions using ONLY the context provided below.

Strict rules:
1. Base your answer ONLY on the provided context.
2. If the context does not contain the information, state it clearly.
3. Never fabricate information.
4. Cite sources when relevant.
5. Answer concisely and precisely.

Context:
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

    Integrates with Ollama's /api/generate endpoint for local LLM
    inference. When Ollama is unreachable (not running, network issue, or
    timeout), the service returns a structured mock response instead of
    raising exceptions.

    This fallback design ensures the RAG pipeline remains functional for
    retrieval and source attribution even without a running LLM, which is
    critical for:
        - CI/CD environments where Ollama is not available.
        - Development setups without a GPU.
        - Demo scenarios where only retrieval quality matters.

    The system prompt enforces context-grounded responses: the LLM is
    instructed to answer ONLY from the provided context and to explicitly
    state when information is insufficient.

    Configuration:
        All settings are read from environment variables via CitadelSettings:
            - OLLAMA_BASE_URL: API endpoint (default: host.docker.internal:11434)
            - OLLAMA_MODEL: Model name (default: mistral)
            - OLLAMA_TIMEOUT: Request timeout in seconds (default: 30.0)

    Example:
        >>> service = LLMService()
        >>> response = await service.generate_response(
        ...     query="What is quantum computing?",
        ...     context_chunks=["Quantum computers use qubits..."]
        ... )
        >>> if response.is_mocked:
        ...     logger.warning("Ollama unavailable — mock response returned")
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
        Generate a context-grounded response using the local LLM.

        Constructs a prompt with a French-language system instruction that
        enforces strict context adherence, then calls Ollama's generate API.

        On any connection or HTTP error, returns a mock response containing:
            - A warning banner indicating AI service unavailability.
            - A preview of the first retrieved chunk (proving retrieval works).
            - The total number of chunks that would have been used.
            - Instructions for starting Ollama.

        Args:
            query: The user's natural language question.
            context_chunks: Retrieved document chunks to ground the response.
                Empty list triggers a "no context available" system message.

        Returns:
            LLMResponse with:
                - content (str): Generated answer or mock fallback text.
                - is_mocked (bool): True if Ollama was unreachable.

        Note:
            This method never raises exceptions. All error paths return
            a valid LLMResponse with is_mocked=True.
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
            return "No context available."

        formatted_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            formatted_parts.append(f"[Source {i}]\n{chunk}")

        return "\n\n".join(formatted_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full prompt with system instructions."""
        system = SYSTEM_PROMPT.format(context=context)
        return f"{system}\n\nQuestion: {query}\n\nAnswer:"

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
