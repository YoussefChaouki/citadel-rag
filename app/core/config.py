"""
CITADEL Configuration

Centralized settings for the CITADEL RAG pipeline.
Reads from environment variables with sensible defaults.

Note:
    This is separate from atlas_template.core.config to keep
    the CITADEL app/ package self-contained.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CitadelSettings:
    """
    Immutable configuration for CITADEL services.

    Attributes:
        ollama_base_url: Base URL for Ollama API.
            Default uses host.docker.internal for Docker-to-host communication.
        ollama_model: Model name to use for generation.
        ollama_timeout: Request timeout in seconds.
    """

    ollama_base_url: str
    ollama_model: str
    ollama_timeout: float

    @classmethod
    def from_env(cls) -> CitadelSettings:
        """Load settings from environment variables."""
        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://host.docker.internal:11434",
            ),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
            ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "30.0")),
        )


# Module-level singleton
settings = CitadelSettings.from_env()
