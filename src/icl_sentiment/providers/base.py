"""Provider abstraction: one interface for every LLM backend.

New providers register themselves with :class:`ProviderRegistry` so the runner
and CLI discover them by name without any hardcoded branching.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Runtime configuration for a provider.

    Attributes:
        name: Registry name of the provider ("openai", "anthropic", "gemini").
        model: Model identifier passed to the API.
        api_key_env: Environment variable holding the API key.
        temperature: Sampling temperature (0 for deterministic labeling).
        max_tokens: Response token cap (labels are one word, so keep small).
        request_interval: Seconds to sleep between requests (rate limiting).
        max_retries: Retries per document before giving up.
        extra: Provider-specific keyword arguments (e.g. Vertex project/location).
    """

    name: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 10
    request_interval: float = 0.0
    max_retries: int = 3
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise OSError(
                f"API key environment variable '{self.api_key_env}' is not set. "
                f"Export it before running (see .env.example)."
            )
        return key


class SentimentProvider(abc.ABC):
    """Base class for LLM sentiment classifiers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abc.abstractmethod
    def _complete(self, prompt: str, system_instruction: str | None = None) -> str:
        """Send one prompt to the backend and return the raw text response."""

    def classify(self, prompt: str, system_instruction: str | None = None) -> str:
        """Classify with retries and exponential backoff; returns raw model output."""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._complete(prompt, system_instruction)
                if self.config.request_interval:
                    time.sleep(self.config.request_interval)
                return response
            except Exception as error:  # noqa: BLE001 - provider SDKs raise varied types
                last_error = error
                wait = 2**attempt
                logger.warning(
                    "%s request failed (attempt %d/%d): %s — retrying in %ds",
                    self.config.name,
                    attempt + 1,
                    self.config.max_retries + 1,
                    error,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Provider '{self.config.name}' failed after "
            f"{self.config.max_retries + 1} attempts"
        ) from last_error

    @property
    def label(self) -> str:
        return f"{self.config.name}:{self.config.model}"


class ProviderRegistry:
    """Name -> provider class registry with a decorator for registration."""

    _providers: ClassVar[dict[str, type[SentimentProvider]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[SentimentProvider]], type[SentimentProvider]]:
        def decorator(provider_cls: type[SentimentProvider]) -> type[SentimentProvider]:
            cls._providers[name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def create(cls, config: ProviderConfig) -> SentimentProvider:
        if config.name not in cls._providers:
            available = ", ".join(sorted(cls._providers)) or "(none registered)"
            raise KeyError(f"Unknown provider '{config.name}'. Available: {available}")
        return cls._providers[config.name](config)

    @classmethod
    def available(cls) -> list:
        return sorted(cls._providers)
