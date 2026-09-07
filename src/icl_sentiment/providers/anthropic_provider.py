from __future__ import annotations

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider


@ProviderRegistry.register("anthropic")
class AnthropicProvider(SentimentProvider):
    """Anthropic Claude backend."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.api_key_env is None:
            config.api_key_env = "ANTHROPIC_API_KEY"
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                "Anthropic backend requires the 'anthropic' package: "
                "pip install icl-sentiment[anthropic]"
            ) from error
        # Identity-linked (service-account) API keys require every request to
        # name the workspace it acts in via the anthropic-workspace-id header.
        import os

        workspace_id = config.extra.get("workspace_id") or os.environ.get(
            "ANTHROPIC_WORKSPACE_ID"
        )
        default_headers = (
            {"anthropic-workspace-id": workspace_id} if workspace_id else None
        )
        self._client = anthropic.Anthropic(
            api_key=config.resolve_api_key(), default_headers=default_headers
        )

    def _complete(self, prompt: str, system_instruction: str | None = None) -> str:
        kwargs = {"system": system_instruction} if system_instruction else {}
        # The 1.x SDK / current Claude models no longer accept sampling
        # parameters (temperature), so config.temperature is not forwarded.
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return next(
            block.text for block in response.content if block.type == "text"
        )
