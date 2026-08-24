from __future__ import annotations

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider


@ProviderRegistry.register("anthropic")
class AnthropicProvider(SentimentProvider):
    """Anthropic Claude backend."""

    SYSTEM_PROMPT = "You are a precise sentiment analyzer. Respond with exactly one word."

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
        self._client = anthropic.Anthropic(api_key=config.resolve_api_key())

    def _complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
