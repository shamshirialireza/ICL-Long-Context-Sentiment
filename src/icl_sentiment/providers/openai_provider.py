from __future__ import annotations

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider


@ProviderRegistry.register("openai")
class OpenAIProvider(SentimentProvider):
    """OpenAI chat-completions backend (e.g. gpt-4o)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.api_key_env is None:
            config.api_key_env = "OPENAI_API_KEY"
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "OpenAI backend requires the 'openai' package: pip install icl-sentiment[openai]"
            ) from error
        self._client = OpenAI(api_key=config.resolve_api_key())

    def _complete(self, prompt: str, system_instruction: str | None = None) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content or ""
