from __future__ import annotations

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider


@ProviderRegistry.register("gemini")
class GeminiProvider(SentimentProvider):
    """Google Gemini backend via the google-generativeai SDK.

    The original notebook used Vertex AI with a service-account JSON; the
    public Gemini API key (GOOGLE_API_KEY) is simpler and equivalent for
    text generation.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.api_key_env is None:
            config.api_key_env = "GOOGLE_API_KEY"
        try:
            import google.generativeai as genai
        except ImportError as error:
            raise ImportError(
                "Gemini backend requires 'google-generativeai': "
                "pip install icl-sentiment[gemini]"
            ) from error
        genai.configure(api_key=config.resolve_api_key())
        self._model = genai.GenerativeModel(config.model)
        self._generation_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }

    def _complete(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt, generation_config=self._generation_config
        )
        return response.text
