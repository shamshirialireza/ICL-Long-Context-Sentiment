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
        self._genai = genai
        self._system_instruction: str | None = None
        self._model = genai.GenerativeModel(config.model)
        self._generation_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }

    def _complete(self, prompt: str, system_instruction: str | None = None) -> str:
        # The SDK only accepts system instructions at model construction, so
        # rebuild the (local, stateless) model object when the instruction changes.
        if system_instruction != self._system_instruction:
            self._model = self._genai.GenerativeModel(
                self.config.model, system_instruction=system_instruction
            )
            self._system_instruction = system_instruction
        # Default gRPC deadline is 600s; a hung request would stall the whole
        # run, so fail fast and let the retry loop in classify() handle it.
        response = self._model.generate_content(
            prompt,
            generation_config=self._generation_config,
            request_options={"timeout": 60},
        )
        return response.text
