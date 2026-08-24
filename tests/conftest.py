"""Shared pytest fixtures, including an offline fake provider for full-pipeline tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider


@ProviderRegistry.register("fake")
class FakeProvider(SentimentProvider):
    """Deterministic, offline provider used only in tests.

    Returns a pseudo-random valid label, seeded from the prompt's length, so
    tests can exercise the full pipeline without network access or API keys
    while staying reproducible across runs.
    """

    LABELS: ClassVar[list] = ["positive", "neutral", "negative"]

    def _complete(self, prompt: str) -> str:
        rng = random.Random(len(prompt))
        return rng.choice(self.LABELS)


@pytest.fixture
def fake_provider_config() -> ProviderConfig:
    return ProviderConfig(name="fake", model="fake-v1", max_retries=0)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "text": [
                "I absolutely loved this, best purchase ever.",
                "It was okay, nothing special about it.",
                "Terrible quality, broke within a day.",
                "Fantastic service and a great product overall.",
                "Just average, does what it says.",
                "Awful experience, would not recommend.",
                "Pretty good, I'm satisfied with the results.",
                "Meh, it's fine I guess.",
                "Worst thing I've bought this year.",
            ],
            "label": [
                "Positive",
                "Neutral",
                "Negative",
                "positive",
                "neutral",
                "negative",
                "positive",
                "neutral",
                "negative",
            ],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path
