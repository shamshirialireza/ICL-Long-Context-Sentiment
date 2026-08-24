"""Few-shot example selection.

The original notebook hardcoded example strings for each dataset and shot count.
This module selects examples dynamically from the dataset itself, stratified
across classes so a 3-shot prompt gets one example per sentiment class, 6-shot
gets two, and so on.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from icl_sentiment.data.loader import Dataset


@dataclass(frozen=True)
class FewShotExample:
    text: str
    label: str


class ExampleSelector:
    """Selects few-shot examples from a dataset, stratified by class."""

    def __init__(self, seed: int = 42, max_example_chars: int | None = None):
        self.seed = seed
        self.max_example_chars = max_example_chars

    def select(self, dataset: Dataset, n_shots: int) -> list[FewShotExample]:
        """Pick ``n_shots`` examples, balanced across the dataset's classes.

        Returns an empty list for ``n_shots=0`` (zero-shot). Raises if a class
        has too few examples to satisfy the request.
        """
        if n_shots == 0:
            return []

        rng = random.Random(self.seed)
        by_class: dict = {}
        for text, label in dataset:
            by_class.setdefault(label, []).append(text)

        classes = sorted(by_class)
        if n_shots % len(classes) != 0:
            raise ValueError(
                f"n_shots={n_shots} is not divisible by the number of classes "
                f"({len(classes)}: {classes}); cannot stratify evenly."
            )
        per_class = n_shots // len(classes)

        examples: list[FewShotExample] = []
        for label in classes:
            pool = by_class[label]
            if len(pool) < per_class:
                raise ValueError(
                    f"Class '{label}' has only {len(pool)} examples; {per_class} needed."
                )
            for text in rng.sample(pool, per_class):
                if self.max_example_chars is not None:
                    text = text[: self.max_example_chars]
                examples.append(FewShotExample(text=text, label=label))

        rng.shuffle(examples)
        return examples

    @staticmethod
    def example_texts(examples: Sequence[FewShotExample]) -> set[str]:
        """Texts used as demonstrations, so the runner can exclude them from evaluation."""
        return {example.text for example in examples}
