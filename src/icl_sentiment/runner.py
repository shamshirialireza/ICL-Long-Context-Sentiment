"""Experiment orchestration.

Runs the full matrix of datasets x models x shot-counts, replacing the ~500
duplicated notebook cells with a single configurable loop. Supports resuming
from partial results (per-row checkpointing) so a rate limit or crash never
loses completed work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from icl_sentiment.data.loader import Dataset, DatasetConfig, DatasetLoader
from icl_sentiment.evaluation.metrics import evaluate_predictions
from icl_sentiment.prompting.examples import ExampleSelector
from icl_sentiment.prompting.templates import PromptBuilder
from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Top-level configuration for a benchmark run.

    Attributes:
        datasets: Datasets to evaluate on.
        providers: Model backends to evaluate.
        shot_counts: In-context example counts to sweep (0 = zero-shot).
        labels: Canonical sentiment labels, also used to stratify few-shot examples.
        document_noun: Word used in prompts to describe one row of text
            (e.g. "comment", "review", "post"). Domain-neutral by default.
        output_dir: Where per-run predictions and aggregate reports are written.
        example_seed: Seed for few-shot example sampling (reproducibility).
        length_unit: "chars" or "words" for the dataset descriptive-statistics table.
    """

    datasets: list[DatasetConfig]
    providers: list[ProviderConfig]
    shot_counts: list[int] = field(default_factory=lambda: [0, 3, 6, 9])
    labels: list[str] = field(default_factory=lambda: ["positive", "neutral", "negative"])
    document_noun: str = "text"
    output_dir: str = "results"
    example_seed: int = 42
    length_unit: str = "chars"

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        import yaml

        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        datasets = [DatasetConfig(**item) for item in raw.get("datasets", [])]
        providers = [ProviderConfig(**item) for item in raw.get("providers", [])]
        kwargs = {
            key: value
            for key, value in raw.items()
            if key not in {"datasets", "providers"}
        }
        return cls(datasets=datasets, providers=providers, **kwargs)


class ExperimentRunner:
    """Executes an :class:`ExperimentConfig` and produces result tables."""

    def __init__(self, config: ExperimentConfig, loader: DatasetLoader | None = None):
        self.config = config
        self.loader = loader or DatasetLoader()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.example_selector = ExampleSelector(seed=config.example_seed)

    def run(self, show_progress: bool = True) -> pd.DataFrame:
        """Run every (dataset, provider, shot-count) combination.

        Returns a long-form results table with one row per combination,
        containing accuracy, macro_f1, micro_f1, weighted_f1, n_samples, n_invalid.
        """
        rows: list[dict] = []
        datasets = self.loader.load_all(self.config.datasets)
        combinations = [
            (dataset, provider_config, n_shots)
            for dataset in datasets
            for provider_config in self.config.providers
            for n_shots in self.config.shot_counts
        ]

        iterator = tqdm(combinations, desc="experiments") if show_progress else combinations
        for dataset, provider_config, n_shots in iterator:
            provider = ProviderRegistry.create(provider_config)
            gold, predictions = self._run_one(dataset, provider, n_shots, show_progress=show_progress)
            result = evaluate_predictions(gold, predictions, labels=tuple(self.config.labels))

            rows.append(
                {
                    "dataset": dataset.name,
                    "provider": provider_config.name,
                    "model": provider_config.model,
                    "n_shots": n_shots,
                    **result.to_dict(),
                }
            )

        return pd.DataFrame(rows)

    def _run_one(
        self,
        dataset: Dataset,
        provider,
        n_shots: int,
        show_progress: bool,
    ) -> tuple:
        """Run one (dataset, provider, shot-count) combination with checkpointing.

        Returns ``(gold_labels, predictions)`` for the rows actually scored.
        Rows used as few-shot demonstrations are excluded from evaluation so a
        model is never scored on a document it was shown as an example.
        """
        checkpoint_path = self._checkpoint_path(dataset.name, provider.label, n_shots)
        cached = self._load_checkpoint(checkpoint_path)

        examples = self.example_selector.select(dataset, n_shots)
        excluded_texts = ExampleSelector.example_texts(examples)
        builder = PromptBuilder(labels=tuple(self.config.labels), document_noun=self.config.document_noun)

        texts = dataset.texts
        gold_labels = dataset.labels
        eval_indices = [i for i, text in enumerate(texts) if text not in excluded_texts]

        predictions: list[str] = list(cached) if cached else []
        remaining = eval_indices[len(predictions):]
        iterator = (
            tqdm(remaining, desc=f"{dataset.name}/{provider.label}/{n_shots}-shot", leave=False)
            if show_progress
            else remaining
        )

        for index in iterator:
            prompt = builder.build(texts[index], examples)
            try:
                raw_response = provider.classify(prompt)
            except Exception as error:  # noqa: BLE001
                logger.error("Giving up on row %d for %s: %s", index, provider.label, error)
                raw_response = ""
            predictions.append(raw_response)
            self._append_checkpoint(checkpoint_path, raw_response)

        gold = [gold_labels[i] for i in eval_indices]
        return gold, predictions

    def _checkpoint_path(self, dataset_name: str, provider_label: str, n_shots: int) -> Path:
        safe_provider = provider_label.replace("/", "_").replace(":", "_")
        filename = f"{dataset_name}__{safe_provider}__{n_shots}shot.jsonl"
        return self.output_dir / "checkpoints" / filename

    @staticmethod
    def _load_checkpoint(path: Path) -> list[str]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line)["response"] for line in handle if line.strip()]

    @staticmethod
    def _append_checkpoint(path: Path, response: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"response": response}) + "\n")
