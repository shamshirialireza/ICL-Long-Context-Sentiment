"""Dataset loading and validation.

Replaces the hardcoded per-dataset cells of the original notebook with a single
config-driven loader that works with any CSV/JSON/Parquet file containing a text
column and a label column.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        name: Short identifier used in results tables (e.g. "facebook", "news").
        path: Path to the data file (.csv, .json, .jsonl, or .parquet).
        text_column: Column holding the documents to classify.
        label_column: Column holding gold sentiment labels.
        drop_rows: Optional row indices to exclude (e.g. known bad annotations).
        limit: Optionally cap the number of rows (useful for smoke tests).
    """

    name: str
    path: str
    text_column: str = "text"
    label_column: str = "label"
    drop_rows: list[int] = field(default_factory=list)
    limit: int | None = None


class Dataset:
    """A loaded, validated dataset ready for classification and evaluation."""

    def __init__(self, config: DatasetConfig, frame: pd.DataFrame):
        self.config = config
        self.frame = frame

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def texts(self) -> list[str]:
        return self.frame[self.config.text_column].astype(str).tolist()

    @property
    def labels(self) -> list[str]:
        return self.frame[self.config.label_column].astype(str).str.strip().str.lower().tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(zip(self.texts, self.labels))

    def label_distribution(self) -> pd.Series:
        return pd.Series(self.labels).value_counts()

    def text_lengths(self) -> pd.Series:
        """Character length of every document, used for descriptive statistics."""
        return self.frame[self.config.text_column].astype(str).str.len()


class DatasetLoader:
    """Loads datasets from disk based on :class:`DatasetConfig`."""

    READERS: ClassVar[dict] = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".jsonl": lambda p: pd.read_json(p, lines=True),
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
    }

    def load(self, config: DatasetConfig) -> Dataset:
        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{config.name}' not found at {path}")

        reader = self.READERS.get(path.suffix.lower())
        if reader is None:
            supported = ", ".join(sorted(self.READERS))
            raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {supported}")

        frame = reader(path)

        for column in (config.text_column, config.label_column):
            if column not in frame.columns:
                raise KeyError(
                    f"Dataset '{config.name}' is missing column '{column}'. "
                    f"Available columns: {list(frame.columns)}"
                )

        frame = frame[frame[config.label_column].notna()]
        frame = frame[frame[config.text_column].notna()]
        if config.drop_rows:
            frame = frame.drop(index=config.drop_rows, errors="ignore")
        frame = frame.reset_index(drop=True)
        if config.limit is not None:
            frame = frame.head(config.limit)

        return Dataset(config, frame)

    def load_all(self, configs: Sequence[DatasetConfig]) -> list[Dataset]:
        return [self.load(config) for config in configs]
