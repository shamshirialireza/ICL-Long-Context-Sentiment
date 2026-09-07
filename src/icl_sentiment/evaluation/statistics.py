#Descriptive statistics for datasets.

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from icl_sentiment.data.loader import Dataset

STAT_COLUMNS = ["No.", "Mean", "SD", "Min.", "25th Pctl.", "Median", "75th Pctl.", "Max."]


def describe_dataset(dataset: Dataset, unit: str = "chars") -> pd.Series:
    """One row of descriptive statistics for a dataset.

    Args:
        dataset: Loaded dataset.
        unit: "chars" for character lengths, "words" for whitespace token counts.
    """
    texts = dataset.frame[dataset.config.text_column].astype(str)
    if unit == "chars":
        lengths = texts.str.len()
    elif unit == "words":
        lengths = texts.str.split().str.len()
    else:
        raise ValueError(f"Unknown unit '{unit}'; use 'chars' or 'words'")

    return pd.Series(
        {
            "No.": len(lengths),
            "Mean": round(lengths.mean(), 2),
            "SD": round(lengths.std(), 2),
            "Min.": int(lengths.min()),
            "25th Pctl.": round(lengths.quantile(0.25), 2),
            "Median": round(lengths.median(), 2),
            "75th Pctl.": round(lengths.quantile(0.75), 2),
            "Max.": int(lengths.max()),
        },
        name=dataset.name,
    )


def describe_datasets(datasets: Sequence[Dataset], unit: str = "chars") -> pd.DataFrame:
    """Descriptive-statistics table across datasets (one row per dataset)."""
    table = pd.DataFrame([describe_dataset(dataset, unit=unit) for dataset in datasets])
    table.index.name = "Dataset"
    return table[STAT_COLUMNS]
