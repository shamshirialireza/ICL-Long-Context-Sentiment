"""Result reshaping helpers for human-readable reports.

Turns the long-form results table from :class:`ExperimentRunner` into the
per-model-per-dataset pivot tables requested for reporting: accuracy and
macro-F1 side by side for every (dataset, model, shot-count) combination.
"""

from __future__ import annotations

import pandas as pd


def summarize(results: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Pivot the long-form results into dataset rows x (model, n_shots) columns."""
    if results.empty:
        return results
    return results.pivot_table(
        index="dataset", columns=["model", "n_shots"], values=metric
    ).round(4)


def model_dataset_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, n_shots) with accuracy and macro-F1 columns.

    This is the primary table for the paper-style results: accuracy and
    macro-F1 for every model on every dataset, across all shot counts.
    """
    if results.empty:
        return results
    columns = ["dataset", "provider", "model", "n_shots", "accuracy", "macro_f1", "micro_f1", "n_samples", "n_invalid"]
    return results[columns].sort_values(["dataset", "model", "n_shots"]).reset_index(drop=True)
