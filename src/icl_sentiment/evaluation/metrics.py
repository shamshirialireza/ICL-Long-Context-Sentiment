#Metric calculations

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

_LABEL_PATTERN = re.compile(r"[a-z]+")


def normalize_label(raw: str | None, valid_labels: Sequence[str]) -> str:
    """Map raw model output to a canonical label.

    Handles casing, whitespace, punctuation, and verbose outputs such as
    "Sentiment: negative." by scanning for the first valid label word.
    Unmatched outputs become "invalid" so they count as errors instead of
    crashing the scorer.
    """
    if raw is None:
        return "invalid"
    text = str(raw).strip().lower()
    valid = set(valid_labels)
    if text in valid:
        return text
    for word in _LABEL_PATTERN.findall(text):
        if word in valid:
            return word
    return "invalid"


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame
    n_samples: int
    n_invalid: int

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "n_samples": self.n_samples,
            "n_invalid": self.n_invalid,
        }


def evaluate_predictions(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str] = ("positive", "neutral", "negative"),
) -> EvaluationResult:
    """Score predictions against gold labels.

    Both inputs are normalized first, so raw model outputs can be passed
    directly. Invalid predictions are retained (scored as wrong), matching how
    a deployed classifier would be penalized for unparseable output.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"Length mismatch: {len(gold)} gold vs {len(predicted)} predictions")

    gold_norm = [normalize_label(value, labels) for value in gold]
    pred_norm = [normalize_label(value, labels) for value in predicted]
    n_invalid = sum(1 for value in pred_norm if value == "invalid")

    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_norm, pred_norm, labels=label_list, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=label_list,
    )

    return EvaluationResult(
        accuracy=accuracy_score(gold_norm, pred_norm),
        macro_f1=f1_score(gold_norm, pred_norm, labels=label_list, average="macro", zero_division=0),
        micro_f1=f1_score(gold_norm, pred_norm, labels=label_list, average="micro", zero_division=0),
        weighted_f1=f1_score(
            gold_norm, pred_norm, labels=label_list, average="weighted", zero_division=0
        ),
        per_class=per_class,
        n_samples=len(gold_norm),
        n_invalid=n_invalid,
    )
