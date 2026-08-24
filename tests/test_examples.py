import pytest

from icl_sentiment.data.loader import Dataset, DatasetConfig
from icl_sentiment.prompting.examples import ExampleSelector


def _dataset() -> Dataset:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "text": [f"doc-{i}" for i in range(9)],
            "label": ["positive"] * 3 + ["neutral"] * 3 + ["negative"] * 3,
        }
    )
    return Dataset(DatasetConfig(name="d", path="unused.csv"), frame)


def test_zero_shot_returns_no_examples():
    selector = ExampleSelector(seed=1)
    assert selector.select(_dataset(), 0) == []


def test_three_shot_is_stratified_one_per_class():
    selector = ExampleSelector(seed=1)
    examples = selector.select(_dataset(), 3)
    labels = sorted(example.label for example in examples)
    assert labels == ["negative", "neutral", "positive"]


def test_six_shot_is_stratified_two_per_class():
    selector = ExampleSelector(seed=1)
    examples = selector.select(_dataset(), 6)
    labels = sorted(example.label for example in examples)
    assert labels == ["negative", "negative", "neutral", "neutral", "positive", "positive"]


def test_uneven_shot_count_raises():
    selector = ExampleSelector(seed=1)
    with pytest.raises(ValueError):
        selector.select(_dataset(), 4)


def test_insufficient_examples_raises():
    selector = ExampleSelector(seed=1)
    with pytest.raises(ValueError):
        selector.select(_dataset(), 12)


def test_selection_is_reproducible_with_same_seed():
    dataset = _dataset()
    first = ExampleSelector(seed=7).select(dataset, 6)
    second = ExampleSelector(seed=7).select(dataset, 6)
    assert [e.text for e in first] == [e.text for e in second]
