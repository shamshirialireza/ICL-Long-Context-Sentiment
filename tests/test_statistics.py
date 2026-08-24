import pandas as pd

from icl_sentiment.data.loader import Dataset, DatasetConfig
from icl_sentiment.evaluation.statistics import STAT_COLUMNS, describe_dataset, describe_datasets


def _dataset(texts, name="d") -> Dataset:
    frame = pd.DataFrame({"text": texts, "label": ["positive"] * len(texts)})
    return Dataset(DatasetConfig(name=name, path="unused.csv"), frame)


def test_describe_dataset_char_stats():
    dataset = _dataset(["ab", "abcd", "abcdef"])
    stats = describe_dataset(dataset, unit="chars")
    assert stats["No."] == 3
    assert stats["Min."] == 2
    assert stats["Max."] == 6
    assert stats["Median"] == 4


def test_describe_dataset_word_stats():
    dataset = _dataset(["one two", "one two three four"])
    stats = describe_dataset(dataset, unit="words")
    assert stats["Min."] == 2
    assert stats["Max."] == 4


def test_describe_dataset_invalid_unit_raises():
    import pytest

    with pytest.raises(ValueError):
        describe_dataset(_dataset(["a"]), unit="tokens")


def test_describe_datasets_returns_one_row_per_dataset():
    datasets = [_dataset(["ab", "abc"], name="first"), _dataset(["a"], name="second")]
    table = describe_datasets(datasets)
    assert list(table.index) == ["first", "second"]
    assert list(table.columns) == STAT_COLUMNS
