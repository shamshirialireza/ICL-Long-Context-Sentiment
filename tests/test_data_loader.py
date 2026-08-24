from pathlib import Path

import pytest

from icl_sentiment.data.loader import DatasetConfig, DatasetLoader


def test_load_csv(sample_csv: Path):
    config = DatasetConfig(name="sample", path=str(sample_csv))
    dataset = DatasetLoader().load(config)

    assert len(dataset) == 9
    assert dataset.name == "sample"
    assert dataset.labels == [
        "positive",
        "neutral",
        "negative",
        "positive",
        "neutral",
        "negative",
        "positive",
        "neutral",
        "negative",
    ]


def test_labels_are_lowercased(sample_csv: Path):
    dataset = DatasetLoader().load(DatasetConfig(name="sample", path=str(sample_csv)))
    assert all(label == label.lower() for label in dataset.labels)


def test_missing_file_raises():
    config = DatasetConfig(name="missing", path="does/not/exist.csv")
    with pytest.raises(FileNotFoundError):
        DatasetLoader().load(config)


def test_missing_column_raises(tmp_path: Path):
    import pandas as pd

    path = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["hi"], "label": ["positive"]}).to_csv(path, index=False)
    config = DatasetConfig(name="bad", path=str(path), text_column="text")
    with pytest.raises(KeyError):
        DatasetLoader().load(config)


def test_drop_rows_and_limit(sample_csv: Path):
    config = DatasetConfig(name="sample", path=str(sample_csv), drop_rows=[0], limit=3)
    dataset = DatasetLoader().load(config)
    assert len(dataset) == 3
    assert "I absolutely loved this, best purchase ever." not in dataset.texts
