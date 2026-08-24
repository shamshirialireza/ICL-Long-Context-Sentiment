from pathlib import Path

from icl_sentiment.data.loader import DatasetConfig
from icl_sentiment.providers.base import ProviderConfig
from icl_sentiment.reports import model_dataset_table, summarize
from icl_sentiment.runner import ExperimentConfig, ExperimentRunner


def _config(sample_csv: Path, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=[DatasetConfig(name="sample", path=str(sample_csv))],
        providers=[ProviderConfig(name="fake", model="fake-v1", max_retries=0)],
        shot_counts=[0, 3],
        output_dir=str(tmp_path / "results"),
    )


def test_runner_produces_one_row_per_combination(sample_csv: Path, tmp_path: Path):
    config = _config(sample_csv, tmp_path)
    results = ExperimentRunner(config).run(show_progress=False)

    assert len(results) == 2  # one dataset x one provider x two shot counts
    assert set(results["n_shots"]) == {0, 3}
    assert {"accuracy", "macro_f1", "micro_f1"}.issubset(results.columns)


def test_runner_writes_checkpoints(sample_csv: Path, tmp_path: Path):
    config = _config(sample_csv, tmp_path)
    ExperimentRunner(config).run(show_progress=False)

    checkpoints = list((tmp_path / "results" / "checkpoints").glob("*.jsonl"))
    assert len(checkpoints) == 2


def test_runner_resumes_from_checkpoint(sample_csv: Path, tmp_path: Path):
    config = _config(sample_csv, tmp_path)
    config.shot_counts = [0]
    runner = ExperimentRunner(config)
    runner.run(show_progress=False)

    checkpoint = next((tmp_path / "results" / "checkpoints").glob("*0shot.jsonl"))
    lines_before = checkpoint.read_text().count("\n")

    # Re-running with a fresh runner should reuse the checkpoint, not duplicate work.
    ExperimentRunner(config).run(show_progress=False)
    lines_after = checkpoint.read_text().count("\n")
    assert lines_before == lines_after


def test_few_shot_excludes_demonstrations_from_scoring(sample_csv: Path, tmp_path: Path):
    config = _config(sample_csv, tmp_path)
    config.shot_counts = [3]
    results = ExperimentRunner(config).run(show_progress=False)
    # 9 total rows, 3 used as demonstrations -> 6 scored
    assert results.iloc[0]["n_samples"] == 6


def test_report_helpers_shape_results(sample_csv: Path, tmp_path: Path):
    config = _config(sample_csv, tmp_path)
    results = ExperimentRunner(config).run(show_progress=False)

    table = model_dataset_table(results)
    assert list(table.columns) == [
        "dataset",
        "provider",
        "model",
        "n_shots",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "n_samples",
        "n_invalid",
    ]

    pivot = summarize(results, metric="macro_f1")
    assert "sample" in pivot.index
