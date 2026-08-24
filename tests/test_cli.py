from pathlib import Path

import yaml

from icl_sentiment import cli


def _write_config(sample_csv: Path, tmp_path: Path) -> Path:
    config = {
        "labels": ["positive", "neutral", "negative"],
        "shot_counts": [0],
        "output_dir": str(tmp_path / "results"),
        "datasets": [{"name": "sample", "path": str(sample_csv)}],
        "providers": [{"name": "fake", "model": "fake-v1", "max_retries": 0}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_cli_run_writes_results_csv(sample_csv: Path, tmp_path: Path, capsys):
    config_path = _write_config(sample_csv, tmp_path)
    cli.main(["run", str(config_path), "--quiet"])

    output = capsys.readouterr().out
    assert "sample" in output
    assert (tmp_path / "results" / "results.csv").exists()


def test_cli_stats_prints_table(sample_csv: Path, tmp_path: Path, capsys):
    config_path = _write_config(sample_csv, tmp_path)
    cli.main(["stats", str(config_path)])

    output = capsys.readouterr().out
    assert "Mean" in output
    assert "sample" in output


def test_cli_providers_lists_fake(capsys):
    cli.main(["providers"])
    output = capsys.readouterr().out
    assert "fake" in output.splitlines()
