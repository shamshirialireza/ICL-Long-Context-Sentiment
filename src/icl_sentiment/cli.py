"""Command-line interface.

    icl-sentiment run config.yaml --output results/
    icl-sentiment stats config.yaml
    icl-sentiment providers
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from icl_sentiment.data.loader import DatasetLoader
from icl_sentiment.evaluation.statistics import describe_datasets
from icl_sentiment.providers.base import ProviderRegistry
from icl_sentiment.reports import model_dataset_table
from icl_sentiment.runner import ExperimentConfig, ExperimentRunner


def _cmd_run(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    if args.output:
        config.output_dir = args.output

    runner = ExperimentRunner(config)
    results = runner.run(show_progress=not args.quiet)

    table = model_dataset_table(results)
    out_path = Path(config.output_dir) / "results.csv"
    table.to_csv(out_path, index=False)

    pd.set_option("display.width", 120)
    print(table.to_string(index=False))
    print(f"\nSaved results to {out_path}")


def _cmd_stats(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    loader = DatasetLoader()
    datasets = loader.load_all(config.datasets)
    table = describe_datasets(datasets, unit=args.unit)

    pd.set_option("display.width", 120)
    print(table.to_string())

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        print(f"\nSaved statistics to {args.output}")


def _cmd_providers(_: argparse.Namespace) -> None:
    for name in ProviderRegistry.available():
        print(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icl-sentiment", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark defined by a YAML config")
    run_parser.add_argument("config", help="Path to experiment YAML config")
    run_parser.add_argument("--output", help="Override the config's output_dir")
    run_parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    run_parser.set_defaults(func=_cmd_run)

    stats_parser = subparsers.add_parser("stats", help="Print dataset descriptive statistics")
    stats_parser.add_argument("config", help="Path to experiment YAML config")
    stats_parser.add_argument("--unit", choices=["chars", "words"], default="chars")
    stats_parser.add_argument("--output", help="Optional CSV path to save the table")
    stats_parser.set_defaults(func=_cmd_stats)

    providers_parser = subparsers.add_parser("providers", help="List registered model providers")
    providers_parser.set_defaults(func=_cmd_providers)

    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
