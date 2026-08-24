"""In-context learning benchmark framework for sentiment analysis with LLMs."""

from icl_sentiment.data.loader import Dataset, DatasetConfig, DatasetLoader
from icl_sentiment.evaluation.metrics import evaluate_predictions
from icl_sentiment.evaluation.statistics import describe_dataset, describe_datasets
from icl_sentiment.prompting.examples import ExampleSelector
from icl_sentiment.prompting.templates import PromptBuilder
from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider
from icl_sentiment.reports import model_dataset_table, summarize
from icl_sentiment.runner import ExperimentConfig, ExperimentRunner

__version__ = "1.0.0"

__all__ = [
    "Dataset",
    "DatasetConfig",
    "DatasetLoader",
    "ExampleSelector",
    "ExperimentConfig",
    "ExperimentRunner",
    "PromptBuilder",
    "ProviderConfig",
    "ProviderRegistry",
    "SentimentProvider",
    "describe_dataset",
    "describe_datasets",
    "evaluate_predictions",
    "model_dataset_table",
    "summarize",
]
