# icl-sentiment

A dynamic, object-oriented framework for benchmarking **in-context learning (zero-shot and
few-shot) sentiment classification** across multiple LLM providers and multiple datasets.

It reimplements the methodology from
[*In-Context Learning for Long-Context Sentiment Analysis on Infrastructure Project Opinions*](https://doi.org/10.48550/arXiv.2410.11265)
as a reusable, general-purpose tool: any text/label CSV works, not just infrastructure or
transportation data, and there is nothing dataset-specific hardcoded anywhere in the codebase.

## Why this exists

The original research code was a single 566-cell notebook: hardcoded Google Drive paths,
hardcoded API keys, hardcoded few-shot example text pasted separately into every
dataset/shot/model combination, and hand-rolled TP/FP counting copy-pasted dozens of times.
This project restructures that pipeline into small, testable, swappable components:

| Concern | Notebook (before) | This repo (after) |
|---|---|---|
| Datasets | Hardcoded `pd.read_csv('/content/drive/...')` per cell | `DatasetLoader` + `DatasetConfig`, any CSV/JSON/Parquet |
| Models | Copy-pasted API calls per dataset/shot | `SentimentProvider` classes registered in a `ProviderRegistry` |
| Few-shot examples | Hand-picked strings duplicated ~40 times | `ExampleSelector` samples class-balanced examples dynamically |
| Prompts | Hardcoded per-dataset prompt strings | `PromptBuilder`, domain-neutral and configurable |
| Metrics | Manual TP/FP/FN counters | `sklearn`-backed accuracy, macro-F1, micro-F1, per-class report |
| Dataset stats | Hardcoded numbers pasted into a plotting cell | Computed dynamically from the actual data |
| Secrets | API keys committed in plaintext | Read from environment variables |
| Execution | Sequential `while` loop, no resume | Checkpointed per row, resumable, retry with backoff |

## Features

- **Provider-agnostic**: OpenAI, Anthropic, and Google Gemini ship out of the box; add a new
  backend by subclassing `SentimentProvider` and registering it — the runner and CLI pick it up
  automatically.
- **Config-driven**: one YAML file declares datasets, providers, and shot counts; the runner
  sweeps the full cartesian product.
- **Domain-neutral prompts**: label set and document noun ("review", "comment", "post", ...) are
  configurable, so the same code works for product reviews, social media, support tickets, or the
  original infrastructure-project comments.
- **Balanced few-shot sampling**: examples are drawn dynamically from the dataset, stratified
  evenly across classes, and excluded from the scored evaluation set (no train/test leakage).
- **Resumable runs**: every prediction is checkpointed to disk as it's made, so a rate limit or
  crash never loses completed work — re-running picks up where it left off.
- **Two report tables out of the box**:
  1. Accuracy / macro-F1 / micro-F1 for every `(dataset, model, shot count)` combination.
  2. Descriptive statistics (`No.`, `Mean`, `SD`, `Min.`, `25th Pctl.`, `Median`, `75th Pctl.`,
     `Max.`) of document length for every dataset, computed from the actual data.

## Installation

```bash
git clone <this-repo-url>
cd icl-sentiment
pip install -e ".[all,dev]"   # or [openai], [anthropic], [gemini] individually
```

Copy `.env.example` to `.env` and fill in the API key(s) for the providers you plan to use, then
`export $(cat .env | xargs)` (or use a tool like `direnv`/`python-dotenv` in your own scripts).

## Quickstart

Two small, generic sample datasets ship in `data/sample/` (product reviews, restaurant feedback)
so you can try the whole pipeline immediately.

### CLI

```bash
# Descriptive statistics for every configured dataset
icl-sentiment stats configs/experiment.example.yaml

# Run the full benchmark matrix (datasets x providers x shot counts)
icl-sentiment run configs/experiment.example.yaml

# List available model providers
icl-sentiment providers
```

`run` writes per-row predictions to `results/checkpoints/` (so it is safe to Ctrl-C and resume)
and a summary table to `results/results.csv` with `accuracy`, `macro_f1`, `micro_f1`, and
`n_invalid` (unparseable model outputs) for every combination.

### Python API

```python
from icl_sentiment import (
    DatasetConfig, ProviderConfig, ExperimentConfig, ExperimentRunner,
    model_dataset_table, describe_datasets, DatasetLoader,
)

config = ExperimentConfig(
    datasets=[
        DatasetConfig(name="reviews", path="data/sample/product_reviews.csv"),
    ],
    providers=[
        ProviderConfig(name="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"),
        ProviderConfig(name="anthropic", model="claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"),
    ],
    shot_counts=[0, 3, 6, 9],
    document_noun="review",
)

results = ExperimentRunner(config).run()
print(model_dataset_table(results))   # accuracy + macro-F1 per model per dataset

datasets = DatasetLoader().load_all(config.datasets)
print(describe_datasets(datasets))    # No. / Mean / SD / Min. / 25th / Median / 75th / Max.
```

## Bring your own dataset

Any CSV, JSON, JSON Lines, or Parquet file with a text column and a label column works:

```yaml
datasets:
  - name: my_dataset
    path: data/my_dataset.csv
    text_column: text        # defaults to "text"
    label_column: label      # defaults to "label"
    drop_rows: [12, 47]       # optional, e.g. known bad annotations
    limit: 500                # optional, cap rows for a quick smoke test
```

Labels are lowercased and whitespace-trimmed automatically. The label set used for prompting and
scoring is configured once, globally:

```yaml
labels: [positive, neutral, negative]
```

Few-shot counts must be evenly divisible by the number of labels (`ExampleSelector` stratifies
examples one class at a time); use e.g. `shot_counts: [0, 2, 4]` for a 2-class dataset.

## Adding a new model provider

```python
from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry, SentimentProvider

@ProviderRegistry.register("my-provider")
class MyProvider(SentimentProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = MyClientSDK(api_key=config.resolve_api_key())

    def _complete(self, prompt: str) -> str:
        return self._client.generate(prompt, model=self.config.model)
```

Retries, exponential backoff, and rate-limit pacing are handled by the base class — a provider
only needs to implement `_complete`. Reference `my-provider` from any YAML config's `providers:`
list and it works with the CLI and runner immediately.

## Architecture

```
icl_sentiment/
├── data/loader.py           DatasetConfig, DatasetLoader, Dataset
├── prompting/
│   ├── examples.py          ExampleSelector — class-balanced few-shot sampling
│   └── templates.py         PromptBuilder — domain-neutral zero/few-shot prompts
├── providers/
│   ├── base.py               SentimentProvider (ABC) + ProviderRegistry + retry/backoff
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── gemini_provider.py
├── evaluation/
│   ├── metrics.py            normalize_label, evaluate_predictions (accuracy, macro/micro-F1)
│   └── statistics.py         describe_dataset / describe_datasets (length statistics)
├── runner.py                 ExperimentConfig, ExperimentRunner — checkpointed orchestration
├── reports.py                 model_dataset_table, summarize — result reshaping
└── cli.py                     `icl-sentiment run|stats|providers`
```

## Testing

```bash
pytest -v
```

Tests run fully offline using a deterministic `FakeProvider` (registered in `tests/conftest.py`),
so the whole pipeline — data loading, example selection, prompt building, checkpointing, metrics,
and the CLI — is exercised in CI without needing any API keys.

## Citation

If you use this framework, please cite the original paper:

```bibtex
@article{icl_sentiment_infra_2024,
  title   = {In-Context Learning for Long-Context Sentiment Analysis on Infrastructure Project Opinions},
  journal = {arXiv preprint arXiv:2410.11265},
  year    = {2024},
  doi     = {10.48550/arXiv.2410.11265}
}
```

## License

MIT — see [LICENSE](LICENSE).
