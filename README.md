# In Context Learning for Long-Context Sentiment Classification

A dynamic framework for benchmarking **in-context learning (zero-shot and
few-shot) for long context sentiment classification** across multiple LLM providers and multiple datasets.

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
git clone https://github.com/shamshirialireza/icl-sentiment.git
cd icl-sentiment
pip install -e ".[all,dev]"   # or [openai], [anthropic], [gemini] individually
```

Copy `.env.example` to `.env` and fill in the API key(s) for the providers you plan to use, then
`export $(cat .env | xargs)` (or use a tool like `direnv`/`python-dotenv` in your own scripts).

## Quickstart

Two small, generic sample datasets ship in `data/sample/` (news articles, Facebook comments)
so you can try the whole pipeline immediately.

### CLI

```bash
# Descriptive statistics for every configured dataset
ICL-Long-Context-Sentiment stats configs/experiment.example.yaml

# Run the full benchmark matrix (datasets x providers x shot counts)
ICL-Long-Context-Sentiment run configs/experiment.example.yaml

# List available model providers
ICL-Long-Context-Sentiment providers
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
        DatasetConfig(name="news", path="data/sample/News.csv"),
    ],
    providers=[
        ProviderConfig(name="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"),
        ProviderConfig(name="anthropic", model="claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY", max_tokens=200),
        ProviderConfig(name="gemini", model="gemini-3.6-flash", api_key_env="GOOGLE_API_KEY", max_tokens=500),
    ],
    shot_counts=[0, 3, 6, 9],
    document_noun="review",
)

results = ExperimentRunner(config).run()
print(model_dataset_table(results))   # accuracy + macro-F1 per model per dataset

datasets = DatasetLoader().load_all(config.datasets)
print(describe_datasets(datasets))    # No. / Mean / SD / Min. / 25th / Median / 75th / Max.
```

### Google Colab Setup

The benchmark runs fine on Colab's free CPU runtime — all model inference happens on the
providers' APIs, so no GPU is needed. Shell commands work in notebook cells with a `!` prefix:

```python
# Cell 1 — get the code and install it
import os

!git clone https://github.com/shamshirialireza/ICL-Long-Context-Sentiment.git
os.chdir('ICL-Long-Context-Sentiment')
!pip install -e ".[all,dev]"   # or [openai], [anthropic], [gemini] individually

# Cell 2 — Insert API keys and values via Colab Secrets FIRST (the 🔑 icon in the left sidebar).
# Don't paste keys directly into cells if you share the colab notebook.

from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# Cell 3 — same CLI as local usage
!ICL-Long-Context-Sentiment providers
!ICL-Long-Context-Sentiment stats configs/experiment.example.yaml
!ICL-Long-Context-Sentiment run configs/experiment.example.yaml
```

Colab VMs are ephemeral: `results/` is wiped when the runtime disconnects, so download what you
need (or write `output_dir` to a mounted Google Drive):

```python
from google.colab import files
files.download("results/results.csv")
```

The [Python API](#python-api) works in cells too, and is often nicer in a notebook — the report
tables are pandas DataFrames, so they render as proper tables instead of printed text.

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
└── cli.py                     `ICL-Long-Context-Sentiment run|stats|providers`
```

## Testing

```bash
pytest -v
```

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
