# Mammoth

Mammoth is project-independent infrastructure for running, training, logging,
and monitoring AI workloads. It provides operational mechanics without owning
model architecture, dataset, loss, or domain-metric semantics.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import mammoth; print(mammoth.__version__)"
```

Install optional integrations only where they are needed:

```bash
uv sync --extra monitor --extra tensorboard --extra torch
```

Inspect the Typer command tree or print the installed version:

```bash
uv run mammoth --help
uv run mammoth --version
```

Inspect the latest attempt for a run, or select one immutable attempt exactly:

```bash
uv run mammoth monitor <run-name> --entry <entry>
uv run mammoth monitor <run-name> --entry <entry> --execution <execution-id>
```

Add `--watch --rich` for the optional interactive view. JSONL remains the
machine-readable live state; plain monitor snapshots never contain ANSI escape
sequences.

Run project-defined commands from a strict workflow file:

```yaml
schema_version: 1
runs:
  example-run:
    steps:
      prepare:
        command: [python, prepare.py]
      train:
        command: [python, train.py]
        needs: [prepare]
```

```bash
uv run mammoth workflow run workflow.yaml --entry runs --dry-run
uv run mammoth workflow run workflow.yaml --entry runs
```

Use `--run <name>` and `--step <name>` for exact selection. Selecting a step
also selects its transitive dependencies. Dry runs validate and print resolved
commands without creating run artifacts.

The optional trainer accepts constructed PyTorch objects and a project step
function:

```python
from mammoth.torch import StepOutput, Trainer, TrainerConfig

def train_step(model, batch, context):
    inputs, targets = batch
    prediction = model(inputs)
    loss = project_loss(prediction, targets)
    return StepOutput(loss=loss, metrics={"project/score": project_score(prediction, targets)})

with Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    train_step=train_step,
    config=TrainerConfig(epochs=10),
) as trainer:
    result = trainer.fit()
```

Mammoth does not construct those objects or define the batch, loss, metric, or
checkpoint-compatibility semantics.

Repository documentation begins with [AGENTS.md](AGENTS.md).
