# Mammoth

Mammoth is project-independent infrastructure for running, training, logging,
and monitoring AI workloads. Your project still defines the model, data, loss,
metrics, and commands; Mammoth handles the repeatable operational work around
them.

You can use Mammoth to:

- run commands in dependency order from a YAML file;
- keep each attempt and its logs under a predictable artifact directory;
- inspect a completed run or watch one while it is active; and
- add a reusable training loop to an existing PyTorch project.

No familiarity with Mammoth is assumed below.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

From the repository root, install the base dependencies and confirm the CLI is
available:

```bash
uv sync
uv run mammoth --version
uv run mammoth --help
```

`uv sync` creates the local environment. Prefixing a command with `uv run`
runs it inside that environment, so manual virtual-environment activation is
not required.

## Run your first workflow

A workflow groups commands into named runs and steps. Create `workflow.yaml`:

```yaml
schema_version: 1
runs:
  demo:
    steps:
      prepare:
        command: [python, -c, "print('Preparing data')"]
      train:
        command: [python, -c, "print('Training model')"]
        needs: [prepare]
```

First validate the file and see the commands without running them:

```bash
uv run mammoth workflow run workflow.yaml --entry runs --dry-run
```

Then execute the workflow:

```bash
uv run mammoth workflow run workflow.yaml --entry runs
```

The command reports whether `demo` completed and prints its execution ID. The
`--entry runs` option tells Mammoth to store operational artifacts beneath the
`runs/` directory. In this example, they appear under:

```text
runs/demo/
├── manifest.json
└── logs/executions/<execution-id>/
```

The names have a simple hierarchy:

- **entry**: the artifact root supplied with `--entry`, such as `runs`;
- **run**: a reusable logical job, such as `demo`;
- **step**: one command within a run, such as `prepare`; and
- **execution**: one immutable attempt to run that job.

Running `demo` again creates a new execution instead of overwriting the old
one.

## Inspect or watch a run

Show the latest execution of `demo`:

```bash
uv run mammoth monitor demo --entry runs
```

To keep refreshing until an active execution finishes, use:

```bash
uv run mammoth monitor demo --entry runs --watch
```

For the optional interactive display, install the monitor dependencies and add
`--rich`:

```bash
uv sync --extra monitor
uv run mammoth monitor demo --entry runs --watch --rich
```

Pass `--execution <execution-id>` to inspect an older attempt exactly. Add
`--telemetry` when you also want CPU and memory information from the computer
running the monitor.

## Select only part of a workflow

Use exact run and step names to narrow an invocation:

```bash
uv run mammoth workflow run workflow.yaml --entry runs --run demo
uv run mammoth workflow run workflow.yaml --entry runs --run demo --step train
```

Selecting `train` also selects its dependency, `prepare`. Repeat `--run` or
`--step` to select more than one name.

Common step settings include:

| Setting | Purpose |
| --- | --- |
| `command` | Required list containing the program and each argument. |
| `needs` | Steps that must run before this step. |
| `cwd` | Working directory, resolved relative to the workflow file. |
| `environment` | Explicit environment variables for the command. |
| `timeout_seconds` | Positive time limit for the command. |
| `on_failure` | `stop` (default) or `continue` with later steps. |
| `launcher` | `local` (default) or `torchrun`. |
| `processes` | Number of processes for a `torchrun` step. |

Mammoth validates unknown settings, missing dependencies, and dependency
cycles before launching commands. Run `uv run mammoth workflow run --help` for
the full CLI reference.

## Optional integrations

Install only the features your project uses:

```bash
uv sync --extra monitor       # Rich display and psutil telemetry
uv sync --extra tensorboard   # TensorBoard logging sink
uv sync --extra torch         # Generic PyTorch runtime and trainer
```

Multiple extras may be installed together:

```bash
uv sync --extra monitor --extra tensorboard --extra torch
```

## Use the PyTorch trainer

The optional trainer fits into a project that already constructs its model,
optimizer, data loaders, and loss. The project supplies a step function that
explains how to interpret one batch:

```python
import torch

from mammoth.torch import StepOutput, Trainer, TrainerConfig


def train_step(model, batch, context):
    inputs, targets = batch
    prediction = model(inputs)
    loss = torch.nn.functional.cross_entropy(prediction, targets)
    accuracy = (prediction.argmax(dim=1) == targets).float().mean()
    return StepOutput(loss=loss, metrics={"accuracy": accuracy})


with Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    train_step=train_step,
    config=TrainerConfig(epochs=10),
) as trainer:
    result = trainer.fit()
```

Mammoth owns ordinary loop mechanics such as device movement, precision,
gradient accumulation, validation, callbacks, scheduling, and checkpoints. It
does not choose your architecture, dataset, batch format, loss, or metric
meaning.

For a direct single-process or `torchrun` invocation, initialize Mammoth's
runtime before the trainer. Rank zero creates the immutable execution, every
rank joins it and receives its own JSONL and text stream, and the trainer uses
the runtime's device and rank identity:

```python
from pathlib import Path

from mammoth.torch import (
    TorchExecutionRequest,
    TorchRuntimeConfig,
    Trainer,
    TrainerConfig,
    initialize_torch_runtime,
)

runtime_config = TorchRuntimeConfig(strategy="ddp", device="auto")
with initialize_torch_runtime(runtime_config) as runtime:
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=Path("runs/example"),
            run_name="example",
            invocation_kind="train",
            intended_phases=("train",),
            command=("python", "train.py"),
        )
    )
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step=train_step,
        config=TrainerConfig(epochs=10, strategy="ddp"),
        runtime=runtime,
    ) as trainer:
        result = trainer.fit()
```

Use `strategy="single"` without `torchrun` for the corresponding local
lifecycle. Project-specific GPU validation, sampler policy, and custom
collectives stay in the consuming project.

## Learn more

- [Architecture](docs/ARCHITECTURE.md) explains package boundaries, runtime
  concepts, artifact layout, and compatibility.
- [Code map](docs/CODEMAP.md) points contributors to implemented modules and
  public symbols.
- [Agent rules](AGENTS.md) describes repository conventions and validation.
