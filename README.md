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

Install the monitor dependencies, then open the latest execution of `demo`:

```bash
uv sync --extra monitor
uv run mammoth monitor demo
```

The monitor reads from `./runs` by default. Use `--entry <path>` to inspect a
different artifact root.

On an interactive terminal, this opens the Textual dashboard with continuous
two-second polling and explicitly viewer-host telemetry enabled. The dashboard
includes compact attempt IDs, resume-aware run progress, integrated progress
bars, throughput, ETA, rank state, lineage-rich attempt history, and responsive
multi-row Braille charts for conventional loss and learning-rate metrics. Exact
detail retains complete immutable IDs and provenance.

Use the arrow keys or `j`/`k` to select an execution, Enter to toggle overview
and detail, `r` to refresh, and `q` to quit. Pass `--execution <execution-id>`
to pin one immutable attempt exactly.

Redirected output remains a single stable ANSI-free snapshot:

```bash
uv run mammoth monitor demo > monitor.txt
```

Interactive invocations can opt out of individual defaults:

```bash
uv run mammoth monitor demo --plain
uv run mammoth monitor demo --no-watch
uv run mammoth monitor demo --no-telemetry
uv run mammoth monitor demo --interval 5 --stale-after 120
```

The existing `--rich`, `--watch`, and `--telemetry` flags remain accepted for
explicit compatibility. Plain mode works without the monitor extra; requesting
the dashboard without its optional dependencies reports the exact installation
command.

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

## Compose execution observability

For a process that already has an `ExecutionContext`, create JSONL and optional
TensorBoard output without claiming its text log:

```python
from mammoth.logging import create_execution_observability
from mammoth.logging.tensorboard import TensorBoardSink

tensorboard = TensorBoardSink(context.run_dir / "logs", rank=rank)
with create_execution_observability(
    context,
    rank=rank,
    additional_sinks=(tensorboard,),
) as observability:
    with observability.observer.periodic_heartbeats(phase="train"):
        observability.observer.progress(
            phase="train",
            task_id="epoch-0",
            completed=1,
            total=10,
            metrics={"train/loss": 0.5},
            coordinates={"epoch": 0, "batch": 0},
            logical_step=0,
        )
```

Use `create_execution_logging` instead when Mammoth should also own the
process-exclusive plain-text handler. Applications attach that returned handler
to their chosen Python logger.

## Optional integrations

Install only the features your project uses:

```bash
uv sync --extra monitor       # Textual dashboard, Rich renderables, and psutil telemetry
uv sync --extra tensorboard   # TensorBoard logging sink
uv sync --extra torch         # Generic PyTorch runtime, trainer, and profiler
```

Multiple extras may be installed together:

```bash
uv sync --extra monitor --extra tensorboard --extra torch
```

## Profile an arbitrary PyTorch workload

The optional profiler measures a caller-owned zero-argument callable, so the
project retains its model construction, inputs, call signature, contexts, and
output meaning:

```python
from mammoth.torch import ProfileConfig, profile_callable, write_profile_report

report = profile_callable(
    lambda: model(images, prompts=prompts),
    config=ProfileConfig(
        device="cuda:0",
        work_units_per_iteration=len(images),
        work_unit="image",
    ),
    components={"encoder": model.encoder, "decoder": model.decoder},
)
write_profile_report(run_dir / "profile.json", report)
```

Cold-start timing, synchronized steady-state latency, caller-labelled
throughput, CUDA allocator peaks, explicit component ranges, normalized
operation rows with optional input shapes, FLOPs, and memory, and optional
Chrome traces remain operational evidence.
Projects can replace the generic nested-tensor output summary when they need a
semantic comparison such as predicted classes or generated-token checks.

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

`WarmupLinearLR` supplies a reusable optimizer-step schedule with linear
warmup and decay. Projects choose its warmup ratio and total-step horizon; an
extended resume rebases optimizer learning rates, while a shortened horizon is
rejected.

Checkpoint policies expose a two-phase typed restore path. Call
`trainer.inspect_checkpoint(path)` to receive a rank-coordinated
`CheckpointInspection`, then pass its `restore_options` to
`trainer.load_checkpoint(path, options=...)`. Mammoth applies and synchronizes
generic optimizer, scheduler, and callback restore/reset actions, trainer-coordinate
restoration, and terminal-state restore/reset; the project keeps its payload parsing, model
compatibility, and metadata policy.

Projects that need more control can supply an `AccumulationPolicy`, scalar
`MetricSpec` values, additive `StatefulMetric` objects, `MetricRoute` mappings,
and a `TrainerCheckpointPolicy`. The train and validation step functions remain
project-owned; Mammoth uses these policies to coordinate logical optimizer
steps, distributed reductions, generic observations, and ordered checkpoint
publication.

Set `TrainerConfig.compile_config` to a `TorchCompileConfig` when the ordinary
forward path should be compiled. Mammoth keeps the supplied module as
`trainer.base_model`, wraps it with DDP when requested, and only then derives
`trainer.execution_model` with `torch.compile`; accumulation still targets the
underlying DDP wrapper's `no_sync()` context.

For heterogeneous ranks, `WeightedAccumulationPolicy` and
`WeightedDistributedBatchSampler` accept arbitrary caller-defined rank weights.
The matching `weighted_partition_counts` and `weighted_partition_indices`
helpers can shard other opaque workloads. `allocate_weighted_tasks` assigns
opaque task IDs with caller-estimated costs by projected normalized rank load.
Projects still choose the weights, eligible hardware, datasets, task-cost
meaning, and `DataLoader` settings.

Set `TrainerConfig(emit_fit_phase_events=False, ...)` when a surrounding
command already owns the outer training phase lifecycle. Mammoth continues to
emit nested tasks, progress, validation phases, heartbeats, and metrics.

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
lifecycle. Project-specific GPU validation, concrete workload policy, and
custom collectives stay in the consuming project.

Projects with custom checkpoint formats can use the same bounded publisher
without adopting Mammoth's checkpoint schema. A plan stages every artifact
before committing its destinations in the declared order:

```python
from mammoth.torch import (
    AsyncCheckpointPublisher,
    CheckpointArtifact,
    CheckpointPlan,
)

plan = CheckpointPlan(
    checkpoint_root=checkpoint_dir,
    artifacts=(
        CheckpointArtifact(best_path, write_inference_checkpoint),
        CheckpointArtifact(latest_path, write_resume_checkpoint),
    ),
    retire_after_commit=(previous_latest_path,),
)

with AsyncCheckpointPublisher(max_pending=1) as publisher:
    publisher.submit(plan)
```

Serialization callbacks and retention choices remain project-owned. Ordered
replacement is crash-safe per file but is not a multi-file transaction, so a
project should place its commit-marker artifact last. Mammoth gives each
serializer a descriptor-anchored path inside a private staging directory and
atomically moves the completed file to its destination. Artifact modes default
to `0o600`; `mode=None` retains serializer-created permissions for a new file
while still preserving an existing regular destination's mode. This confined
durability path requires POSIX descriptor-relative filesystem operations and raises
`NotImplementedError` before publication when they are unavailable.

Projects can also apply process-global Torch numerical settings without using
the trainer or profiler. `TorchBackendConfig` controls TF32, float32 matmul
precision, cuDNN benchmarking, and deterministic modes; `TorchSeedPolicy`
selects which generic Python and Torch generators receive the caller's seed.
Mammoth does not choose those values or configure dataset workers.

## Learn more

- [Architecture](docs/ARCHITECTURE.md) explains package boundaries, runtime
  concepts, artifact layout, and compatibility.
- [Code map](docs/CODEMAP.md) points contributors to implemented modules and
  public symbols.
- [Agent rules](AGENTS.md) describes repository conventions and validation.
