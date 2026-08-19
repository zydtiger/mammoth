# Mammoth

[![CI](https://github.com/zydtiger/mammoth/actions/workflows/ci.yml/badge.svg)](https://github.com/zydtiger/mammoth/actions/workflows/ci.yml)

Mammoth is project-independent infrastructure for running, training, logging,
and monitoring AI workloads. Your project still defines the model, data, loss,
metrics, and commands; Mammoth handles the repeatable operational work around
them.

You can use Mammoth to:

- plan and run serial command workflows from Python;
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

A workflow groups final command arguments into named runs and steps. Define it
in Python, inspect the side-effect-free plan, then run it:

```python
from pathlib import Path

from mammoth.workflow import Run, Step, Workflow

workflow = Workflow(
    root=Path("runs"),
    runs=(
        Run(
            "demo",
            steps=(
                Step("prepare", ("python", "-c", "print('Preparing data')")),
                Step("train", ("python", "-c", "print('Training model')")),
            ),
        ),
    ),
)

for command in workflow.plan():
    print(command.run_name, command.step_name, command.command)

result = workflow.run()
raise SystemExit(result.exit_code)
```

`Workflow.plan()` validates names and ordering and derives each `RunLayout`
without creating directories, leases, metadata, or lifecycle events.
`Workflow.run()` stores each activated run beneath the workflow root. In this
example, operational artifacts appear under:

```text
runs/demo/
├── checkpoints/
├── results/
├── vis/
└── logs/executions/<execution-id>/
```

The names have a simple hierarchy:

- **root**: the workflow artifact root, such as `runs`;
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

Progress counters are unitless producer-owned numbers. Within each task,
`completed`, `total`, and any reported throughput must describe the same
logical work quantity; producers omit throughput when they cannot keep that
relationship. The monitor renders every present throughput as `b/s` without
defining or converting the underlying work batch.

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

## Identify an existing artifact's exact bytes

`mammoth.core` can record and later verify the raw bytes of one local regular
file without assigning it a role or interpreting its contents:

```python
from pathlib import Path

from mammoth.core import inspect_artifact, verify_artifact

receipt = inspect_artifact(Path("checkpoints/latest.pt"))
# Store receipt.path, receipt.size_bytes, and receipt.sha256 in your own schema.
verify_artifact(receipt)
```

Inspection uses bounded chunks through one open file descriptor and rejects a
changing path or file. Final-component symlinks, directories, and special files
are rejected; missing paths raise `FileNotFoundError`. A receipt identifies only
the bytes observed during inspection, so verify it again immediately before an
operation that depends on those bytes.

When parsing must consume those bytes, use a descriptor-bound read session
instead of reopening the path yourself:

```python
from pathlib import Path

from mammoth.core import open_artifact_session

with open_artifact_session(Path("metadata.json")) as session:
    receipt = session.receipt
    with session.open_reader() as reader:
        raw_metadata = reader.read()
        # Parse raw_metadata with project-owned logic.

# Use parsed results only after the outer context exits successfully.
# receipt is the exact-byte identity for the parsed artifact.
```

Each nested reader starts at offset zero and supports normal binary `read`,
`seek` (including rewind), and `tell` operations. Readers are serial: do not nest or share them
between threads. Mammoth keeps the descriptor and transient filesystem state
private, rejects a final-component symlink but leaves ancestor-symlink and
containment policy to the caller, and verifies the visible path and exact bytes
again on successful session exit. A session detects changes at these boundaries;
it is not an immutable snapshot for a parser that reads while an external writer
changes the file. A reader is closed when its nested context ends; a completed
session keeps its receipt but rejects new readers with `RuntimeError`.

## Control workflow ordering and execution metadata

`run-major` is the default: Mammoth dispatches runs in declaration order and
each run's steps in declaration order. `step-major` interleaves runs by an
explicit canonical phase order:

```python
workflow = Workflow(
    root=Path("runs"),
    order="step-major",
    step_order=("train", "validate"),
    runs=(
        Run("a", (Step("train", ("train-a",)), Step("validate", ("validate-a",)))),
        Run("b", (Step("train", ("train-b",)), Step("validate", ("validate-b",)))),
    ),
)
```

Each run's declaration must be a duplicate-free subsequence of `step_order`;
a run may omit a canonical phase. `Run.root` may override `Workflow.root` for
mixed-root plans. A `Step` accepts only the final `command` argv plus optional
`cwd`, `timeout_seconds`, and `environment`; projects construct `torchrun` argv
themselves when needed.

Use `Execution` for generic immutable metadata. Mammoth derives intended phases
from step names, records the workflow invocation from `sys.argv`, and resolves
the previous execution while holding the logical-run lease. A run-local
`resolve_execution(context)` hook may replace its baseline `Execution` after
layout preparation and lease acquisition but before metadata publication. A
`before_first_step(context)` hook runs once after `execution_started` and before
the first `phase_started` record.

Every child receives only the four canonical join variables
`MAMMOTH_EXECUTION_ID`, `MAMMOTH_RUN_NAME`, `MAMMOTH_INVOCATION_KIND`, and
`MAMMOTH_PHASE`; static run and step environments cannot override Mammoth-owned
names. On the first timeout, signal, nonzero exit, launch failure, or hook
failure, dispatch stops. `WorkflowResult` retains structured run and step
outcomes and exposes the final process-compatible `exit_code`. Runs never
started after an ordinary failure remain blocked in memory without artifacts.

## Capture a supervised command

For callers that need a child's text output without reimplementing pipe
draining or process-tree cleanup, use the opt-in one-call API. Commands remain
tokenized argument arrays: Mammoth does not invoke a shell.

```python
from mammoth.workflow import run_captured_process

result = run_captured_process(
    ("python", "-c", "print('hello')"),
    cwd=None,
    environment={},
    timeout_seconds=30,
)
print(result.stdout)
```

`stdout` and `stderr` are captured separately. A non-zero return code remains
a returned process fact with its output intact. On timeout, Mammoth terminates
and reaps the observable process tree, then returns the partial output along
with `timed_out`; callers retain all retry, logging, artifact, and status
policy.

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
to their chosen Python logger. Dispatch gives each sink a bounded worker,
coalesces only JSONL's unprocessed non-final progress,
and retains TensorBoard scalar history. It accepts scalar CPU observations only
and currently rejects media until Mammoth defines an immutable CPU snapshot
contract. `max_pending_observations` bounds each sink's active queue and also
the number of retained JSONL progress scopes.

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

## Profile dependent named regions

For a compound workflow, callers can retain values between measured regions
while Mammoth aggregates only the samples they elect to report:

```python
from mammoth.torch import NamedPhaseProfiler

phases = NamedPhaseProfiler("cuda:0")
features = phases.measure("encode", lambda: model.encoder(images), record=False)
scores = phases.measure("decode", lambda: model.decoder(features))
phase_summaries = phases.summaries()
```

The phase names and their ordering remain caller-owned. `NamedPhaseProfiler`
records generic `torch.profiler` ranges when used inside a caller-owned profiler
context; `normalize_operation_profiles()` turns that context's operation rows
into Mammoth's stable value objects.

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

For the default batch mover, `TrainerConfig.cuda_prefetch` defaults to `True`.
On CUDA it uses one dedicated copy stream to transfer one following batch while
the current batch runs on the compute stream. This path activates only when all
CPU tensor leaves of that batch are pinned; pageable or unsupported batches,
CPU devices, and caller-supplied `batch_mover` functions retain their existing
synchronous movement. Set `cuda_prefetch=False` to disable the pipeline
deterministically. Mammoth does not change DataLoader pin-memory, worker,
sampling, or collation policy.

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

## Direct execution sessions

CPU-only and non-Torch programs can use the framework-neutral direct session
without installing or importing `torch`. `ExecutionSession.create()` owns a
single-process execution, its rank-zero logs, the logical-run lease, lifecycle
events, and generic observer or background-pipeline cleanup:

```python
from pathlib import Path

from mammoth.execution import ExecutionSession, ExecutionSpec

with ExecutionSession.create(
    ExecutionSpec(
        run_dir=Path("runs/report"),
        run_name="report",
        invocation_kind="report",
        intended_phases=("render",),
    )
) as session:
    with session.phase_scope("render"):
        render_report(session.observer)
```

Workflow children use `ExecutionSession.attach(expected)` instead. It requires
the four canonical `MAMMOTH_*` join variables and exactly validates the run,
invocation, phase, single-process topology, config reference, runtime metadata,
and resume facts without claiming the producer lease. A direct session records
phase success, failure, interruption, or skip and one terminal process outcome.

### PyTorch runtime integration

For a standalone single-process or `torchrun` invocation, initialize the Torch
runtime before the trainer. Every rank calls strict `create_execution()` and
then opens its rank-local logging stream; rank zero alone publishes metadata
and retains the logical-run lease. A workflow child instead calls strict
`attach_execution()` with the expected workflow metadata before constructing
its model or data loaders. Creation rejects an inherited canonical execution
ID, while attachment requires all four canonical join variables and never
claims the producer lease.

When resuming, the caller must resolve artifact provenance before constructing
the `ExecutionSpec` and provide the checkpoint's canonical lowercase SHA-256
and starting epoch. `parent_execution_id` remains optional for externally
supplied artifacts without trusted producer provenance, as does
`starting_global_step` when it is unknown. `resume_checkpoint` is only a
sanitized artifact reference: it never causes Mammoth to inspect the checkpoint
or infer a parent from a path, filename, phase, or timestamp. Attachment exactly
compares the run, invocation, phase membership, topology, config reference,
runtime metadata, and all five nullable resume facts before logging or project
work begins.

```python
from pathlib import Path

from mammoth.execution import ExecutionSpec
from mammoth.torch import (
    RuntimeConfig,
    Trainer,
    TrainerConfig,
    initialize_runtime,
)

runtime_config = RuntimeConfig(strategy="ddp", device="auto")
with initialize_runtime(runtime_config) as runtime:
    runtime.create_execution(
        ExecutionSpec(
            run_dir=Path("runs/example"),
            run_name="example",
            invocation_kind="train",
            intended_phases=("train",),
            resume_checkpoint=Path("runs/example/checkpoints/latest.pt"),
            resume_checkpoint_sha256="a" * 64,
            parent_execution_id="producer-attempt",
            starting_epoch=4,
            starting_global_step=1200,
        )
    )
    runtime.start_execution_logging()
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
lifecycle. `mammoth.torch.ExecutionSpec` remains a compatibility re-export;
its `ExecutionSession` compatibility adapter composes the neutral lifecycle and
retains only Torch `Trainer` ownership. Device validation, concrete workload
policy, DDP collectives, and checkpoint semantics stay in the Torch layer.

Projects with custom checkpoint formats can pass a `TrainerCheckpointPolicy`
that captures resumable and best-model writers, plus a
`CheckpointSavePolicy(mode="latest", save_best=True, every_epochs=1)`.
Mammoth then selects publication after validation, names standard checkpoint
files, and applies retention while the project retains payload serialization.
After a plan commits, `Callback.on_checkpoint_published()` receives a
`CheckpointPublication` containing one `PublishedCheckpoint` per artifact with
its path, role, epoch, byte size, and SHA-256. The hash and size describe the
completed temporary artifact immediately before atomic publication, so projects
can update manifests without reopening the checkpoint.

After a restart, `discover_resumable_checkpoints(checkpoint_root)` returns
immutable filename-derived `ResumableCheckpointCandidate` values for direct
children named `epoch_<N>.pt` or `latest_epoch_<N>.pt`. Candidates are ordered
by descending epoch, with `latest` before `epoch` on an exact tie. Discovery
does not open, hash, or validate entries; projects still own regular-file and
symlink policy, payload compatibility, warnings, and final resume selection.

```python
from pathlib import Path

from mammoth.torch import discover_resumable_checkpoints

checkpoint_root = Path("runs/example/checkpoints")
for candidate in discover_resumable_checkpoints(checkpoint_root):
    # Project code validates candidate.path and its payload before restoring it.
    print(candidate.path, candidate.role, candidate.epoch)
```

Framework-independent callers can overlap ordered CPU or I/O work with their
main workload through `BoundedBackgroundPipeline`. Submissions retain their
typed input association, and the pending bound applies backpressure before more
work is accepted:

```python
from mammoth.core import BoundedBackgroundPipeline

with BoundedBackgroundPipeline(render_artifact, max_pending=1) as pipeline:
    first = pipeline.submit(first_artifact)
    pipeline.submit(second_artifact)
    completed = pipeline.flush()
    assert first.result() == completed[0].result
    for item in completed:
        pipeline.acknowledge(item.submission)

```

Results and attributed failures remain pipeline-owned until
`acknowledge(submission)` is called, so interruption cannot erase an accepted
outcome between internal state changes and caller bookkeeping. If interruption
lands after queue acceptance, `submit()` completes the ownership transfer;
call `take_deferred_interrupt()` after submission and propagate the returned
`KeyboardInterrupt` or `SystemExit` once the caller has recorded that handoff.
Callers resolving an interrupt at the return boundary may use
`owns_input(value)` to test ownership by object identity. An interrupt after
acknowledgment removal is likewise deferred so the observed outcome is not
lost.

An active `mammoth.execution.ExecutionSession` can create and own the same
pipeline through `session.create_background_pipeline(...)`, ensuring accepted
work closes before observers, logging, and its finalizers. The Torch session
adapter exposes the same factory while closing owned trainers first.

For a framework-neutral result spanning several local files or directories,
build a core transaction from stable consumer specifications, stage every
payload, then publish it. Planning safely creates only missing target-parent
directories; callers still render payloads and define their semantic
validators. Use a namespace that identifies the logical publication, including
its create or replacement intent.

```python
from pathlib import Path

from mammoth.core import (
    TransactionArtifactSpec,
    build_artifact_transaction_plan,
    move_directory_into_transaction_stage,
    publish_artifact_transaction,
    stage_transaction_file,
)

plan = build_artifact_transaction_plan(
    namespace="reports-create",
    artifacts=(
        TransactionArtifactSpec("report", Path("reports/current.json"), "file"),
        TransactionArtifactSpec("payload", Path("reports/current-data"), "directory"),
    ),
    replace=False,
)
move_directory_into_transaction_stage(plan, "payload", rendered_directory)
stage_transaction_file(plan, "report", rendered_json_bytes)
publish_artifact_transaction(plan)
```

The lower-level `TransactionArtifact` and `ArtifactTransactionPlan` APIs
remain available when a caller already owns a complete plan. Recovery always
requires the expected plan explicitly so Mammoth can bind an interrupted
journal to the caller's current topology and validators.

The lower-level bounded publisher remains available outside the trainer. A
plan stages every artifact before committing its destinations in the declared
order:

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

At this lower level, serialization callbacks and retention choices remain
project-owned. Ordered
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
