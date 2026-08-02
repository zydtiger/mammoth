# Mammoth Architecture

## Boundary

Mammoth owns operational infrastructure. A consuming project owns the meaning
of its computation.

This boundary is deliberately stricter than merely avoiding imports from one
originating repository. Mammoth must not encode assumptions about:

- model architectures or model factories;
- dataset classes, paths, transforms, labels, or sampling policies;
- batch shapes or model call signatures;
- losses or scientific metric calculations;
- project configuration models;
- checkpoint compatibility or inference export formats; or
- domain phases such as segmentation, plotting, or result refresh.

The optional PyTorch layer may understand generic training concepts such as an
epoch, optimizer step, precision policy, and gradient accumulation. It must
receive already constructed framework objects and project-supplied step
functions.

## Dependency layers

Dependencies point downward only:

```text
consuming project
    │
    ├── mammoth.torch       optional generic nn.Module/DataLoader training
    ├── mammoth.workflow    declarative steps and launchers
    ├── mammoth.monitor     replay, terminal UI, and telemetry
    ├── mammoth.logging     JSONL, text, and TensorBoard sinks
    └── mammoth.core        identities, layout, events, artifacts, provenance
```

`mammoth.core` should use the Python standard library. TensorBoard, Rich,
Textual, psutil, and PyTorch belong in optional dependency groups and must not
be imported by the core package.

## Runtime model

Mammoth uses six nested concepts:

1. **Entry**: an arbitrary artifact root such as `runs` or `experiments`.
2. **Run**: a stable logical identity beneath one entry.
3. **Execution**: one immutable attempt to produce or continue a run.
4. **Producer**: a runner or process/rank that exclusively owns one stream.
5. **Phase and task**: arbitrary project-named work scopes.
6. **Observation**: lifecycle, progress, heartbeat, metric, or terminal state.

Core event consumers treat phase names, task names, coordinates, metric names,
and artifact extensions as opaque validated data.

## Artifact layout

`RunLayout(entry, run_name)` resolves the stable run directory and its
operational paths. The default contract is:

```text
<entry>/<run-name>/
├── manifest.json
├── checkpoints/                 project-owned contents
├── logs/
│   ├── .logical-run.lock        Mammoth producer lease
│   ├── events.out.tfevents.*    Mammoth TensorBoard sink
│   └── executions/
│       └── <execution-id>/
│           ├── execution.json   immutable attempt identity
│           ├── runner.jsonl     optional workflow producer
│           ├── rank-N.jsonl     process/rank producer
│           └── rank-N.log       human-readable diagnostics
├── results/                     project-owned contents
└── vis/                         project-owned contents
```

The entry path is supplied by the caller. Mammoth does not assign semantic
meaning to entry names.

## Logging responsibilities

### JSONL

JSONL is the live operational source of truth. Each producer exclusively owns
one append-only file. Records contain stable identity and lifecycle fields plus
opaque project coordinates and metrics. Progress may be throttled and replaced;
lifecycle and terminal records flush immediately. A writer failure disables
only that writer and must not terminate the workload.

### TensorBoard

TensorBoard stores dense numerical and media history. Mammoth manages writer
ownership, logical steps, flushing, and shutdown. Projects choose metric names
and compute every value. The primary process writes by default; other ranks use
a no-op sink unless explicitly configured otherwise.

### Text logs

Per-producer text logs contain diagnostics and tracebacks. Monitoring never
parses them for state; it consumes immutable metadata and JSONL events.

## Monitoring

The generic monitor reconstructs executions, producer liveness, task trees,
progress, throughput, arbitrary metric trends, failures, and lineage. Generic
ETA uses only completed work, total work, and observed time.

The optional PyTorch integration may provide a training view for Mammoth's own
trainer. Project-specific pipeline projections are supplied as data or plugins,
not embedded in the monitor.

Viewer-host telemetry and execution-host telemetry are distinct. Local resource
sampling must never be presented as historical or remote execution provenance.

## Generic PyTorch trainer

The trainer accepts constructed objects:

- `torch.nn.Module`;
- optimizer and optional scheduler;
- training and optional validation `DataLoader` objects; and
- project functions that interpret batches and return scalar loss/metrics.

Mammoth may own the ordinary loop mechanics: mode switching, device transfer,
precision, backward, accumulation, clipping, optimizer/scheduler steps,
standard DDP, callbacks, logging, interruption handling, and registered-state
checkpoint publication.

Complex algorithms with several optimizers, alternating updates, reinforcement
learning control flow, or custom collectives keep their loop in the consuming
project and use Mammoth only for orchestration, logging, monitoring, and
artifact mechanics.

## Compatibility policy

Artifact readers preserve compatibility with schema-version-1 execution and
event records from the originating project. Package and environment names may
change, but historical artifact paths and wire records remain readable. A
schema version changes only for a wire-contract change, not for a module
rename.
