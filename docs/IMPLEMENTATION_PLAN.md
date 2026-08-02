# Mammoth Implementation Plan

The phases below implement the contracts in `docs/ARCHITECTURE.md`. Each phase
ends with a usable vertical slice.

## Phase 0: Establish the independent project

Status: complete.

- Initialize a standalone uv library at `~/mammoth`.
- Use a `src/mammoth` package and Python 3.12 baseline.
- Record the architectural boundary and compatible run layout.
- Keep runtime dependencies empty until the core API requires them.

Exit condition: `mammoth` imports from its own uv environment and the project
contains no dependency on the originating repository.

## Phase 1: Core identities and artifact layout

Status: complete.

Implement the first `mammoth.core` vertical slice:

- `RunLayout` and safe run-name validation;
- execution ID generation and validation;
- immutable execution metadata;
- explicit previous/resume/parent lineage references;
- credential-safe command, URL, and metadata sanitization;
- logical-run producer leases;
- atomic JSON and opaque artifact publication helpers; and
- historical schema-v1 metadata loading.

Exit condition: two independent fake projects can create and join attempts in
different entry roots while producing the same operational layout.

## Phase 2: Append-only execution events

Status: complete.

Implement the framework-neutral event layer:

- lifecycle, task, progress, heartbeat, and terminal events;
- arbitrary validated coordinates and bounded display metrics;
- one exclusive JSONL stream per producer;
- throttled replaceable progress and periodic heartbeats;
- recursive secret sanitization;
- historical replay and incremental active tailing;
- precise malformed-record errors and append-only mutation checks; and
- compatibility reading for existing schema-v1 event streams.

Exit condition: an instrumented standalone command can be observed live using
only filesystem artifacts, and logging failure does not stop the command.

## Phase 3: Logging facade and TensorBoard

Status: complete.

Add a sink-oriented logging package:

- `RunObserver` as the producer-facing API;
- JSONL event sink;
- process/rank text-log ownership;
- TensorBoard scalar and media sink;
- rank-aware primary/no-op behavior;
- metric routing without metric calculation;
- deterministic flushing and shutdown; and
- optional dependencies that do not affect `mammoth.core` imports.

Exit condition: one progress call can update live JSONL state and selected
TensorBoard series while preserving their distinct retention semantics.

## Phase 4: Passive monitor

Build monitoring above immutable metadata and event readers:

- discovery and exact execution selection;
- lineage-aware attempt history;
- producer/rank and task state folding;
- generic progress, throughput, metric trends, and ETA;
- warning isolation for malformed or missing streams;
- stable ANSI-free snapshots;
- optional Rich/Textual interactive interface; and
- optional psutil/vendor telemetry with explicit viewer-host labeling.

Preserve the command shape:

```bash
mammoth monitor <run-name> --entry <entry>
mammoth monitor <run-name> --entry <entry> --execution <execution-id>
```

Exit condition: the monitor can read both Mammoth-generated runs and existing
compatible runs without importing project code.

## Phase 5: Declarative workflow runner

Generalize the existing experiment-runner ideas into arbitrary steps:

- YAML workflow parsing with defaults and strict validation;
- safe run and step selection;
- ordered dependencies, initially a directed acyclic graph;
- local subprocess launcher;
- optional `torchrun` launcher;
- environment inheritance and secret-safe provenance;
- dry runs that create no execution artifacts;
- signal, timeout, descendant, and process-group supervision;
- stop, skip, and failure policies; and
- runner-owned lifecycle events.

Phase names and command arguments remain entirely project-defined.

Exit condition: two unrelated command-line projects can use the same workflow
schema and monitor without Mammoth-specific code in their commands beyond the
execution join environment.

## Phase 6: Generic PyTorch trainer

Add the optional `mammoth.torch` layer with a deliberately bounded first API:

- one constructed `nn.Module`;
- one optimizer and optional scheduler;
- constructed train and optional validation `DataLoader` objects;
- project-supplied train and validation step functions;
- recursive batch-to-device transfer with an override;
- fp32, bf16, and fp16 precision policies;
- gradient accumulation and clipping;
- standard single-process and DDP strategies;
- scalar metric aggregation policies;
- callback-based validation and early stopping;
- registered `state_dict` checkpoint state;
- bounded asynchronous atomic publication; and
- direct integration with `RunObserver`.

Exit condition: small classification and non-classification examples can use
the same trainer by changing only their model, loaders, and step functions.

## Phase 7: First-project migration

Adopt Mammoth incrementally in the originating project:

1. Read existing artifacts through compatibility readers.
2. Replace execution metadata and JSONL writers behind compatibility imports.
3. Replace the monitor with the generic monitor plus project view definitions.
4. Move workflow process supervision to Mammoth while keeping project command
   construction local.
5. Adopt the generic trainer only where its bounded contract matches exactly.
6. Retain project-owned loops for specialized behavior.

No artifact migration or destructive rewrite should be required.

## Deferred capabilities

Defer these until another real project supplies requirements:

- FSDP, pipeline parallelism, or remote distributed coordination;
- Slurm, Kubernetes, and cloud launchers;
- object-store event and artifact backends;
- several optimizers or automatic GAN/RL control flow;
- generic inference serving; and
- a web monitoring service.

The extension protocols should permit these later without making the initial
implementation speculative.
