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

Status: complete.

Build monitoring above immutable metadata and event readers:

- discovery and exact execution selection;
- lineage-aware attempt history;
- producer/rank and task state folding;
- generic progress, throughput, metric trends, and ETA;
- warning isolation for malformed or missing streams;
- stable ANSI-free snapshots;
- default-on-TTY Textual interface with responsive Rich panels, execution
  navigation, progress bars, ETA, and arbitrary metric trends;
- optional psutil/vendor telemetry with explicit viewer-host labeling.

Exit condition: the monitor can read both Mammoth-generated runs and existing
compatible runs without importing project code.

## Phase 5: Programmatic workflow runner

Status: complete.

Generalize the existing experiment-runner ideas into arbitrary Python-defined
serial steps:

- immutable `Workflow`, `Run`, `Step`, and `Execution` inputs;
- side-effect-free planning with run-major and canonical step-major order;
- final opaque argv supplied by callers, including caller-built `torchrun`;
- workflow-root defaults with per-run root overrides;
- post-lease execution resolution and one before-first-step boundary;
- canonical child join variables without aliases or provider frameworks;
- signal, timeout, descendant, and process-group supervision;
- first-failure blocking without artifacts for unstarted runs; and
- runner-owned lifecycle events and structured `WorkflowResult` outcomes.

Phase names and command arguments remain entirely project-defined.

Exit condition: two unrelated projects can construct the same programmatic
workflow surface and monitor its artifacts without Mammoth-specific code in
their commands beyond strict execution attachment through the canonical join
environment.

## Phase 6: Generic PyTorch trainer

Status: complete.

Add the optional `mammoth.torch` layer with a deliberately bounded first API:

- one constructed `nn.Module`;
- one optimizer and optional scheduler;
- reusable warmup-linear scheduling with extended-horizon resume;
- constructed train and optional validation `DataLoader` objects;
- project-supplied train and validation step functions;
- recursive batch-to-device transfer with an override;
- fp32, bf16, and fp16 precision policies;
- gradient accumulation and clipping;
- standard single-process and DDP strategies;
- context-managed reverse-order ownership for session-created observers and trainers;
- scalar metric aggregation policies;
- callback-based validation and early stopping;
- typed two-phase checkpoint inspection and selective generic restore/reset;
- registered `state_dict` checkpoint state;
- bounded asynchronous atomic publication; and
- direct integration with `RunObserver`.

Exit condition: small classification and non-classification examples can use
the same trainer by changing only their model, loaders, and step functions.

## Phase 7: First-project migration

Status: complete.

Adopt Mammoth incrementally in the originating project:

1. Read existing artifacts through compatibility readers.
2. Replace execution metadata and JSONL writers behind compatibility imports.
3. Replace the monitor with the generic monitor plus project view definitions.
4. Move workflow process supervision to Mammoth while keeping project command
   construction local.
5. Adopt the generic trainer directly with project-owned step and checkpoint
   policies.
6. Delegate training-resource cleanup to the existing execution session.
7. Retain project-owned loops only for algorithms outside the trainer contract.

No artifact migration or destructive rewrite should be required.

The originating project now uses compatibility facades for execution metadata
and JSONL streams, defaults to the generic monitor while retaining an explicit
project view, and delegates subprocess supervision without moving command
construction or scheduling. Its ordinary segmentation training now constructs
Mammoth's trainer directly; the project retains architecture steps, semantic
metrics, manifest behavior, and checkpoint formats as injected policies.

## Phase 8: Unified PyTorch execution runtime

Status: complete.

Compose the framework-level pieces used by both generic and project-owned
training loops:

- optional sanitized runtime provenance in schema-version-1 metadata;
- single-process and standard DDP initialization and cleanup;
- generic rank, local-rank, world-size, backend, and device identity;
- common object and tensor collectives plus caller-weighted local partitions;
- rank-wide execution creation, joining, and startup consensus;
- exclusive rank text logs and JSONL observation streams;
- generic process and phase lifecycle completion and cleanup;
- primary-rank TensorBoard and checkpoint defaults; and
- optional runtime consumption by the generic trainer, including interruption
  checkpoint publication.

Exit condition: one direct invocation can use the same API in single-process or
two-rank DDP mode, and a rank-local startup failure reaches every participant
without moving project hardware policy or concrete workload weights into
Mammoth.

## Phase 9: Ordered checkpoint publication

Status: complete.

Extend the optional PyTorch checkpoint layer for projects whose checkpoint
meaning remains local but whose publication mechanics are reusable:

- caller-owned opaque serializers and payloads;
- trainer-owned all/latest retention, standard naming, and validation-driven
  best-model selection;
- preparation of every artifact before ordered atomic replacement;
- exact post-commit retirement confined to the checkpoint root;
- bounded asynchronous publication with explicit failure propagation; and
- exact-byte publication receipts delivered through trainer lifecycle hooks;
- persistent checkpoint catalogs deferred until receipt consumers require one;
- compatibility with the existing registered-state single-file API.

Exit condition: a consuming project can publish inference and resume artifacts
from one immutable snapshot while retaining its formats, compatibility rules,
commit order, and retention policy.

## Phase 10: Generic PyTorch callable profiling

Status: complete.

Add model-independent profiling to the optional PyTorch layer:

- arbitrary caller-owned zero-argument workloads;
- separate cold-start, warmup, steady-state, and instrumented passes;
- synchronized wall and CUDA timing with caller-labelled throughput;
- CUDA allocator peaks and normalized operation evidence;
- explicit caller-selected component ranges and optional Chrome traces;
- generic nested tensor output metadata with semantic summarizer overrides;
- reversible process-global Torch runtime controls; and
- immutable versioned reports with atomic JSON publication.

Exit condition: unrelated PyTorch models with different calls and output
containers can use the same profiler without Mammoth constructing or
interpreting either workload.

## Phase 11: Generic weighted distributed workload

Status: complete.

Generalize the integer scheduling mechanics needed by heterogeneous DDP while
leaving every concrete workload decision with the consuming project:

- deterministic weighted integer counts and contiguous index ranges;
- arbitrary positive rank weights and world sizes;
- deterministic weighted-cost allocation for opaque task IDs;
- a dataset-independent distributed batch sampler over opaque indices;
- full global accumulation windows with rank-local batch counts; and
- correct mean-loss scaling after standard DDP gradient averaging.

Exit condition: a consumer can supply its own rank weights and datasets while
Mammoth owns the matching sampler, partition, and accumulation mechanics.

## Phase 12: Generic Torch backend configuration

Status: complete.

Promote reusable process-global numerical controls out of profiler-specific
code:

- caller-selected TF32, float32 matmul precision, and cuDNN benchmarking;
- caller-selected cuDNN and Torch deterministic modes;
- effective backend-state capture and reversible overrides; and
- selective Python, Torch CPU, and available CUDA generator seeding.

Exit condition: training, inference, and profiling consumers share one backend
implementation while retaining their concrete values, logging policy, and data
seeding behavior.

## Phase 13: Named-phase PyTorch profiling

Status: complete.

Add project-neutral timing for dependent caller-owned regions:

- opaque non-empty caller-selected phase names and returned results;
- optional unreported warmup or diagnostic samples;
- immutable wall and optional CUDA-device latency summaries per phase;
- caller-visible profiler ranges without a prescribed phase vocabulary;
- public process-local CUDA synchronization and allocator operations; and
- public operation-row normalization for caller-owned profiler sessions.

Exit condition: a consuming project can measure dependent regions and preserve
its own workload construction, profiler lifecycle, phase semantics, and report
format without reimplementing generic PyTorch profiling mechanics.

## Phase 14: Framework-neutral direct execution sessions

Status: complete.

Extract direct single-process execution lifecycle composition above the core
and logging layers while retaining PyTorch runtime compatibility:

- public immutable execution specifications and context-managed direct sessions
  that do not import or require PyTorch;
- strict direct creation and workflow-child attachment with unchanged immutable
  metadata, canonical joins, artifact paths, sanitization, and lease rules;
- neutral process and phase terminal events, observer and background-pipeline
  ownership, heartbeat support, deterministic cleanup, and exception
  precedence;
- a Torch session adapter that composes the neutral session while retaining
  device, DDP, collective, Trainer, checkpoint, and process-group ownership;
  and
- compatibility re-export of `ExecutionSpec` through `mammoth.torch`.

Exit condition: CPU-only callers can establish and monitor a direct execution
without importing Torch, while single-process and DDP Torch workflows preserve
their existing lifecycle, consensus, cleanup, and public imports.

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
