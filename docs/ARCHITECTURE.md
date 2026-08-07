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

Framework-neutral callers may submit typed inputs to one
`BoundedBackgroundPipeline`. Its single worker preserves acceptance order,
applies a caller-selected bound across queued and active work, and associates
each result or failure with the accepted submission. Accepted work is never
cancelled. Results and failures remain owned until explicit acknowledgment;
interruption after queue acceptance is deferred for the submitter to
propagate after recording the handoff. Exact input-identity ownership checks
cover the caller return boundary, and interruption after acknowledgment removal
is deferred. Later flushes or closes therefore retain unacknowledged outcomes,
while cleanup remains idempotent and does not replace an active workload
exception. Optional done callbacks run on a pipeline-owned
dispatcher only after the submission state is final, so reentrant waits cannot
block the sole work thread and callback failures cannot replace worker outcomes.

`AsyncCheckpointPublisher` is a compatibility adapter over this pipeline. It
retains checkpoint-specific snapshot, plan, receipt, and `Future` APIs while
delegating ordered execution, pending bounds, failure ownership, and shutdown
to the framework-neutral lifecycle. Its generic mapping APIs default to an
`auto` capture policy: a conservative pre-allocation CUDA-memory check may
capture one immutable GPU snapshot and transfer it in the worker, otherwise
they retain the synchronous CPU snapshot path. The `cpu` policy always takes
the compatibility path. Project-owned ordered publication plans continue to
require caller-owned immutable captures.

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

Execution metadata may contain an optional sanitized `runtime` object. It
records allowlisted framework facts such as strategy, backend, and device type;
it never captures a complete process environment. Omitting this extension keeps
historical schema-version-1 records unchanged.

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
and compute every value. An observation may select its dense-history logical
step explicitly; otherwise the sink infers it from generic coordinates and then
falls back to its local sequence. The primary process writes by default; other
ranks use a no-op sink unless explicitly configured otherwise.

### Text logs

Per-producer text logs contain diagnostics and tracebacks. Monitoring never
parses them for state; it consumes immutable metadata and JSONL events. Each
process holds an exclusive append lease for its rank log until logging closes.
`ExecutionLogging` composes that handler with the rank's JSONL sink and any
caller-selected sinks, including rank-aware TensorBoard. Consumers that own a
different text presentation use `ExecutionObservability` to obtain the same
JSONL-first observer lifecycle without claiming the rank log; the structured
bundle remains independently composable with Mammoth's text handler.

`RunObserver` owns optional periodic-heartbeat scopes above its sink fan-out.
Heartbeat scheduling stays process-local, observes the configured idle interval,
and retains sink failure isolation. Projects continue to choose phase, task, and
message policy.

## Monitoring

The generic monitor reconstructs executions, producer liveness, task trees,
progress, throughput, arbitrary metrics, failures, and lineage. Generic ETA
uses only completed work, total work, and observed time. A run-level monitor
keeps every valid immutable execution available for navigation while selected
resume-lineage histories provide continuous project-neutral metric state.

An interactive terminal launches the optional Textual dashboard by default.
Textual owns refresh workers, keyboard navigation, scrolling, and responsive
layout; Rich renderables provide progress bars, execution panels, task tables,
and terminal-width-aware training charts. Redirected output remains one stable
ANSI-free snapshot. Conventional loss and learning-rate names affect
interactive presentation only and do not alter folding or assign scientific
meaning. `docs/MONITOR.md` owns the detailed reconstruction and presentation
contract.

The optional PyTorch integration may provide a training view for Mammoth's own
trainer. Project-specific pipeline projections are supplied as data or plugins,
not embedded in the monitor.

Viewer-host telemetry and execution-host telemetry are distinct. Local resource
sampling must never be presented as historical or remote execution provenance.
The Textual dashboard samples optional viewer CPU, memory, and GPU state,
labels it as viewer-host data, and isolates sampling failures. The monitor
remains passive: neither rendering nor telemetry writes artifacts, contacts
producers, or controls an execution.

## PyTorch execution runtime

`TorchExecutionRuntime` owns framework-level single-process or standard DDP
state. It resolves rank, local rank, world size, and device; initializes an
uninitialized default process group; exposes common object and tensor
collectives; validates optional caller-selected launch constraints; applies
caller-supplied rank weights to generic count and index partitions; and destroys
only a process group that it created. Execution establishment is available
separately from rank-logging startup so projects can validate their own lineage
and attach presentation sinks without recreating the runtime. The runtime does
not encode GPU models, concrete workload weights, or project topology rules.

Rank zero creates a direct execution and holds its logical-run lease, or joins
an execution identified by `MAMMOTH_EXECUTION_ID`. A consuming project may
explicitly pass temporary compatibility alias names through
`TorchExecutionRequest`; Mammoth validates those names and their values with
the canonical variable, but does not define, publish, or persist any
consumer-specific alias. Every rank validates the same immutable context, opens
its own JSONL and text streams, and reaches startup consensus. A failure on any
rank is reported coherently before project work begins. TensorBoard's rank-aware
sink and trainer checkpoints default to rank zero.

`TorchExecutionSession` owns process and phase lifecycle events after logging
starts. It is also the context-managed owner of observers, background
pipelines, and trainers created through its factories. Factory inputs such as
models, optimizers, schedulers, loaders, policies, serializers, metrics, and
directly supplied observers remain borrowed. Owned trainers close before owned
background pipelines, which close before owned observers regardless of
construction order. This flushes artifact work before metric sinks. Mammoth then
closes execution logging, releases leases, and destroys only a process group
created by the runtime. Projects may attach presentation cleanup through the
session close hook. Cleanup is idempotent, and cleanup failures are attached to
an active workload exception instead of replacing it.

A workflow execution is owned by its single runner and may launch steps with
different process counts. Joined workflow children therefore validate run and
phase identity while their process streams record the child runtime's actual
world size; direct executions retain strict metadata/runtime topology matching.

## Generic PyTorch trainer

The trainer accepts constructed objects:

- `torch.nn.Module`;
- optimizer and optional scheduler;
- training and optional validation `DataLoader` objects; and
- project functions that interpret batches and return scalar loss/metrics.

The project may also provide an accumulation policy, scalar reduction specs,
additive stateful metrics, metric sink routes, and a checkpoint policy. These
contracts describe mechanics only: names and update values remain opaque to
Mammoth, while project code owns every metric calculation and checkpoint
serializer.

Mammoth may own the ordinary loop mechanics: mode switching, device transfer,
precision, backward, accumulation, clipping, optimizer/scheduler steps,
standard DDP, callbacks, logging, interruption handling, and registered-state
checkpoint publication. A trainer may consume a `TorchExecutionRuntime` for
device/rank identity and its active execution observer; constructing the
trainer without a runtime remains supported for callers that already own their
process group.

For DDP, Mammoth suppresses reducer communication only for non-final
microbatches in an accumulation window. Every rank reaches a Python-status
consensus before its final backward, then lets PyTorch's native DDP reducer
average gradient buckets during that backward. This preserves one reduction
boundary per logical optimizer step without trainer-owned CUDA scalar
materialization or per-parameter reductions. A rank-local error before that
final backward is reported coherently; a failure during native backward follows
the DDP launcher's fail-fast behavior because another rank may already be in a
reducer collective. The trainer requires standard native-reducer invariants;
projects with dynamic gradient participation must configure or own a compatible
DDP training loop.

`WarmupLinearLR` provides the reusable BERT-style zero-to-base warmup followed
by linear decay to zero. Projects choose its warmup ratio and optimizer-step
horizon. Checkpoint restore preserves the saved cursor, accepts an unchanged or
extended active horizon, rebases every optimizer parameter group onto an
extended curve, and rejects horizon shrinkage.

The caller-supplied module remains the canonical `base_model`. Mammoth derives
an `execution_model` by applying device placement, then optional DDP wrapping,
then optional caller-configured `torch.compile`. Step functions receive the
execution model. Mammoth retains the underlying DDP wrapper privately so
native gradient accumulation can use `no_sync()` only before its final
microbatch even when the execution path is compiled. Projects keep
architecture-specific methods on the canonical model when those methods are
not part of the ordinary forward contract.

Mammoth's validation-metric early-stopping callback owns the generic best-value
and consecutive-failure state machine and stops on the `patience`-th failed
check. `TrainerState.stopped_early` is the sole persisted terminal decision;
`fit()` returns without touching loaders, callbacks, or optimizer state when a
restored state is already terminal. The trainer reads the callback's transient
`improved` signal after validation to select best-model publication, while the
project continues to choose the metric, mode, patience, minimum delta, and
serializer.

`StepOutput` carries the optional loss, already-computed scalar metrics, and
opaque updates for registered stateful metrics. Mammoth reduces configured
distributed training-window summaries and train/validation epoch summaries,
then applies separate batch and epoch routes. Validation batch routes and
metrics configured with `distributed=False` remain rank-local. The trainer
emits generic phase, task, progress, heartbeat, completion, and failure
observations; projects select phase names, metric names, and display fields.
When a surrounding command already owns the outer training phase, it disables
the trainer's fit-level phase records while retaining Mammoth's nested task,
progress, validation-phase, heartbeat, and metric observations.

An accumulation policy receives rank identity and the local loader length, then
returns the local microbatch count and loss scale for each shared optimizer
window. Every rank must produce the same number of optimizer windows. Explicit
per-window scales cover unequal partial windows without assigning workload
meaning to Mammoth. Native DDP reaches failure consensus before each final
backward and lets the reducer average that window's gradient buckets. The
persisted global-step cursor counts all ranks' microbatches at completed
windows; step callbacks receive a deterministic rank-ordered position inside
the active window.
Consumers may supply post-optimizer metric providers for values, such as the
current scheduler rate, that only become authoritative after Mammoth completes
the optimizer and scheduler boundary.
The completed optimizer cursor remains checkpoint state, while consumers may
select a completed-step or zero-based logical clock for routed sink history.

The generic weighted policy converts arbitrary positive caller-supplied rank
weights into integer local microbatch counts and the DDP loss scale for one
global window. The matching batch sampler partitions opaque dataset indices,
drops incomplete global windows, and exposes `set_epoch()` for deterministic
reshuffling. Contiguous weighted index ranges support validation or inference
partitioning without constructing or interpreting a dataset. Consuming
projects retain concrete weights, device eligibility, DataLoader construction,
and every model and dataset policy.

For coarse independently executable work, the weighted task allocator accepts
only opaque string IDs, nonnegative numeric costs, and positive rank weights.
It considers tasks largest-cost-first, breaks equal-cost ties by ID, minimizes
each rank's projected cost divided by its weight, and breaks rank ties by lower
rank. Projects retain task discovery, cost estimation, skip/resume policy,
capacity checks, execution, artifacts, and failure reporting.

Complex algorithms with several optimizers, alternating updates, reinforcement
learning control flow, or custom collectives keep their loop in the consuming
project and use Mammoth only for orchestration, logging, monitoring, and
artifact mechanics.

## Checkpoint publication

Mammoth's optional PyTorch layer provides two checkpoint paths. The generic
trainer may use Mammoth's versioned registered-state payload and restore it
directly. Projects with established checkpoint schemas instead capture one
immutable snapshot and provide resumable and best-model serializers.

A trainer checkpoint policy captures project state when Mammoth's
`CheckpointSavePolicy` selects resumable or best-model publication and
translates its own format into Mammoth's typed restore contract. The trainer
first calls the policy once on rank zero through `inspect_checkpoint()`, then
shares its `CheckpointInspection`, including the immutable `RestoreOptions`
selection, with every rank. `load_checkpoint()` restores the project-owned
model representation and lets Mammoth restore or reset generic optimizer,
scheduler, callback, and terminal early-stop state while restoring trainer coordinates. Its
returned `TrainerCheckpointRestore` reports restored and reset components plus
opaque project metadata. Mammoth infers absent cursors from the active
accumulation plan and requires coordinates, terminal state, component reports,
and metadata to agree across DDP ranks before training resumes. Projects may
also request a synchronized manual publication. Rank zero publishes an
interruption snapshot without entering collectives that could deadlock when a
signal reaches only one process. The checkpoint context identifies the reason
and exposes the prior restore report to the project serializer.

`Trainer.fit()` publishes that interruption plan automatically when
`KeyboardInterrupt` escapes the loop, then preserves and re-raises the original
interruption. Callers may disable this behavior explicitly when an outer system
owns interruption persistence. Under DDP, failure consensus preserves an
interrupt as `KeyboardInterrupt` on every rank instead of converting it to an
ordinary failure; rank zero can therefore publish while peers perform only
local checkpoint shutdown.

The trainer applies publisher backpressure before asking a project checkpoint
policy to capture state. `CheckpointSavePolicy` selects `all` or `latest`
resumable retention, periodic cadence, and optional best-model publication.
Mammoth names zero-based `epoch_<N>.pt`, `latest_epoch_<N>.pt`, and
`best.safetensors`; best publication follows `EarlyStopping.improved` on every
validation epoch and is independent of resumable cadence. Manual and
interrupted saves publish resumable state only. The publisher receives
caller-owned immutable state before background work, bounds pending
publications, prepares and syncs every artifact before the first commit,
replaces best before the resumable commit marker, and retires old latest files
only after every commit succeeds. Before each atomic rename, Mammoth hashes the
completed temporary artifact and records its exact size. Successful plans yield
typed `PublishedCheckpoint` receipts with path, role, epoch, size, and SHA-256;
the trainer retains their futures and delivers receipts through observers and
callbacks during checkpoint flush. Paths are confined to the declared
checkpoint root. Projects retain serializers, payload fields, compatibility
rules, receipt consumers, and restore policy. Resume discovery remains a
filesystem concern; publication does not maintain a persistent checkpoint
catalog.

The lower-level ordered-plan API remains available for non-trainer artifact
publication where callers need custom names and retirement targets.

Atomic replacement applies to each file independently. An interruption between
ordered replacements may expose a prefix of the plan, so the caller places its
commit-marker artifact last. True multi-file transactions require a different
generation-directory and atomic-index layout and are not implied by this API.

## Generic PyTorch backend configuration

The optional PyTorch layer applies caller-selected float32 matmul precision,
CUDA matmul and cuDNN TF32 policy, cuDNN benchmarking, cuDNN determinism, and
deterministic-algorithm mode as process-global backend configuration. The same
API captures effective state and provides reversible overrides. A separate seed
policy selects Python, Torch CPU, and available Torch CUDA generators without
assigning a concrete seed or dataset-worker policy. Consuming projects retain
all chosen values, environment variables, logging filters, and DataLoader seed
construction.

Compile backends use `inductor` only when the caller leaves the backend unset;
an explicitly empty backend is invalid. Deterministic warning-only behavior is
likewise valid only when the caller explicitly selects deterministic-algorithm
mode, so an omitted mode cannot silently inherit a warning policy.

When a caller supplies both the newer matmul-precision control and the legacy
CUDA TF32 boolean, Mammoth expresses the legacy boolean's precedence through
the new API so PyTorch never enters an unreadable mixed-API state.

## Generic PyTorch profiling

The optional PyTorch layer profiles arbitrary caller-owned zero-argument
callables. The caller constructs the model and inputs, captures positional or
keyword arguments, selects inference, autocast, or autograd contexts, and owns
the semantic interpretation of every output. Mammoth owns synchronized cold
and steady-state timing, caller-labelled throughput, CUDA allocator evidence,
explicit caller-selected component ranges, normalized operation summaries,
optional traces, reversible Torch runtime settings, and versioned report
publication. Shape recording groups operation rows by input shape and delegates
its temporary runtime settings to the generic backend configuration API.

The default output summary records only project-neutral tensor and container
metadata. Callers provide semantic summarizers for predicted classes, masks,
tokens, numerical tolerances, or scientific metrics. Mammoth does not discover
architecture components, generate inputs, load checkpoints, select compile
scopes, or define what one work unit means. Instrumented profiler time remains
separate from the uninstrumented latency distribution.

## Compatibility policy

Artifact readers preserve compatibility with schema-version-1 execution and
event records from the originating project. Package and environment names may
change, but historical artifact paths and wire records remain readable. A
schema version changes only for a wire-contract change, not for a module
rename.
