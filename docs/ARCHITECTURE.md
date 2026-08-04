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
collectives; and destroys only a process group that it created. Execution
establishment is available separately from the combined rank-logging startup
so compatibility adapters can retain a project-specific logging facade. It
does not encode GPU models, workload weights, samplers, or project topology
rules.

Rank zero creates a direct execution and holds its logical-run lease, or joins
an execution already identified by `MAMMOTH_EXECUTION_ID` or the compatible
`TISAM_EXECUTION_ID` hook. Every rank validates the same immutable context,
opens its own JSONL and text streams, and reaches startup consensus. A failure
on any rank is reported coherently before project work begins. TensorBoard's
rank-aware sink and trainer checkpoints default to rank zero.

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

Mammoth may own the ordinary loop mechanics: mode switching, device transfer,
precision, backward, accumulation, clipping, optimizer/scheduler steps,
standard DDP, callbacks, logging, interruption handling, and registered-state
checkpoint publication. A trainer may consume a `TorchExecutionRuntime` for
device/rank identity and its active execution observer; constructing the
trainer without a runtime remains supported for callers that already own their
process group.

Complex algorithms with several optimizers, alternating updates, reinforcement
learning control flow, or custom collectives keep their loop in the consuming
project and use Mammoth only for orchestration, logging, monitoring, and
artifact mechanics.

## Checkpoint publication

Mammoth's optional PyTorch layer provides two checkpoint paths. The generic
trainer may use Mammoth's versioned registered-state payload and restore it
directly. Projects with established checkpoint schemas instead submit ordered
publication plans containing opaque serializer callbacks and exact retirement
paths.

The publisher snapshots or receives caller-owned immutable state before
background work, bounds pending publications, prepares and syncs every artifact
before the first commit, replaces destinations in declared order, and retires
old artifacts only after every commit succeeds. Paths are confined to the
declared checkpoint root. Mammoth does not choose filenames, serializers,
payload fields, compatibility rules, best-model policy, or retention targets.

Atomic replacement applies to each file independently. An interruption between
ordered replacements may expose a prefix of the plan, so the caller places its
commit-marker artifact last. True multi-file transactions require a different
generation-directory and atomic-index layout and are not implied by this API.

## Generic PyTorch profiling

The optional PyTorch layer profiles arbitrary caller-owned zero-argument
callables. The caller constructs the model and inputs, captures positional or
keyword arguments, selects inference, autocast, or autograd contexts, and owns
the semantic interpretation of every output. Mammoth owns synchronized cold
and steady-state timing, caller-labelled throughput, CUDA allocator evidence,
explicit caller-selected component ranges, normalized operation summaries,
optional traces, reversible Torch runtime settings, and versioned report
publication. Shape recording groups operation rows by input shape. When a
caller supplies both the newer matmul-precision control and the legacy CUDA
TF32 boolean, Mammoth expresses the legacy boolean's precedence through the
new API so PyTorch never enters an unreadable mixed-API state.

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
