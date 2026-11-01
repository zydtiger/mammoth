# Mammoth Code Map

## Source Tree

```text
src/mammoth/
├── __init__.py
├── __main__.py
├── cli.py
├── compat.py
├── execution.py
├── core/
│   ├── _filesystem/
│   │   ├── __init__.py
│   │   ├── confined.py
│   │   ├── io.py
│   │   ├── leases.py
│   │   ├── locking.py
│   │   ├── publication.py
│   │   ├── text.py
│   │   └── windows.py
│   ├── __init__.py
│   ├── _event_stream.py
│   ├── artifacts.py
│   ├── events.py
│   ├── execution.py
│   ├── groups.py
│   ├── identity.py
│   ├── layout.py
│   ├── leases.py
│   ├── pipeline.py
│   ├── telemetry.py
│   ├── transactions.py
│   └── workstore.py
├── logging/
│   ├── __init__.py
│   ├── dispatch.py
│   ├── execution.py
│   ├── jsonl.py
│   ├── model.py
│   ├── observer.py
│   ├── tensorboard.py
│   └── text.py
├── monitor/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── documents.py
│   ├── fleet.py
│   ├── model.py
│   ├── psutil_telemetry.py
│   ├── render.py
│   ├── rich_ui.py
│   ├── telemetry.py
│   ├── textual_ui.py
│   ├── web.py
│   └── web_static/
├── queue/
│   ├── __init__.py
│   ├── api.py
│   ├── daemon.py
│   ├── executor.py
│   ├── model.py
│   ├── sampling.py
│   ├── scheduler.py
│   ├── service.py
│   └── store.py
├── workflow/
│   ├── __init__.py
│   ├── launch.py
│   └── runner.py
├── torch/
│   ├── __init__.py
│   ├── backend.py
│   ├── batch.py
│   ├── callbacks.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── runtime.py
│   ├── scheduling.py
│   ├── state.py
│   └── trainer.py
└── py.typed
```

`core/_event_stream.py` owns level-1 Zstandard frames, the standard final seek
table, the disposable live index, append-preserving reopen, and incremental
compressed byte reads. `core/events.py` and `core/groups.py` retain event schema
and sequence validation above that shared storage layer.

## Public Symbol Index

| Import path | Symbols |
| --- | --- |
| `mammoth` | `__version__` |
| `mammoth.core` | `RunLayout`, `ExecutionContext`, `ExecutionMetadata`, `LogicalRunLease`, framework-neutral `RetireableLeaseNamespace` claim/close/terminalize/reconciliation APIs and typed lease-namespace errors, unitless `ExecutionEvent` and `ExecutionEventWriter` APIs with schema-v1 legacy-unit reading, `ExecutionEventTailReader`, `ExecutionEventReadError`, `BoundedBackgroundPipeline`, typed background submissions/results, attributed background failures, `ArtifactReceipt`, `inspect_artifact`, `verify_artifact`, descriptor-bound `ArtifactReadSession` / `open_artifact_session` parsing, atomic and prepared artifact publication helpers, high-level `TransactionArtifactSpec` / `build_artifact_transaction_plan` plus exclusive file and directory staging helpers, and retained low-level `ArtifactTransactionPlan` / `TransactionArtifact` planning, sealing, shared target-key and transaction-local lease namespaces, local publication, coordinated journal recovery with an optional `missing_ok` no-journal outcome, a `transaction_journal_exists` convenience, and typed result APIs for several local file or directory targets across declared non-overlapping roots; explicit-only execution lifecycle functions, event readers, sanitizers, resume-checkpoint SHA-256 validation, `is_immutable_log_entry` consumer-facing immutable-log-entry classification, identity validators, `validate_group_id`, and `derive_run_name` stable path-derived run-name construction; canonical workflow-child contract constants `EXECUTION_ID_ENV`, `RUN_NAME_ENV`, `INVOCATION_KIND_ENV`, `PHASE_ENV`, `LOGICAL_RUN_LEASE_FILENAME`, `EventName`, and `EXECUTION_EVENT_SCHEMA_VERSION`; optional entry-level `GroupLayout` path resolution, `GroupManifest` / `GroupMember` immutable group provenance, `generate_group_id`, `publish_group_manifest` / `load_group_manifest` atomic manifest publication and reload, schema-v1-style `GroupEvent` / `GroupEventWriter` / `read_group_events` append-only group event streams, incremental `GroupEventTailReader` / `GroupEventReadError` group-event tailing, and the `GROUP_SCHEMA_VERSION` / `GROUP_EVENT_SCHEMA_VERSION` / `GroupEventName` constants; framework-neutral recoverable chunked `WorkStoreSession` claim/commit/verify/cleanup lifecycle with optional caller-selected `<namespace>/leases/` placement protected by a deterministic store coordinator, verified-journal `completed_chunk_ids` / `completed_chunks` chunk-ID-to-marker read-back, `claim_work_store_lease`, `inspect_work_store` fail-closed prior-state classification (`WorkStoreInspection` with the same `completed_chunk_ids` / `completed_chunks` read-back, `WorkStoreStatus`), `cleanup_work_store`, the `mammoth-work-store-jsonl-v1` journal format constants, and the `WorkStoreError` family; and optional entry-level; `TelemetryRecorder`, `TelemetryTailReader`, and `telemetry_stream_path` execution-host telemetry |
| `mammoth.queue` | `QueueClient`, `QueueApiServer`, and `job_document` Unix-socket API; `run_service` and `ServiceAlreadyRunningError` process composition; `QueueService` and `ServiceConfig` state, tick, and recovery; `schedule`, `GpuState`, `Request`, `Reservation`, and `Placement` pure admission; `HostSampler`, `Sampler`, `HostSample`, `GpuSample`, `ExecutionMemory`, and `SamplingError` host sampling; `ActlExecutor`, `Executor`, `ExecutionRecord`, and `ExecutorError` durable execution; `JobStore` persistence; `Job`, `JobSpec`, `JobStatus`, `Dependency`, the `QueueError` family, `parse_size`, `parse_duration`, `format_size`, `default_state_dir`, and `default_socket_path` values |
| `mammoth.execution` | Immutable `ExecutionSpec` plus context-managed direct `ExecutionSession` creation and strict single-process workflow-child attachment; framework-neutral process/phase lifecycle, observability, generic observer/pipeline ownership, lease/logging cleanup, and terminal-outcome handling; `ExecutionObserver` detached no-op phase/task/progress/heartbeat call surface, its shared `NULL_EXECUTION_OBSERVER` instance, and `SessionExecutionObserver` forwarding the same call surface onto a live session; `ExecutionSpec.telemetry` opt-out, and `stage`/`display` on `ExecutionObserver` |
| `mammoth.logging` | `Observation`, `Media`, `ObservationSink`, `RunObserver`, `JsonlEventSink`, `ExecutionObservability`, `ExecutionLogging`, `ProcessTextLogHandler`, `ProcessTextLogLease`, `claim_process_text_log`, `create_execution_observability`, `create_execution_logging`, `create_process_text_handler`; `RunObserver.stage` and `RunObserver.display` presentation events |
| `mammoth.logging.tensorboard` | `TensorBoardSink` |
| `mammoth.monitor` | `ExecutionMonitor`, `RunMonitor`, `MonitorSnapshot`, `RunSnapshot`, `ProducerKey`, `ProducerState`, `TaskState`, `MetricPoint`, `ViewerTelemetry`, single-run discovery/folding/rendering functions, and viewer telemetry sampling; `FleetMonitor`, `FleetSnapshot`, `GroupSnapshot`, `GroupMemberSnapshot`, `StepSnapshot`, `LooseRunSnapshot`, `MemberProgress`, `MemberStatus`, `discover_group_ids`, `discover_run_names`, and `render_fleet_snapshot` / `render_group_snapshot` plain rendering; `StagePoint` and `DisplayItem` folded presentation values |
| `mammoth.workflow` | Immutable `Workflow`, `Run`, `Step`, and `Execution` inputs, including `Workflow`'s optional opaque `group_metadata`; `ExecutionResolutionContext` and `BeforeFirstStepContext` run-local boundaries; side-effect-free `CommandPlan` planning; `WorkflowResult` (with the published `group_id`, if any), `RunResult`, `StepResult`, and `DispatchResult`; the structural `Launcher` protocol for `Workflow`'s optional injectable-launcher seam; supervised process values and captured-process execution |
| `mammoth.torch` | `TorchBackendConfig`, `TorchBackendState`, `TorchSeedPolicy`, backend/seed application and reversible backend context, `TORCH_LOG_COMPONENTS`/`TORCH_LOG_ARTIFACTS` and `parse_torch_logs` generic `TORCH_LOGS`-style token parsing, `autocast_dtype_for_device`/`autocast_context` and `sdpa_backend_context` runtime selection contexts, shared device resolution, `Runtime`, compatibility `ExecutionSpec` re-export, and a context-managed `ExecutionSession` adapter that composes the neutral session with Torch trainer ownership; strict `Runtime.create_execution()` and `Runtime.attach_execution()`, `RuntimeConfig`, `initialize_runtime`, `Trainer`, `TrainerConfig`, `TorchCompileConfig`, `TrainerState`, `TrainerResult`, `StepContext`, `StepFunction`, `StepOutput`, `WarmupLinearLR`, `AccumulationPlan`, `AccumulationPolicy`, `UniformAccumulationPolicy`, `WeightedAccumulationPolicy`, `WeightedDistributedBatchSampler`, `WeightedTaskAssignment`, weighted task allocation and count/index partition helpers, `MetricSpec`, `MetricRoute`, `MetricAccumulator`, `StatefulMetric`, callable profiling configuration/results/runtime controls/report publication, named-phase measurement and immutable summaries, public CUDA synchronization/allocator helpers, public profiler-row normalization, `resolve_profiler_sort_key` legacy/current profiler sort-key normalization, `tensor_metadata` compact tensor layout/health diagnostics, `Callback`, `EarlyStopping`, `CheckpointMode`, `CheckpointCaptureMode`, `CheckpointRole`, `CheckpointSavePolicy`, `CheckpointComponent`, `CheckpointInspection`, `RestoreOptions`, `TrainerCheckpointContext`, `TrainerCheckpointRestore`, `TrainerCheckpointWriters`, `TrainerCheckpointPolicy`, `StateRegistry`, `CheckpointArtifact`, `CheckpointPlan`, `PublishedCheckpoint`, `ResumableCheckpointCandidate`, and standard checkpoint filename/parser/discovery APIs, `CheckpointPublication`, `AsyncCheckpointPublisher` compatibility adapter over the core pipeline, `clone_checkpoint_value` recursive detach-and-clone-to-CPU checkpoint state cloning, recursive batch movement with bounded CUDA copy-stream prefetch for eligible pinned batches, trainer-owned checkpoint selection/naming/retention, ordered publication and exact-byte receipt delivery, checkpoint creation, and checkpoint restoration |

## Command Routes

```text
mammoth.cli.app (Typer)
├── mammoth monitor (Typer sub-app)
│   ├── tui [RUN] [--entry] [--group] [--match] [--execution] → mammoth.cli.monitor_tui
│   │   ├── RUN given → RunMonitor → textual_ui.MonitorApp
│   │   └── RUN omitted → FleetMonitor → textual_ui.FleetApp
│   │       └── FleetScreen → GroupScreen → RunScreen (screen stack)
│   ├── show [RUN] [--entry] [--group] [--match] [--execution] [--watch] [--json]
│   │   └── mammoth.cli.monitor_show
│   │       ├── plain → render_snapshot / render_fleet_snapshot / render_group_snapshot
│   │       └── --json → mammoth.monitor.documents
│   ├── web [--root]... [--host] [--port] → mammoth.cli.monitor_web
│   │   └── mammoth.monitor.web.create_web_server
│   │       └── MonitorService → FleetMonitor sweep, lazy RunMonitor, TelemetryTailReader
│   ├── serve [--socket] [--host] [--port] [--allow-queue-writes] → mammoth.cli.monitor_serve
│   │   └── mammoth.monitor.web.create_queue_monitor_server
│   │       └── MonitorService → QueueClient.run_dirs → queue_entries, plus the Queue page
│   └── presentation contract → docs/MONITOR.md
└── mammoth queue (Typer sub-app)
    ├── submit, list, cancel, priority, wait → mammoth.queue.QueueClient
    └── serve → mammoth.queue.run_service
        ├── hold_state_lock (exclusive per-host service lock)
        └── QueueService.run (tick: track, sample, schedule, launch through actl)

caller-owned Python
    └── mammoth.workflow.Workflow
        ├── construction → freeze inputs + validate names/order + derive canonical dispatch
        ├── plan → project the canonical dispatch to side-effect-free command plans
        └── run (main thread + POSIX pthread_sigmask) → mask handler transitions + own setup/execution signals
            + publish group manifest once (first run only) + open group event stream
            + prepare layout + claim lease
            └── optional resolve_execution → immutable execution context
                └── execution lifecycle → optional before_first_step
                    └── serial supervised step dispatch (launcher seam) + structured result
                        + group run/step lifecycle events + terminal group status
```

## Current Import Graph

`mammoth.compat` supplies interpreter-level dataclass options, exception notes,
and strict iterable pairing to the framework-neutral and optional Torch layers.

```text
mammoth
├── mammoth.cli
│   ├── Typer
│   ├── mammoth.monitor
│   └── mammoth.queue
├── mammoth.queue
│   └── mammoth.core (locking only)
├── mammoth.execution
│   ├── mammoth.core
│   └── mammoth.logging
├── mammoth.monitor
│   ├── mammoth.core
│   └── Textual / Rich / psutil (optional modules only)
├── mammoth.workflow
│   ├── mammoth.core
│   └── mammoth.logging
├── mammoth.torch
│   ├── mammoth.core
│   ├── mammoth.execution
│   ├── mammoth.logging
│   └── PyTorch and torch.profiler (optional extra)
├── mammoth.logging
│   ├── mammoth.core
│   └── tensorboardX (optional module only)
└── mammoth.core
    ├── Python standard library / typing-extensions
    ├── portalocker (pywin32 on Windows)
    └── zstandard (event compression)
```


## Platform backends

`core._filesystem` is the private platform boundary:

- `io` owns binary opens, offset reads, file validation, and replacement.
- `publication` owns prepared artifacts, staging cleanup, and directory sync.
- `confined` binds publication to root identities and confines target operations.
- `locking` adapts `portalocker` descriptor locks and contention errors.
- `text` owns exclusive append-only text descriptors.
- `leases` selects logical-run lease formats and owns Windows recovery.
- `windows` contains native handles and pinned directory traversal.

`core.artifacts` re-exports the established prepared-artifact API and owns byte
verification. `torch.checkpoint` has one ordered publication loop using
`publication_root`; it owns no platform selection or native filesystem calls.
All descriptor locking, including POSIX-only lease/work-store/queue clients,
uses `core._filesystem.locking`. Mammoth does not import `fcntl` directly;
`portalocker` selects the operating-system lock primitive.
Platform guarantees and scope live in `docs/ARCHITECTURE.md`.
