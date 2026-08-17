# Mammoth Code Map

## Source Tree

```text
src/mammoth/
├── __init__.py
├── __main__.py
├── cli.py
├── core/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── events.py
│   ├── execution.py
│   ├── identity.py
│   ├── layout.py
│   ├── pipeline.py
│   └── transactions.py
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
│   ├── model.py
│   ├── psutil_telemetry.py
│   ├── render.py
│   ├── rich_ui.py
│   ├── telemetry.py
│   └── textual_ui.py
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

## Public Symbol Index

| Import path | Symbols |
| --- | --- |
| `mammoth` | `__version__` |
| `mammoth.core` | `RunLayout`, `ExecutionContext`, `ExecutionMetadata`, `LogicalRunLease`, unitless `ExecutionEvent` and `ExecutionEventWriter` APIs with schema-v1 legacy-unit reading, `ExecutionEventTailReader`, `ExecutionEventReadError`, `BoundedBackgroundPipeline`, typed background submissions/results, attributed background failures, `ArtifactReceipt`, `inspect_artifact`, `verify_artifact`, descriptor-bound `ArtifactReadSession` / `open_artifact_session` parsing, atomic and prepared artifact publication helpers, and `ArtifactTransactionPlan` / `TransactionArtifact` planning, sealing, per-root leasing, local publication, coordinated journal recovery, and typed result APIs for several local file or directory targets across declared non-overlapping roots; explicit-only execution lifecycle functions, event readers, sanitizers, resume-checkpoint SHA-256 validation, and identity validators |
| `mammoth.logging` | `Observation`, `Media`, `ObservationSink`, `RunObserver`, `JsonlEventSink`, `ExecutionObservability`, `ExecutionLogging`, `ProcessTextLogHandler`, `ProcessTextLogLease`, `claim_process_text_log`, `create_execution_observability`, `create_execution_logging`, `create_process_text_handler` |
| `mammoth.logging.tensorboard` | `TensorBoardSink` |
| `mammoth.monitor` | `ExecutionMonitor`, `RunMonitor`, `MonitorSnapshot`, `RunSnapshot`, `ProducerKey`, `ProducerState`, `TaskState`, `MetricPoint`, `ViewerTelemetry`, discovery/folding/rendering functions, and viewer telemetry sampling |
| `mammoth.workflow` | Immutable `Workflow`, `Run`, `Step`, and `Execution` inputs; `ExecutionResolutionContext` and `BeforeFirstStepContext` run-local boundaries; side-effect-free `CommandPlan` planning; `WorkflowResult`, `RunResult`, `StepResult`, and `DispatchResult`; supervised process values and captured-process execution |
| `mammoth.torch` | `TorchBackendConfig`, `TorchBackendState`, `TorchSeedPolicy`, backend/seed application and reversible backend context, shared device resolution, `Runtime`, context-managed `ExecutionSession` with owned observer/background-pipeline/trainer factories, `RuntimeConfig`, immutable `ExecutionSpec`, strict `Runtime.create_execution()` and `Runtime.attach_execution()`, `initialize_runtime`, `Trainer`, `TrainerConfig`, `TorchCompileConfig`, `TrainerState`, `TrainerResult`, `StepContext`, `StepFunction`, `StepOutput`, `WarmupLinearLR`, `AccumulationPlan`, `AccumulationPolicy`, `UniformAccumulationPolicy`, `WeightedAccumulationPolicy`, `WeightedDistributedBatchSampler`, `WeightedTaskAssignment`, weighted task allocation and count/index partition helpers, `MetricSpec`, `MetricRoute`, `MetricAccumulator`, `StatefulMetric`, callable profiling configuration/results/runtime controls/report publication, named-phase measurement and immutable summaries, public CUDA synchronization/allocator helpers, public profiler-row normalization, `Callback`, `EarlyStopping`, `CheckpointMode`, `CheckpointCaptureMode`, `CheckpointRole`, `CheckpointSavePolicy`, `CheckpointComponent`, `CheckpointInspection`, `RestoreOptions`, `TrainerCheckpointContext`, `TrainerCheckpointRestore`, `TrainerCheckpointWriters`, `TrainerCheckpointPolicy`, `StateRegistry`, `CheckpointArtifact`, `CheckpointPlan`, `PublishedCheckpoint`, `ResumableCheckpointCandidate`, and standard checkpoint filename/parser/discovery APIs, `CheckpointPublication`, `AsyncCheckpointPublisher` compatibility adapter over the core pipeline, recursive batch movement with bounded CUDA copy-stream prefetch for eligible pinned batches, trainer-owned checkpoint selection/naming/retention, ordered publication and exact-byte receipt delivery, checkpoint creation, and checkpoint restoration |

## Command Routes

```text
mammoth.cli.app (Typer)
└── mammoth monitor
    └── mammoth.cli.run_monitor
        ├── redirected/plain → mammoth.monitor.ExecutionMonitor
        └── interactive TTY → mammoth.monitor.RunMonitor
            └── mammoth.monitor.textual_ui.MonitorApp
                └── presentation contract → docs/MONITOR.md

caller-owned Python
    └── mammoth.workflow.Workflow
        ├── construction → freeze inputs + validate names/order + derive canonical dispatch
        ├── plan → project the canonical dispatch to side-effect-free command plans
        └── run (main thread + POSIX pthread_sigmask) → mask handler transitions + own setup/execution signals
            + prepare layout + claim lease
            └── optional resolve_execution → immutable execution context
                └── execution lifecycle → optional before_first_step
                    └── serial supervised step dispatch + structured result
```

## Current Import Graph

```text
mammoth
├── mammoth.cli
│   ├── Typer
│   └── mammoth.monitor
├── mammoth.monitor
│   ├── mammoth.core
│   └── Textual / Rich / psutil (optional modules only)
├── mammoth.workflow
│   ├── mammoth.core
│   └── mammoth.logging
├── mammoth.torch
│   ├── mammoth.core
│   ├── mammoth.logging
│   └── PyTorch and torch.profiler (optional extra)
├── mammoth.logging
│   ├── mammoth.core
│   └── tensorboardX (optional module only)
└── mammoth.core
    └── Python standard library
```
