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
│   └── pipeline.py
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
│   ├── config.py
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
| `mammoth.core` | `RunLayout`, `ExecutionContext`, `ExecutionMetadata`, `LogicalRunLease`, `ExecutionEvent`, `ExecutionEventWriter`, `ExecutionEventTailReader`, `ExecutionEventReadError`, `BoundedBackgroundPipeline`, typed background submissions/results, attributed background failures, atomic and prepared artifact publication helpers, explicit-only execution lifecycle functions, event readers, sanitizers, and identity validators |
| `mammoth.logging` | `Observation`, `Media`, `ObservationSink`, `RunObserver`, `JsonlEventSink`, `ExecutionObservability`, `ExecutionLogging`, `ProcessTextLogHandler`, `ProcessTextLogLease`, `claim_process_text_log`, `create_execution_observability`, `create_execution_logging`, `create_process_text_handler` |
| `mammoth.logging.tensorboard` | `TensorBoardSink` |
| `mammoth.monitor` | `ExecutionMonitor`, `RunMonitor`, `MonitorSnapshot`, `RunSnapshot`, `ProducerKey`, `ProducerState`, `TaskState`, `MetricPoint`, `ViewerTelemetry`, discovery/folding/rendering functions, and viewer telemetry sampling |
| `mammoth.workflow` | `WorkflowConfig`, `RunConfig`, `StepConfig`, `CommandPlan`, `SupervisedProcess`, `ProcessResult`, `WorkflowResult`, `RunResult`, `StepResult`, workflow loading/planning/running functions, and command construction |
| `mammoth.torch` | `TorchBackendConfig`, `TorchBackendState`, `TorchSeedPolicy`, backend/seed application and reversible backend context, shared device resolution, `TorchExecutionRuntime`, context-managed `TorchExecutionSession` with owned observer/background-pipeline/trainer factories, `TorchRuntimeConfig`, `TorchExecutionRequest`, `initialize_torch_runtime`, `Trainer`, `TrainerConfig`, `TorchCompileConfig`, `TrainerState`, `TrainerResult`, `StepContext`, `StepFunction`, `StepOutput`, `WarmupLinearLR`, `AccumulationPlan`, `AccumulationPolicy`, `UniformAccumulationPolicy`, `WeightedAccumulationPolicy`, `WeightedDistributedBatchSampler`, `WeightedTaskAssignment`, weighted task allocation and count/index partition helpers, `MetricSpec`, `MetricRoute`, `MetricAccumulator`, `StatefulMetric`, callable profiling configuration/results/runtime controls/report publication, `Callback`, `EarlyStopping`, `CheckpointMode`, `CheckpointCaptureMode`, `CheckpointRole`, `CheckpointSavePolicy`, `CheckpointComponent`, `CheckpointInspection`, `RestoreOptions`, `TrainerCheckpointContext`, `TrainerCheckpointRestore`, `TrainerCheckpointWriters`, `TrainerCheckpointPolicy`, `StateRegistry`, `CheckpointArtifact`, `CheckpointPlan`, `PublishedCheckpoint`, `CheckpointPublication`, `AsyncCheckpointPublisher` compatibility adapter over the core pipeline, batch movement, trainer-owned checkpoint selection/naming/retention, ordered publication and exact-byte receipt delivery, checkpoint creation, and checkpoint restoration |

## Command Routes

```text
mammoth.cli.app (Typer)
├── mammoth monitor
│   └── mammoth.cli.run_monitor
│       ├── redirected/plain → mammoth.monitor.ExecutionMonitor
│       └── interactive TTY → mammoth.monitor.RunMonitor
│           └── mammoth.monitor.textual_ui.MonitorApp
│               └── presentation contract → docs/MONITOR.md
└── mammoth workflow
    └── mammoth workflow run
        └── mammoth.cli.run_workflow_command
            └── mammoth.workflow.run_workflow
```

## Current Import Graph

```text
mammoth
├── mammoth.cli
│   ├── Typer
│   ├── mammoth.monitor
│   └── mammoth.workflow
├── mammoth.monitor
│   ├── mammoth.core
│   └── Textual / Rich / psutil (optional modules only)
├── mammoth.workflow
│   ├── mammoth.core
│   ├── mammoth.logging
│   └── PyYAML
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
