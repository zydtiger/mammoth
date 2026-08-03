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
│   └── layout.py
├── logging/
│   ├── __init__.py
│   ├── execution.py
│   ├── jsonl.py
│   ├── model.py
│   ├── observer.py
│   ├── tensorboard.py
│   └── text.py
├── monitor/
│   ├── __init__.py
│   ├── model.py
│   ├── psutil_telemetry.py
│   ├── render.py
│   ├── rich_ui.py
│   └── telemetry.py
├── workflow/
│   ├── __init__.py
│   ├── config.py
│   ├── launch.py
│   └── runner.py
├── torch/
│   ├── __init__.py
│   ├── batch.py
│   ├── callbacks.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── runtime.py
│   ├── state.py
│   └── trainer.py
└── py.typed
```

## Public Symbol Index

| Import path | Symbols |
| --- | --- |
| `mammoth` | `__version__` |
| `mammoth.core` | `RunLayout`, `ExecutionContext`, `ExecutionMetadata`, `LogicalRunLease`, `ExecutionEvent`, `ExecutionEventWriter`, `ExecutionEventTailReader`, `ExecutionEventReadError`, artifact writers, execution lifecycle functions, event readers, sanitizers, and identity validators |
| `mammoth.logging` | `Observation`, `Media`, `ObservationSink`, `RunObserver`, `JsonlEventSink`, `ExecutionLogging`, `ProcessTextLogHandler`, `ProcessTextLogLease`, `claim_process_text_log`, `create_execution_logging`, `create_process_text_handler` |
| `mammoth.logging.tensorboard` | `TensorBoardSink` |
| `mammoth.monitor` | `ExecutionMonitor`, `MonitorSnapshot`, `ProducerKey`, `ProducerState`, `TaskState`, `MetricPoint`, `ViewerTelemetry`, discovery/folding/rendering functions, and viewer telemetry sampling |
| `mammoth.workflow` | `WorkflowConfig`, `RunConfig`, `StepConfig`, `CommandPlan`, `SupervisedProcess`, `ProcessResult`, `WorkflowResult`, `RunResult`, `StepResult`, workflow loading/planning/running functions, and command construction |
| `mammoth.torch` | `TorchExecutionRuntime`, `TorchRuntimeConfig`, `TorchExecutionRequest`, `initialize_torch_runtime`, `Trainer`, `TrainerConfig`, `TrainerState`, `TrainerResult`, `StepContext`, `StepOutput`, `Callback`, `EarlyStopping`, `MetricSpec`, `MetricAccumulator`, `StateRegistry`, `AsyncCheckpointPublisher`, batch movement, checkpoint creation, and checkpoint restoration |

## Command Routes

```text
mammoth.cli.app (Typer)
├── mammoth monitor
│   └── mammoth.cli.run_monitor
│       └── mammoth.monitor.ExecutionMonitor
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
│   └── rich / psutil (optional modules only)
├── mammoth.workflow
│   ├── mammoth.core
│   ├── mammoth.logging
│   └── PyYAML
├── mammoth.torch
│   ├── mammoth.core
│   ├── mammoth.logging
│   └── PyTorch (optional extra)
├── mammoth.logging
│   ├── mammoth.core
│   └── tensorboardX (optional module only)
└── mammoth.core
    └── Python standard library
```
