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
└── py.typed
```

## Implemented Symbols

`src/mammoth/core/` provides the implemented framework-neutral foundation:

```text
RunLayout and identity validation
atomic local artifact publication
immutable execution metadata and producer leases
credential-safe provenance fields
schema-v1 execution event writers, replay, and active tailing
```

`src/mammoth/logging/` provides the producer-facing `RunObserver`, the JSONL
adapter, process text-log handler, and optional rank-aware TensorBoard sink.

`src/mammoth/monitor/` discovers attempts, incrementally folds event streams,
renders stable plain snapshots, and optionally provides Rich and psutil views.
`src/mammoth/cli.py` exposes the installed `mammoth` command.

`src/mammoth/workflow/` strictly parses schema-v1 YAML, resolves step DAGs,
constructs local or `torchrun` plans, supervises process groups, and emits
runner-owned lifecycle events.

`src/mammoth/__init__.py` exports only `mammoth.__version__`. The public core
surface is exported by `mammoth.core`.

## Current Import Graph

```text
mammoth
├── mammoth.cli
│   └── mammoth.monitor
├── mammoth.monitor
│   ├── mammoth.core
│   └── rich / psutil (optional modules only)
├── mammoth.workflow
│   ├── mammoth.core
│   ├── mammoth.logging
│   └── PyYAML
├── mammoth.logging
│   ├── mammoth.core
│   └── tensorboardX (optional module only)
└── mammoth.core
    └── Python standard library
```
