# Mammoth Code Map

## Source Tree

```text
src/mammoth/
├── __init__.py
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

`src/mammoth/__init__.py` exports only `mammoth.__version__`. The public core
surface is exported by `mammoth.core`.

## Current Import Graph

```text
mammoth
├── mammoth.logging
│   ├── mammoth.core
│   └── tensorboardX (optional module only)
└── mammoth.core
    └── Python standard library
```
