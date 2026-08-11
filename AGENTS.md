# Agent Rules

This file is the first document every agent should read before changing
Mammoth. Read `docs/CODEMAP.md` next for repository navigation, then open only
the files needed for the task. Before changing monitor reconstruction,
presentation, telemetry, refresh, or interaction behavior, read
`docs/MONITOR.md`.

## Core Principles

Before editing, verify whether the current implementation already satisfies the
request. Do not make cosmetic renames, drive-by refactors, or unrelated cleanup.
Preserve unrelated user changes in a dirty worktree.

Treat `docs/ARCHITECTURE.md` as the sole owner of package boundaries, dependency
direction, runtime concepts, artifact layout, logging roles, trainer scope, and
compatibility policy. Do not restate those contracts in another document or
implement behavior that contradicts them.

## Batch Terminology

In agent conversations and status reports, always qualify whether training work
means a `microbatch` or a `logical batch`. A microbatch is one local loader batch
processed inside an accumulation window. A logical batch is the complete
accumulation window that produces one optimizer step.

When a user says `batch` without qualification, infer `logical batch` and state
that inference in the response. Interpret the term as `microbatch` only when the
user says so explicitly. When discussing validation, where optimizer-step
accumulation does not define a logical batch, use `validation DataLoader batch`
explicitly.

## File Ownership

Every current repository file has one primary purpose. Update the owning file
when its subject changes instead of duplicating the same contract elsewhere.

| File | Purpose | Update when |
| --- | --- | --- |
| `AGENTS.md` | Repository rules, dependency boundaries, validation policy, Git prefixes, and file ownership. | Agent workflow, repository conventions, validation, Git policy, or documentation ownership changes. |
| `README.md` | Concise user-facing project introduction and setup entry point. | Public package purpose, installation, stable CLI/API, or user-visible behavior changes. |
| `LICENSE` | MIT license terms and copyright notice. | The project license or copyright notice changes. |
| `docs/ARCHITECTURE.md` | Durable package boundary, dependency direction, runtime model, logging roles, trainer scope, and compatibility policy. | Responsibilities move between layers or a stable architectural contract changes. |
| `docs/MONITOR.md` | Canonical passive-monitor reconstruction, dashboard hierarchy, telemetry, responsive presentation, and interaction behavior. | Monitor folding, layout, display naming or units, resource probes, refresh policy, or keyboard behavior changes. |
| `docs/IMPLEMENTATION_PLAN.md` | Ordered delivery phases, acceptance conditions, migration sequence, and explicitly deferred capabilities. | A phase is started/completed, delivery order changes, or deferred scope is approved. |
| `docs/CODEMAP.md` | Map of implemented source paths, symbols, and import relationships. | Implemented files or symbols move, appear, or disappear, or their import relationships change. |
| `pyproject.toml` | Package identity, Python requirement, build backend, dependencies, CLI entry point, optional extras, and tool configuration. | Packaging, dependencies, commands, build behavior, or development-tool policy changes. |
| `uv.lock` | Exact uv resolution generated from `pyproject.toml`. Never edit manually. | Regenerate with uv whenever dependency metadata changes. |
| `.python-version` | uv/Python development baseline. | The supported development interpreter changes deliberately. |
| `.gitignore` | Generated-file and local-environment exclusions. | A new reproducible build, cache, environment, or local artifact needs an exclusion. |
| `src/mammoth/__init__.py` | Lightweight root package metadata and intentionally small stable exports. | Package version or a truly root-level stable export changes. |
| `src/mammoth/__main__.py` | `python -m mammoth` forwarding entry point. | Module execution behavior changes. |
| `src/mammoth/cli.py` | Public Typer application, typed commands, and console exit routing. | A public command, option, or exit behavior changes. |
| `src/mammoth/py.typed` | PEP 561 marker declaring inline type information. | Keep present and empty while Mammoth ships typed source. |
| `src/mammoth/core/__init__.py` | Public framework-neutral core exports. | A stable core symbol is added, removed, or renamed. |
| `src/mammoth/core/artifacts.py` | Atomic local bytes, text, JSON, and opaque artifact publication. | Local publication durability or writer behavior changes. |
| `src/mammoth/core/events.py` | Schema-v1 event values, append-only producer writers, replay, and active tailing. | Event validation, retention, compatibility, or stream behavior changes. |
| `src/mammoth/core/execution.py` | Immutable execution metadata, lineage, sanitization, discovery, joins, and logical-run leases. | Attempt identity, provenance, compatibility, or lease behavior changes. |
| `src/mammoth/core/identity.py` | Filesystem-safe run-name and execution-ID validation. | Identity syntax or length limits change. |
| `src/mammoth/core/layout.py` | Stable caller-entry/run-name artifact path resolution. | The run-directory contract changes. |
| `src/mammoth/core/pipeline.py` | Framework-neutral ordered background execution, bounded backpressure, ownership, result attribution, interruption recovery, and cleanup. | Generic background pipeline behavior or public values change. |
| `src/mammoth/logging/__init__.py` | Public lightweight logging exports that do not require TensorBoard. | A stable logging symbol is added, removed, or renamed. |
| `src/mammoth/logging/execution.py` | Per-process execution logging bundle for JSONL observations and exclusive text diagnostics. | Execution logging composition or ownership changes. |
| `src/mammoth/logging/jsonl.py` | Adapter from sink-neutral observations to append-only execution events. | JSONL routing or flush behavior changes. |
| `src/mammoth/logging/model.py` | Immutable sink-neutral metric, media, and lifecycle observations. | The logging sink contract changes. |
| `src/mammoth/logging/observer.py` | Producer facade, lifecycle contexts, sink fan-out, and failure isolation. | Producer-facing logging behavior changes. |
| `src/mammoth/logging/tensorboard.py` | Optional rank-aware TensorBoard scalar and media history sink. | TensorBoard routing, step selection, or ownership changes. |
| `src/mammoth/logging/text.py` | Process-exclusive plain-text Python logging handler. | Text-log ownership or formatting changes. |
| `src/mammoth/monitor/__init__.py` | Public passive-monitor exports. | A stable monitor symbol is added, removed, or renamed. |
| `src/mammoth/monitor/model.py` | Execution discovery, lineage, incremental stream reads, and project-neutral state folding. | Monitor selection or reconstructed state changes. |
| `src/mammoth/monitor/dashboard.py` | Responsive Rich renderables for the optional Textual dashboard. | Interactive panels, progress bars, metric charts, or wide/compact presentation changes. |
| `src/mammoth/monitor/psutil_telemetry.py` | Optional psutil-backed viewer-host samples. | Optional CPU or memory sampling changes. |
| `src/mammoth/monitor/render.py` | Canonical stable ANSI-free monitor snapshot rendering. | Plain monitor output changes. |
| `src/mammoth/monitor/rich_ui.py` | Compatibility route from the former Rich helper to the Textual application. | Legacy interactive-helper compatibility changes. |
| `src/mammoth/monitor/telemetry.py` | Standard-library viewer-host telemetry with explicit provenance labels. | Base local telemetry changes. |
| `src/mammoth/monitor/textual_ui.py` | Optional Textual application lifecycle, refresh worker, navigation, and resize handling. | Interactive monitor lifecycle, bindings, or polling behavior changes. |
| `src/mammoth/workflow/__init__.py` | Public programmatic workflow planning, supervision, and result exports. | A stable workflow symbol is added, removed, or renamed. |
| `src/mammoth/workflow/launch.py` | Final-argv subprocess launch plus reusable launcher/descendant supervision. | Launch, timeout, signal, or descendant handling changes. |
| `src/mammoth/workflow/runner.py` | Side-effect-free planning, serial run/step attempts, narrow hooks, canonical child environments, and lifecycle ownership. | Workflow orchestration behavior changes. |
| `src/mammoth/torch/__init__.py` | Public optional runtime, trainer, profiler, callback, metric, batch, and checkpoint exports. | A stable PyTorch integration symbol is added, removed, or renamed. |
| `src/mammoth/torch/backend.py` | Generic process-global PyTorch numerical backend configuration, state capture, reversible overrides, and RNG seed policy. | TF32, matmul precision, cuDNN, deterministic-algorithm, or seed behavior changes. |
| `src/mammoth/torch/device.py` | Shared explicit and automatic PyTorch device resolution. | Generic device-string resolution or availability validation changes. |
| `src/mammoth/torch/runtime.py` | Generic single/DDP process-group, collective, execution-startup, and rank-logging lifecycle. | PyTorch runtime identity, collectives, startup consensus, or cleanup changes. |
| `src/mammoth/torch/batch.py` | Recursive common-container tensor transfer to one torch device. | Default batch transfer behavior changes. |
| `src/mammoth/torch/callbacks.py` | Generic trainer lifecycle callbacks and metric-based early stopping. | Callback hooks or early-stopping behavior changes. |
| `src/mammoth/torch/checkpoint.py` | Registered state, project trainer-checkpoint contracts, restore, and bounded asynchronous atomic publication. | Trainer checkpoint policy, publication mechanics, or schema changes. |
| `src/mammoth/torch/metrics.py` | Scalar reductions, additive stateful metrics, and batch/epoch sink routing. | Metric aggregation or routing policy changes. |
| `src/mammoth/torch/profiling.py` | Model-independent callable timing, Torch operation profiling, runtime controls, and versioned reports. | Generic PyTorch profiling behavior or report schema changes. |
| `src/mammoth/torch/scheduling.py` | Generic warmup-linear learning-rate scheduling, weighted rank partitioning, distributed batch sampling, accumulation plans, policies, and logical-window loss scales. | Learning-rate schedules, workload partitioning, sampler, accumulation-policy, or scaling behavior changes. |
| `src/mammoth/torch/state.py` | Serializable ordinary trainer loop coordinates. | Trainer resume coordinates change. |
| `src/mammoth/torch/trainer.py` | Constructed-object single/DDP loops, project policy integration, validation, observability, and checkpoint lifecycle. | Generic trainer behavior or policy integration changes. |
| `tests/test_artifacts.py` | Atomic artifact publication unit coverage. | Artifact publication behavior changes. |
| `tests/test_events.py` | Event validation, writer, replay, tailing, and legacy-field unit coverage. | Event behavior changes. |
| `tests/test_execution.py` | Execution metadata, lineage, sanitization, compatibility, and lease unit coverage. | Execution behavior changes. |
| `tests/test_layout.py` | Run identity and artifact-layout unit coverage. | Layout or run-name validation changes. |
| `tests/test_pipeline.py` | Ordered background execution, backpressure, attribution, interruption, and cleanup coverage. | Generic background pipeline behavior changes. |
| `tests/test_logging.py` | Observer, sink isolation, JSONL routing, text, and TensorBoard unit coverage. | Logging behavior changes. |
| `tests/test_cli.py` | Typer version, monitor help, usage-error, and removed-workflow-route coverage. | Root or monitor CLI behavior changes. |
| `tests/test_monitor.py` | Discovery, lineage, folding, rendering, telemetry, malformed streams, and CLI unit coverage. | Monitor or monitor CLI behavior changes. |
| `tests/test_workflow.py` | Programmatic models, both serial orders, lifecycle boundaries, failure results, canonical environments, and subprocess supervision coverage. | Workflow behavior changes. |
| `tests/test_torch.py` | Multi-task trainer, device movement, precision, accumulation, metrics, callbacks, checkpoints, and DDP unit coverage. | Optional trainer behavior changes. |
| `tests/test_backend.py` | Generic PyTorch backend configuration, restoration, and seed-policy coverage. | Backend or seed configuration behavior changes. |
| `tests/test_profiling.py` | Callable profiling, output summaries, component ranges, runtime restoration, report, and CUDA-conditional coverage. | Generic PyTorch profiling behavior changes. |

Generated `.venv/`, `dist/`, caches, and build metadata are not source files.
Do not document or commit their generated contents.

## Issue Workflows

Before creating an issue, investigate the request, search for duplicates, and
prepare the exact title, body, and acceptance criteria for user approval. Create
and verify the issue only after that approval. Issue creation does not authorize
implementation.

Implement an issue only when the existing issue has an explicit pickup request.
Use an isolated task branch and worktree, satisfy the approved scope, run
proportionate validation, and obtain an independent clean-context review before
publication. Pushing or opening a pull request, merging, and closing the issue
require their applicable explicit approvals. After merge, verify the exact
commit and issue state before removing only clean task branches and worktrees.

## Python And API Standards

- Use Python 3.12 syntax unless the project baseline is deliberately changed.
- Add a module docstring to every Python module.
- Keep all imports at module top level after the docstring and
  `from __future__ import annotations`.
- Use protocols and small immutable value objects at integration boundaries.
- Prefer explicit dependency injection over imports from consuming projects.
- Keep filesystem writes atomic where the public contract promises durable
  publication.
- Never persist a complete process environment or unsanitized credentials.
- Do not add a compatibility abstraction until there is a real consumer for it.

## uv Workflow

Use uv for all environment, dependency, build, and command operations:

```bash
uv sync
uv run <command>
uv add <dependency>
uv remove <dependency>
uv lock --check
uv build
```

Do not use `pip install` directly. Add optional integrations to a suitable
extra or dependency group rather than making core imports pull them in.
Whenever `pyproject.toml` dependency metadata changes, update `uv.lock` in the
same change.

## Validation

After every Python change, run:

```bash
uv sync
uv lock --check
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Use `uv run pytest --cov=mammoth --cov-report=term-missing` for completion and
release audits. For documentation-only changes, inspect the diff and run
`git diff --check`. Do not claim a check passed unless it was actually run.

## Git Policy

Do not stage, commit, push, create branches, or open pull requests unless the
user requests that Git operation. Keep unrelated changes unstaged.

Every commit message must use exactly one of these pinned prefixes followed by
a colon and an imperative summary:

| Prefix | Use for |
| --- | --- |
| `feat` | New user-facing or infrastructure functionality. |
| `fix` | Correctness fixes and restoration of intended behavior. |
| `refactor` | Code restructuring with no intended behavior change. |
| `docs` | Documentation-only changes. |
| `test` | Test-only changes or added test coverage. |
| `conf` | Runtime or project configuration changes. |
| `perf` | Measured performance improvements with unchanged semantics. |
| `build` | Packaging, dependencies, lockfile, or build-system changes. |
| `ci` | Continuous-integration configuration changes. |

Example: `feat: add append-only execution event writer`.

Task branches, when requested, must use the matching pinned category prefix:
`feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `conf/`, `perf/`, `build/`, or
`ci/`, followed by a concise lowercase hyphenated name. Do not put a person,
agent, model, tool, or session identity in a branch name.

## Documentation Maintenance

The file-ownership table above is the sole index of documentation purposes.
Do not repeat that index in another file.

Do not use repository Markdown as a parallel issue backlog. Record only durable
contracts, current implementation state, and an approved implementation
sequence. Update every affected owner document in the same change when a
public path, dependency boundary, artifact contract, or module responsibility
changes.
