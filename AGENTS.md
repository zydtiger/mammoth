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
| `.pre-commit-config.yaml` | The mechanical gate: lock consistency, Ruff, mypy, and file-level checks at the commit stage, and pytest at the `pre-push` stage. Run by `prek` locally and by CI over every file. | A mechanically checkable rule is added, removed, or rescoped. |
| `.github/workflows/ci.yml` | GitHub Actions validation: a `lint` job running the commit-stage hooks once, and a matrixed `test` job running the `pre-push` stage across the supported Python versions, then `uv build` and an install smoke test on the lowest one. Invokes the hook runner rather than restating hook commands. | CI triggers, jobs, the tested Python versions, or validation coverage change. |
| `src/mammoth/__init__.py` | Lightweight root package metadata and intentionally small stable exports. | Package version or a truly root-level stable export changes. |
| `src/mammoth/__main__.py` | `python -m mammoth` forwarding entry point. | Module execution behavior changes. |
| `src/mammoth/cli.py` | Public Typer application, typed commands, and console exit routing. | A public command, option, or exit behavior changes. |
| `src/mammoth/py.typed` | PEP 561 marker declaring inline type information. | Keep present and empty while Mammoth ships typed source. |
| `src/mammoth/execution.py` | Framework-neutral direct execution sessions: immutable specs, create/strict-attach lifecycle, phase and terminal events, logging and heartbeat-capable observer ownership, generic pipelines, leases, deterministic cleanup, and a detached no-op/session-backed observer call surface. | Neutral direct-session lifecycle, spec fields, attach strictness, cleanup ordering, or the no-op/session-backed observer call surface changes. |
| `src/mammoth/core/__init__.py` | Public framework-neutral core exports. | A stable core symbol is added, removed, or renamed. |
| `src/mammoth/core/artifacts.py` | Atomic local bytes, text, JSON, and opaque artifact publication. | Local publication durability or writer behavior changes. |
| `src/mammoth/core/events.py` | Schema-v1 event values, append-only producer writers, replay, and active tailing. | Event validation, retention, compatibility, or stream behavior changes. |
| `src/mammoth/core/execution.py` | Immutable execution metadata, lineage, sanitization, discovery, joins, logical-run leases, and immutable-log-entry classification for consumer log resets. | Attempt identity, provenance, compatibility, lease, or immutable-log-entry classification behavior changes. |
| `src/mammoth/core/groups.py` | Optional entry-level group manifests (atomic publication, generated-ID collision retry, opaque caller-metadata round trip), schema-v1-style append-only group event streams, and incremental `GroupEventTailReader` group-event tailing. | Group manifest fields, group ID generation, or group event schema/durability/tailing behavior changes. |
| `src/mammoth/core/identity.py` | Filesystem-safe run-name, execution-ID, group-ID, and device-spec validation, plus `derive_run_name` stable path-derived run-name construction. | Identity syntax, length limits, or derived-name construction changes. |
| `src/mammoth/core/layout.py` | Stable caller-entry/run-name artifact path resolution, plus `GroupLayout` entry/group-ID group artifact path resolution and `QueueLayout` entry-level device-queue path resolution. | The run-directory, group-directory, or queue-directory contract changes. |
| `src/mammoth/core/leases.py` | Framework-neutral retireable lease namespaces, generation fencing, terminal retirement, and crash reconciliation. | Publication-scoped lease acquisition, retirement, cleanup, recovery, or filesystem safety changes. |
| `src/mammoth/core/pipeline.py` | Framework-neutral ordered background execution, bounded backpressure, ownership, result attribution, interruption recovery, and cleanup. | Generic background pipeline behavior or public values change. |
| `src/mammoth/core/transactions.py` | Multi-artifact transaction planning, exclusive staging, per-root leasing, journal schemas, coordinated publication, and semantic recovery preflight. | Transaction planning, staging, journal, publication, or recovery behavior changes. |
| `src/mammoth/core/workstore.py` | Framework-neutral recoverable chunked work-store leasing, hash-chained completion journal, durable creation, fail-closed prior-state classification, and verified-journal committed-marker read-back. | Work-store leasing, journal format, durability, classification, or committed-marker read-back behavior changes. |
| `src/mammoth/queue/__init__.py` | Public device-aware job queue exports. | A stable queue symbol is added, removed, or renamed. |
| `src/mammoth/queue/spool.py` | Job and job-outcome value objects, atomic pending-spool submission/listing/cancellation, and the shared multi-writer-safe completion journal. | Job/journal schema, spool-file atomicity, cancellation refusal, or journal durability behavior changes. |
| `src/mammoth/queue/serve.py` | Exclusive per-device lease claiming, fail-closed crash reconciliation, and the foreground FIFO device-lane runner over `mammoth.workflow.launch`. | Device-lease semantics, claim atomicity, interruption classification, or the serve-loop signal-handling contract changes. |
| `src/mammoth/logging/__init__.py` | Public lightweight logging exports that do not require TensorBoard. | A stable logging symbol is added, removed, or renamed. |
| `src/mammoth/logging/execution.py` | Per-process execution logging bundle for JSONL observations and exclusive text diagnostics. | Execution logging composition or ownership changes. |
| `src/mammoth/logging/jsonl.py` | Adapter from sink-neutral observations to append-only execution events. | JSONL routing or flush behavior changes. |
| `src/mammoth/logging/model.py` | Immutable sink-neutral metric, media, and lifecycle observations. | The logging sink contract changes. |
| `src/mammoth/logging/observer.py` | Producer facade, lifecycle contexts, sink fan-out, and failure isolation. | Producer-facing logging behavior changes. |
| `src/mammoth/logging/tensorboard.py` | Optional rank-aware TensorBoard scalar and media history sink. | TensorBoard routing, step selection, or ownership changes. |
| `src/mammoth/logging/text.py` | Process-exclusive plain-text Python logging handler. | Text-log ownership or formatting changes. |
| `src/mammoth/monitor/__init__.py` | Public passive-monitor exports. | A stable monitor symbol is added, removed, or renamed. |
| `src/mammoth/monitor/model.py` | Single-run execution discovery, lineage, incremental stream reads, and project-neutral state folding. | Monitor selection or reconstructed single-run state changes. |
| `src/mammoth/monitor/fleet.py` | Passive fleet and group discovery and folding from group manifests, incrementally tailed group event streams, and cheap per-run execution tails; ad-hoc `--match` grouping. | Fleet or group roll-up fields, folding sources, or ad-hoc grouping behavior changes. |
| `src/mammoth/monitor/dashboard.py` | Responsive Rich renderables for the optional Textual dashboard, at the run, fleet, and group levels. | Interactive panels, progress bars, metric charts, or wide/compact presentation changes at any level. |
| `src/mammoth/monitor/psutil_telemetry.py` | Optional psutil-backed viewer-host samples. | Optional CPU or memory sampling changes. |
| `src/mammoth/monitor/render.py` | Canonical stable ANSI-free monitor snapshot rendering for the run, fleet, and group views. | Plain monitor output changes at any level. |
| `src/mammoth/monitor/rich_ui.py` | Compatibility route from the former Rich helper to the Textual application. | Legacy interactive-helper compatibility changes. |
| `src/mammoth/monitor/telemetry.py` | Standard-library viewer-host telemetry with explicit provenance labels. | Base local telemetry changes. |
| `src/mammoth/monitor/textual_ui.py` | Optional Textual application lifecycle, refresh worker, the Fleet -> Group -> Run screen stack, navigation, and resize handling. | Interactive monitor lifecycle, screen-stack navigation, bindings, or polling behavior changes. |
| `src/mammoth/workflow/__init__.py` | Public programmatic workflow planning, supervision, and result exports. | A stable workflow symbol is added, removed, or renamed. |
| `src/mammoth/workflow/launch.py` | Final-argv subprocess launch plus reusable launcher/descendant supervision. | Launch, timeout, signal, or descendant handling changes. |
| `src/mammoth/workflow/runner.py` | Side-effect-free planning, serial run/step attempts, narrow hooks, canonical child environments, lifecycle ownership, and lazy entry-level group manifest/event publication. | Workflow orchestration or group persistence behavior changes. |
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
| `tests/test_execution.py` | Execution metadata, lineage, sanitization, compatibility, lease, and immutable-log-entry classification unit coverage. | Execution behavior changes. |
| `tests/test_groups.py` | Group-ID validation, `GroupLayout` path resolution, group manifest atomicity/collision/metadata-round-trip, and group event writer/reader/incremental-tail-reader unit coverage. | Group manifest, layout, or event-stream/tailing behavior changes. |
| `tests/test_layout.py` | Run identity, `derive_run_name`, and artifact-layout unit coverage. | Layout, run-name validation, or derived-name construction changes. |
| `tests/test_leases.py` | Retireable lease namespace contention, generation fencing, retirement, crash reconciliation, and safety coverage. | Lease namespace behavior changes. |
| `tests/test_pipeline.py` | Ordered background execution, backpressure, attribution, interruption, and cleanup coverage. | Generic background pipeline behavior changes. |
| `tests/test_transactions.py` | Transaction planning, staging, leasing, journal, publication, recovery, and interruption unit coverage. | Transaction behavior changes. |
| `tests/test_workstore.py` | Work-store lease exclusivity, interruption-and-resume, journal tamper detection, prior-state classification, durability, cleanup-ordering, and committed-marker read-back coverage. | Work-store behavior changes. |
| `tests/test_queue.py` | Job/journal validation, device-lease exclusivity and lane concurrency, FIFO ordering, device matching, single-claim contention, before/during/after interruption classification, and `mammoth queue submit/list/cancel/serve` CLI coverage. | Queue behavior or its CLI commands change. |
| `tests/test_logging.py` | Observer, sink isolation, JSONL routing, text, and TensorBoard unit coverage. | Logging behavior changes. |
| `tests/test_cli.py` | Typer version, monitor help, usage-error, removed-workflow-route, and fleet/group CLI invocation coverage. | Root or monitor CLI behavior changes. |
| `tests/test_fleet.py` | Fleet and group folding from group manifests, group event streams, and run tails (including partial writes and crashed producers), ad-hoc `--match` grouping, and their plain-mode rendering. | Fleet or group folding, discovery, or plain-mode rendering behavior changes. |
| `tests/test_monitor.py` | Single-run discovery, lineage, folding, rendering, telemetry, malformed streams, CLI unit coverage, and fleet/group Rich presentation and Textual screen-stack navigation. | Monitor or monitor CLI behavior changes. |
| `tests/test_workflow.py` | Programmatic models, both serial orders, lifecycle boundaries, failure results, canonical environments, subprocess supervision, and group manifest/event publication (including terminal status on success, failure, and signal-driven interruption) coverage. | Workflow or group persistence behavior changes. |
| `tests/test_execution_session.py` | Neutral direct-session create/attach, lifecycle event, cleanup-ordering, torch-free import, and no-op/session-backed observer coverage. | Neutral direct-session behavior or the no-op/session-backed observer call surface changes. |
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

Lock consistency, Ruff, and mypy are enforced by the commit-stage hooks and
pytest by the `pre-push` stage hook; both stages are what CI runs over every
file. Install the runner once per machine:

```bash
uv tool install prek
prek install
```

After every Python change, run what the hooks do not cover:

```bash
uv sync
uv build
```

Apply the hooks outside a commit with `prek run --all-files`, and the test
stage with `prek run --all-files --hook-stage pre-push`.

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

## Versioning And Releases

Mammoth is pre-1.0 and versioned with semantic versioning. The public surfaces
are the importable API under `src/mammoth/`, the on-disk journal and store
layouts, and the CLI's commands, options, and exit behavior. A change is
breaking only against one of those.

- Before `1.0.0`, bump the minor version for a breaking change to any public
  surface and the patch version for everything else, including fixes and
  additive API, layout, or CLI surface.
- From `1.0.0` on, a breaking change requires a major bump, new
  backward-compatible functionality a minor bump, and a compatible fix a patch
  bump.

Before `1.0.0` the project is under active development, so a breaking change is
expected and needs no migration path; still name it as breaking in the release
notes. `1.0.0` is a deliberate commitment to the stability of those surfaces,
not the number after `0.9`; confirm reaching it with the user. After `1.0.0`,
prefer deprecating the old form, introducing its replacement alongside it,
giving consumers at least one minor release to migrate, and removing the old
form in the next major release. Break compatibility directly only for a genuine
mistake whose cost outweighs migration, and record why no deprecation path was
possible.

Keep `pyproject.toml` as the package-version source of truth, keep
`src/mammoth/__init__.py` `__version__` equal to it, and refresh `uv.lock`
whenever the version changes.

Release only from a clean `main` synchronized with `origin/main`, after the
change has merged, so a release records what shipped rather than what is
proposed.

1. Verify `main` is synchronized and passes the validation commands above.
2. Set the version in `pyproject.toml` and `__init__.py`, refresh `uv.lock`,
   and commit with `build: release Mammoth X.Y.Z`.
3. Create an annotated `vX.Y.Z` tag on that exact commit and push it.
4. Publish a GitHub Release from that tag. Its notes are the version history:
   state user-visible changes, grouped as added, changed, fixed, deprecated,
   and breaking, and give each deprecation its replacement and each breaking
   entry its migration path or the reason none was possible.
5. Verify the published tag, the release, and installation from the exact tag.

Publishing a release is one action. Ask for explicit approval once, immediately
before pushing the release tag, presenting the exact version, the commit it
will point at, and the rationale for that number. That approval covers
publishing the GitHub Release from that tag. Ask again only for a different
version or commit, or to modify or delete a release that already exists.
Approval to edit, commit, push, or merge ordinary changes is never release
approval.

Never move or replace a published release tag; correct a mistaken release by
publishing the next version. A repository ruleset named `protect-release-tags`
enforces this on the remote for `refs/tags/v*`, denying tag deletion,
non-fast-forward updates, and updates, with no bypass. A rejected tag push is
that rule working, not a broken remote.

## Documentation Maintenance

The file-ownership table above is the sole index of documentation purposes.
Do not repeat that index in another file.

Do not use repository Markdown as a parallel issue backlog. Record only durable
contracts, current implementation state, and an approved implementation
sequence. Update every affected owner document in the same change when a
public path, dependency boundary, artifact contract, or module responsibility
changes.
