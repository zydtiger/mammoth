# Agent Rules

This file is the first document every agent should read before changing
Mammoth. Read `docs/CODEMAP.md` next for repository navigation, then open only
the files needed for the task.

## Core Principles

Before editing, verify whether the current implementation already satisfies the
request. Do not make cosmetic renames, drive-by refactors, or unrelated cleanup.
Preserve unrelated user changes in a dirty worktree.

Treat `docs/ARCHITECTURE.md` as the sole owner of package boundaries, dependency
direction, runtime concepts, artifact layout, logging roles, trainer scope, and
compatibility policy. Do not restate those contracts in another document or
implement behavior that contradicts them.

## File Ownership

Every current repository file has one primary purpose. Update the owning file
when its subject changes instead of duplicating the same contract elsewhere.

| File | Purpose | Update when |
| --- | --- | --- |
| `AGENTS.md` | Repository rules, dependency boundaries, validation policy, Git prefixes, and file ownership. | Agent workflow, repository conventions, validation, Git policy, or documentation ownership changes. |
| `README.md` | Concise user-facing project introduction and setup entry point. | Public package purpose, installation, stable CLI/API, or user-visible behavior changes. |
| `docs/ARCHITECTURE.md` | Durable package boundary, dependency direction, runtime model, logging roles, trainer scope, and compatibility policy. | Responsibilities move between layers or a stable architectural contract changes. |
| `docs/IMPLEMENTATION_PLAN.md` | Ordered delivery phases, acceptance conditions, migration sequence, and explicitly deferred capabilities. | A phase is started/completed, delivery order changes, or deferred scope is approved. |
| `docs/CODEMAP.md` | Map of implemented source paths, symbols, and import relationships. | Implemented files or symbols move, appear, or disappear, or their import relationships change. |
| `pyproject.toml` | Package identity, Python requirement, build backend, dependencies, optional extras, and tool configuration. | Packaging, dependencies, commands, build behavior, or development-tool policy changes. |
| `uv.lock` | Exact uv resolution generated from `pyproject.toml`. Never edit manually. | Regenerate with uv whenever dependency metadata changes. |
| `.python-version` | uv/Python development baseline. | The supported development interpreter changes deliberately. |
| `.gitignore` | Generated-file and local-environment exclusions. | A new reproducible build, cache, environment, or local artifact needs an exclusion. |
| `src/mammoth/__init__.py` | Lightweight root package metadata and intentionally small stable exports. | Package version or a truly root-level stable export changes. |
| `src/mammoth/py.typed` | PEP 561 marker declaring inline type information. | Keep present and empty while Mammoth ships typed source. |

Generated `.venv/`, `dist/`, caches, and build metadata are not source files.
Do not document or commit their generated contents.

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

Validate in proportion to the change. The current scaffold supports:

```bash
uv sync
uv lock --check
uv run python -m compileall -q src
uv build
```

For documentation-only changes, inspect the diff and run `git diff --check`.
When Ruff, mypy, and pytest are added to the development dependency group,
Python changes must also run their repository-wide configured commands. Do not
claim a check passed unless it was actually run.

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
