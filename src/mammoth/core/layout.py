"""Stable run-path resolution shared by every Mammoth infrastructure layer.

The execution, logging, monitoring, workflow, and trainer packages consume
``RunLayout``. Consuming projects provide the entry path and opaque run name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mammoth.core.identity import validate_execution_id, validate_group_id, validate_run_name


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Resolve the stable ``<entry>/<run-name>`` artifact contract."""

    entry: Path
    run_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry", Path(self.entry))
        object.__setattr__(self, "run_name", validate_run_name(self.run_name))

    @property
    def run_dir(self) -> Path:
        """Return the logical run directory without creating it."""
        return self.entry / self.run_name

    @property
    def manifest_path(self) -> Path:
        """Return the consuming-project manifest location."""
        return self.run_dir / "manifest.json"

    @property
    def checkpoints_dir(self) -> Path:
        """Return the project-owned checkpoint directory."""
        return self.run_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        """Return the Mammoth-owned operational log directory."""
        return self.run_dir / "logs"

    @property
    def executions_dir(self) -> Path:
        """Return the immutable execution-attempt container."""
        return self.logs_dir / "executions"

    @property
    def results_dir(self) -> Path:
        """Return the project-owned result directory."""
        return self.run_dir / "results"

    @property
    def visualizations_dir(self) -> Path:
        """Return the project-owned visualization directory."""
        return self.run_dir / "vis"

    def execution_dir(self, execution_id: str) -> Path:
        """Return one execution directory after validating its identity."""
        return self.executions_dir / validate_execution_id(execution_id)

    def prepare(self, *, project_directories: bool = True) -> RunLayout:
        """Create stable operational paths and optionally project-owned paths."""
        self.executions_dir.mkdir(parents=True, exist_ok=True)
        if project_directories:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class GroupLayout:
    """Resolve the stable ``<entry>/.mammoth/groups/<group-id>`` artifact contract.

    A group is an optional entry-level record the workflow executor publishes
    to name the runs it dispatched together. It sits between :class:`RunLayout`'s
    entry and each member run: ``entry`` is the same caller-supplied artifact
    root a :class:`RunLayout` resolves against, while ``group_id`` identifies
    one workflow invocation. Consuming projects and every other Mammoth layer
    never require this directory to exist; an entry with no ``.mammoth/``
    subtree remains fully valid.
    """

    entry: Path
    group_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry", Path(self.entry))
        object.__setattr__(self, "group_id", validate_group_id(self.group_id))

    @property
    def groups_root(self) -> Path:
        """Return the entry-level container for every published group."""
        return self.entry / ".mammoth" / "groups"

    @property
    def group_dir(self) -> Path:
        """Return this group's stable directory without creating it."""
        return self.groups_root / self.group_id

    @property
    def manifest_path(self) -> Path:
        """Return the immutable group manifest location."""
        return self.group_dir / "manifest.json"

    @property
    def events_path(self) -> Path:
        """Return the append-only group event stream location."""
        return self.group_dir / "events.jsonl"

    def prepare(self) -> GroupLayout:
        """Create the stable group directory without creating manifest or events."""
        self.group_dir.mkdir(parents=True, exist_ok=True)
        return self
