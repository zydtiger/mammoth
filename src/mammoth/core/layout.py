"""Stable run-path resolution shared by every Mammoth infrastructure layer.

The execution, logging, monitoring, workflow, and trainer packages consume
``RunLayout``. Consuming projects provide the entry path and opaque run name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mammoth.core.identity import validate_execution_id, validate_run_name


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
