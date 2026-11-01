"""Queue job values, submission validation, and the shared error family.

A job is an opaque command plus the resources it asks for. Mammoth never parses
its argv or decides where its artifacts go; ``run_dir`` only links a job to
monitor views and keys measured-peak inheritance.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args

from mammoth.compat import DATACLASS_SLOTS

VISIBLE_DEVICES_VARIABLE = "CUDA_VISIBLE_DEVICES"

JobStatus = Literal[
    "pending",
    "launching",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
    "dependency_failed",
]
DependencyMode = Literal["ok", "any"]

JOB_STATUSES: frozenset[str] = frozenset(get_args(JobStatus))
ACTIVE_STATUSES: frozenset[str] = frozenset({"pending", "launching", "running"})
TERMINAL_STATUSES: frozenset[str] = JOB_STATUSES - ACTIVE_STATUSES

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?$", re.IGNORECASE)
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


class QueueError(RuntimeError):
    """Base class for every queue failure reported to an operator."""


class JobValidationError(QueueError, ValueError):
    """Raised when a submitted job or request is malformed."""


class JobNotFoundError(QueueError):
    """Raised when a job ID or group matches nothing."""


class JobStateError(QueueError):
    """Raised when a request does not apply to the job's current state."""


def parse_size(text: str) -> int:
    """Return bytes for a binary size such as ``28G`` or ``512MiB``."""
    match = _SIZE.match(text.strip())
    if match is None:
        raise JobValidationError(f"Invalid size: {text!r}")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()])


def format_size(value: Union[int, None]) -> str:
    """Return a short binary size, or ``-`` when no value exists."""
    if value is None:
        return "-"
    for unit, scale in (("T", 1024**4), ("G", 1024**3), ("M", 1024**2)):
        if value >= scale:
            return f"{value / scale:.1f}{unit}"
    return f"{value}B"


def parse_duration(text: str) -> int:
    """Return whole seconds for a duration such as ``90s``, ``5m``, or ``14d``."""
    match = _DURATION.match(text.strip())
    if match is None:
        raise JobValidationError(f"Invalid duration: {text!r}")
    seconds = int(float(match.group(1)) * _DURATION_UNITS[match.group(2).lower()])
    if seconds < 1:
        raise JobValidationError(f"Duration must be at least one second: {text!r}")
    return seconds


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Dependency:
    """One earlier job that must finish before a dependent may become ready."""

    job_id: str
    mode: DependencyMode = "ok"

    def __post_init__(self) -> None:
        if not _JOB_ID.match(self.job_id):
            raise JobValidationError(f"Invalid dependency job ID: {self.job_id!r}")
        if self.mode not in ("ok", "any"):
            raise JobValidationError(f"Dependency mode must be ok or any: {self.mode!r}")

    @classmethod
    def parse(cls, text: str) -> Dependency:
        """Parse ``<job-id>`` or ``<job-id>:ok|any``."""
        job_id, _, mode = text.partition(":")
        return cls(job_id, mode or "ok")  # type: ignore[arg-type]


@dataclass(frozen=True, **DATACLASS_SLOTS)
class JobSpec:
    """Everything a submitter states about one job; immutable after acceptance."""

    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str]
    env_files: tuple[str, ...] = ()
    gpus: int = 1
    gpu: tuple[int, ...] = ()
    vram: Union[int, None] = None
    ram: Union[int, None] = None
    like: Union[str, None] = None
    after: tuple[Dependency, ...] = ()
    group: Union[str, None] = None
    priority: int = 0
    timeout_seconds: Union[int, None] = None
    run_dir: Union[str, None] = None

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(item, str) and item for item in self.argv):
            raise JobValidationError("Job argv must be a non-empty list of non-empty strings")
        _require_absolute("cwd", self.cwd)
        for name, value in self.env.items():
            if not _ENV_NAME.match(name) or not isinstance(value, str):
                raise JobValidationError(f"Invalid environment variable: {name!r}")
        # A NUL cannot reach exec, so it must fail here rather than at launch.
        if any("\0" in item for item in (*self.argv, self.cwd, *self.env.values())):
            raise JobValidationError("Job argv, cwd, and env must not contain NUL")
        for path in self.env_files:
            _require_absolute("env file", path)
        if self.gpus < 0:
            raise JobValidationError("Job gpus must not be negative")
        if VISIBLE_DEVICES_VARIABLE in self.env and self.gpus > 0:
            raise JobValidationError(
                f"{VISIBLE_DEVICES_VARIABLE} is set by the scheduler; use --gpu to pin a GPU"
            )
        if self.gpu and len(self.gpu) != self.gpus:
            raise JobValidationError("Pinned GPU count must equal the job's gpus")
        if len(set(self.gpu)) != len(self.gpu) or any(index < 0 for index in self.gpu):
            raise JobValidationError("Pinned GPU indices must be unique and not negative")
        if self.vram is not None and (self.vram <= 0 or self.gpus == 0):
            raise JobValidationError("Job vram must be positive and requires a GPU")
        if self.ram is not None and self.ram <= 0:
            raise JobValidationError("Job ram must be positive")
        if self.group is not None and not _GROUP_NAME.match(self.group):
            raise JobValidationError(f"Invalid group name: {self.group!r}")
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise JobValidationError("Job timeout must be at least one second")
        if self.run_dir is not None:
            _require_absolute("run_dir", self.run_dir)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible form stored and sent over the API."""
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "env_files": list(self.env_files),
            "gpus": self.gpus,
            "gpu": list(self.gpu),
            "vram": self.vram,
            "ram": self.ram,
            "like": self.like,
            "after": [{"job_id": item.job_id, "mode": item.mode} for item in self.after],
            "group": self.group,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JobSpec:
        """Validate one JSON payload; ``gpus`` defaults to the pin count, else 1."""
        try:
            gpu = tuple(_integer(item, "gpu") for item in payload.get("gpu") or ())
            gpus = payload.get("gpus")
            return cls(
                argv=tuple(payload["argv"]),
                cwd=str(payload["cwd"]),
                env=dict(payload.get("env") or {}),
                env_files=tuple(payload.get("env_files") or ()),
                gpus=_integer(gpus, "gpus") if gpus is not None else len(gpu) or 1,
                gpu=gpu,
                vram=_optional_integer(payload.get("vram"), "vram"),
                ram=_optional_integer(payload.get("ram"), "ram"),
                like=payload.get("like"),
                after=tuple(
                    Dependency(item["job_id"], item.get("mode", "ok"))
                    for item in payload.get("after") or ()
                ),
                group=payload.get("group"),
                priority=_integer(payload.get("priority", 0), "priority"),
                timeout_seconds=_optional_integer(
                    payload.get("timeout_seconds"), "timeout_seconds"
                ),
                run_dir=payload.get("run_dir"),
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise JobValidationError(f"Malformed job: {error!r}") from error


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Job:
    """One durable queue record: the immutable spec plus scheduler-owned state."""

    job_id: str
    sequence: int
    spec: JobSpec
    status: JobStatus
    priority: int
    submitted_at: float
    gpu_indices: tuple[int, ...] = ()
    reserved_vram: Union[int, None] = None
    reserved_ram: Union[int, None] = None
    tx_id: Union[str, None] = None
    launching_at: Union[float, None] = None
    started_at: Union[float, None] = None
    finished_at: Union[float, None] = None
    exit_code: Union[int, None] = None
    label: Union[str, None] = None
    message: Union[str, None] = None
    cancel_requested: bool = False
    peak_vram: Union[int, None] = None
    peak_ram: Union[int, None] = None
    peak_partial: bool = False
    last_sampled_at: Union[float, None] = None

    @property
    def execution_name(self) -> str:
        """Return the stable executor name that finds this job after a crash."""
        return f"mammoth-{self.job_id}"

    @property
    def is_terminal(self) -> bool:
        """Return whether this job's outcome is final."""
        return self.status in TERMINAL_STATUSES


def check_local_paths(spec: JobSpec) -> None:
    """Refuse paths that cannot work on this host, before a job waits in line."""
    if not os.path.isdir(spec.cwd):
        raise JobValidationError(f"Job cwd is not a directory: {spec.cwd}")
    for path in spec.env_files:
        if not os.path.isfile(path):
            raise JobValidationError(f"Job env file does not exist: {path}")


def default_state_dir() -> Path:
    """Return the per-user directory that holds the service's durable state."""
    root = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(root) / "mammoth" / "queue"


def default_socket_path() -> Path:
    """Return the per-user API socket path shared by the service and its clients."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return Path(runtime) / "mammoth" / "queue.sock" if runtime else default_state_dir() / "sock"


def _require_absolute(name: str, value: Any) -> None:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise JobValidationError(f"Job {name} must be an absolute path: {value!r}")


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise JobValidationError(f"Job {name} must be an integer: {value!r}")
    return value


def _optional_integer(value: Any, name: str) -> Union[int, None]:
    return None if value is None else _integer(value, name)
