"""Registered-state checkpoints with bounded asynchronous atomic publication.

The trainer registers framework objects here. Consuming projects retain
responsibility for deciding whether their model and checkpoint states match.
"""

from __future__ import annotations

import os
import stat
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol

import torch

from mammoth.core.artifacts import (
    PreparedArtifact,
    atomic_publish,
    directory_open_flags,
    discard_prepared_artifact,
    prepare_artifact_in_directory,
    publish_prepared_artifact,
    sync_directory_descriptor,
)

CHECKPOINT_SCHEMA_VERSION = 1
CheckpointReason = Literal["scheduled", "manual", "interrupted"]
CheckpointMode = Literal["all", "latest"]
CheckpointComponent = Literal[
    "model",
    "optimizer",
    "scheduler",
    "callbacks",
    "trainer",
    "stopped_early",
    "scaler",
    "project",
]
RestoreAction = Literal["restore", "reset"]
_CHECKPOINT_COMPONENTS = frozenset(
    {
        "model",
        "optimizer",
        "scheduler",
        "callbacks",
        "trainer",
        "stopped_early",
        "scaler",
        "project",
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointSavePolicy:
    """Select generic resumable retention and metric-best publication."""

    mode: CheckpointMode = "latest"
    save_best: bool = True
    every_epochs: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"all", "latest"}:
            raise ValueError("checkpoint mode must be 'all' or 'latest'")
        if not isinstance(self.save_best, bool):
            raise ValueError("checkpoint save_best must be a boolean")
        if (
            isinstance(self.every_epochs, bool)
            or not isinstance(self.every_epochs, int)
            or self.every_epochs < 1
        ):
            raise ValueError("checkpoint every_epochs must be a positive integer")


@dataclass(frozen=True, slots=True)
class RestoreOptions:
    """Select whether generic mutable training state is restored or reset."""

    optimizer: RestoreAction = "restore"
    scheduler: RestoreAction = "restore"
    callbacks: RestoreAction = "restore"
    stopped_early: RestoreAction = "restore"

    def __post_init__(self) -> None:
        for name in ("optimizer", "scheduler", "callbacks", "stopped_early"):
            if getattr(self, name) not in {"restore", "reset"}:
                raise ValueError(f"{name} restore action must be 'restore' or 'reset'")


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Project-neutral summary produced before checkpoint state is applied."""

    available_components: frozenset[CheckpointComponent]
    restore_options: RestoreOptions = RestoreOptions()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.available_components, frozenset):
            raise TypeError("available_components must be a frozenset")
        invalid_components = self.available_components.difference(_CHECKPOINT_COMPONENTS)
        if invalid_components:
            raise ValueError(f"unsupported checkpoint components: {invalid_components}")
        if not isinstance(self.restore_options, RestoreOptions):
            raise TypeError("restore_options must be RestoreOptions")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("checkpoint inspection metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class TrainerCheckpointRestore:
    """Normalized project checkpoint state and Mammoth's applied-state report."""

    epoch: int
    optimizer_step: int | None = None
    global_step: int | None = None
    stopped_early: bool = False
    optimizer_state_dict: Mapping[str, Any] | None = field(default=None, repr=False)
    scheduler_state_dict: Mapping[str, Any] | None = field(default=None, repr=False)
    callback_state_dicts: Mapping[int, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    restored_components: frozenset[CheckpointComponent] = frozenset()
    reset_components: frozenset[CheckpointComponent] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < -1:
            raise ValueError("restored epoch must be an integer >= -1")
        for name in ("optimizer_step", "global_step"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"restored {name} must be a non-negative integer or None")
        if not isinstance(self.stopped_early, bool):
            raise ValueError("restored stopped_early must be a boolean")
        for name in ("optimizer_state_dict", "scheduler_state_dict"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
        if not isinstance(self.callback_state_dicts, Mapping) or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(state, Mapping)
            for index, state in self.callback_state_dicts.items()
        ):
            raise TypeError(
                "callback_state_dicts must map non-negative callback indices to mappings"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("checkpoint restore metadata must be a mapping")
        if not isinstance(self.restored_components, frozenset) or not isinstance(
            self.reset_components,
            frozenset,
        ):
            raise TypeError("restored and reset components must be frozensets")
        invalid_components = (
            self.restored_components | self.reset_components
        ).difference(_CHECKPOINT_COMPONENTS)
        if invalid_components:
            raise ValueError(f"unsupported checkpoint components: {invalid_components}")
        overlap = self.restored_components & self.reset_components
        if overlap:
            raise ValueError(f"checkpoint components cannot be restored and reset: {overlap}")


@dataclass(frozen=True)
class CheckpointArtifact:
    """One opaque checkpoint artifact and its caller-owned serializer."""

    destination: Path
    writer: Callable[[Path], object]
    mode: int | None = 0o600
    preserve_permissions: bool = True


@dataclass(frozen=True)
class TrainerCheckpointWriters:
    """Project serializers closed over one immutable checkpoint snapshot."""

    resumable: Callable[[Path], object]
    best: Callable[[Path], object] | None = None

    def __post_init__(self) -> None:
        if not callable(self.resumable):
            raise TypeError("resumable checkpoint writer must be callable")
        if self.best is not None and not callable(self.best):
            raise TypeError("best checkpoint writer must be callable or None")


@dataclass(frozen=True)
class CheckpointPlan:
    """Ordered local checkpoint publication with exact post-commit retirement."""

    checkpoint_root: Path
    artifacts: tuple[CheckpointArtifact, ...]
    retire_after_commit: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CheckpointPublication:
    """Paths committed and retired by one completed checkpoint plan."""

    published: tuple[Path, ...]
    retired: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class TrainerCheckpointContext:
    """Completed-epoch inputs supplied to a project checkpoint policy."""

    epoch: int
    global_step: int
    optimizer_step: int
    stopped_early: bool
    training_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float] | None
    reason: CheckpointReason = "scheduled"
    restore: TrainerCheckpointRestore | None = None


class TrainerCheckpointPolicy(Protocol):
    """Retain project checkpoint meaning behind Mammoth trainer mechanics."""

    def inspect(
        self,
        path: Path,
    ) -> CheckpointInspection:
        """Inspect project metadata and recommend generic restore actions."""
        ...

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        """Restore project objects and return generic trainer coordinates."""
        ...

    def capture(self, context: TrainerCheckpointContext) -> TrainerCheckpointWriters:
        """Capture immutable project state and return its two serializers."""
        ...


@dataclass(frozen=True)
class _AnchoredCheckpointPlan:
    """Normalized plan bound to the submitted checkpoint-root identity."""

    checkpoint_root: Path
    root_identity: tuple[int, int]
    artifacts: tuple[CheckpointArtifact, ...]
    retire_after_commit: tuple[Path, ...]


class Stateful(Protocol):
    """Minimal registered checkpoint object contract."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return serializable state."""

    def load_state_dict(self, state: Mapping[str, Any]) -> Any:
        """Restore serialized state."""


class StateRegistry:
    """Name and restore generic stateful objects without project knowledge."""

    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}

    def register(self, name: str, value: Any) -> None:
        """Register one unique non-empty checkpoint state name."""
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint state names must be non-empty strings")
        if name in self._objects:
            raise ValueError(f"checkpoint state {name!r} is already registered")
        if not callable(getattr(value, "state_dict", None)) or not callable(
            getattr(value, "load_state_dict", None)
        ):
            raise TypeError(f"checkpoint state {name!r} must implement state_dict/load_state_dict")
        self._objects[name] = value

    def state_dict(self) -> dict[str, Mapping[str, Any]]:
        """Snapshot every registered object by stable name."""
        return {name: value.state_dict() for name, value in self._objects.items()}

    @property
    def names(self) -> frozenset[str]:
        """Return the registered state names used by checkpoint selection."""
        return frozenset(self._objects)

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore registered objects and optionally reject missing or extra names."""
        missing = sorted(set(self._objects).difference(state))
        unexpected = sorted(set(state).difference(self._objects))
        if strict and (missing or unexpected):
            raise ValueError(
                f"checkpoint state mismatch; missing={missing}, unexpected={unexpected}"
            )
        for name, value in self._objects.items():
            if name not in state:
                continue
            item = state[name]
            if not isinstance(item, Mapping):
                raise ValueError(f"checkpoint state {name!r} must be a mapping")
            value.load_state_dict(item)


class AsyncCheckpointPublisher:
    """Publish cloned CPU state on one worker with a bounded pending queue."""

    def __init__(self, *, max_pending: int = 1) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mammoth-checkpoint")
        self._pending: deque[Future[Any]] = deque()
        self._closed = False
        self._lock = RLock()

    @property
    def pending_count(self) -> int:
        """Return queued or running publications."""
        with self._lock:
            self._discard_completed()
            return len(self._pending)

    def wait_for_submission_slot(self) -> None:
        """Apply queue backpressure before a caller captures its next snapshot."""
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint publisher is closed")
            self._await_submission_slot()

    def publish(self, path: Path, payload: Mapping[str, Any]) -> Future[Path]:
        """Clone state to CPU and submit one bounded atomic publication."""
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint publisher is closed")
            self._await_submission_slot()
            snapshot = snapshot_to_cpu(payload)
            future = self._executor.submit(publish_torch_payload, Path(path), snapshot)
            self._pending.append(future)
            return future

    def submit(self, plan: CheckpointPlan) -> Future[CheckpointPublication]:
        """Submit one caller-owned immutable ordered checkpoint plan."""
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint publisher is closed")
            validated = validate_checkpoint_plan(plan)
            anchored = anchor_checkpoint_plan(validated)
            self._await_submission_slot()
            future = self._executor.submit(publish_anchored_checkpoint_plan, anchored)
            self._pending.append(future)
            return future

    def flush(self) -> None:
        """Wait for and surface every pending publication result."""
        with self._lock:
            first_error: BaseException | None = None
            while self._pending:
                try:
                    self._resolve_oldest()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error

    def close(self) -> None:
        """Flush and shut down the worker exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.flush()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> AsyncCheckpointPublisher:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _discard_completed(self) -> None:
        while self._pending and self._pending[0].done():
            self._resolve_oldest()

    def _await_submission_slot(self) -> None:
        self._discard_completed()
        while len(self._pending) >= self.max_pending:
            self._resolve_oldest()

    def _resolve_oldest(self) -> None:
        pending = self._pending[0]
        try:
            pending.result()
        except (KeyboardInterrupt, SystemExit) as error:
            if pending.done() and pending.exception() is error:
                self._pending.popleft()
            raise
        except BaseException:
            self._pending.popleft()
            raise
        else:
            self._pending.popleft()


def validate_checkpoint_plan(plan: CheckpointPlan) -> CheckpointPlan:
    """Normalize and validate one local publication plan before worker submission."""
    if not isinstance(plan, CheckpointPlan):
        raise TypeError("checkpoint plan must be a CheckpointPlan")
    checkpoint_root = Path(plan.checkpoint_root)
    artifacts = tuple(plan.artifacts)
    if not artifacts:
        raise ValueError("checkpoint plan must contain at least one artifact")
    root_resolved = checkpoint_root.resolve()

    destinations: set[Path] = set()
    normalized_artifacts: list[CheckpointArtifact] = []
    for artifact in artifacts:
        if not isinstance(artifact, CheckpointArtifact):
            raise TypeError("checkpoint artifacts must be CheckpointArtifact values")
        destination = Path(artifact.destination)
        resolved = resolve_confined_path(root_resolved, destination, role="publication")
        if destination.is_symlink():
            raise ValueError(f"checkpoint publication target must not be a symlink: {destination}")
        if resolved in destinations:
            raise ValueError(f"duplicate checkpoint publication target: {destination}")
        if not callable(artifact.writer):
            raise TypeError("checkpoint artifact writer must be callable")
        if artifact.mode is not None and (
            isinstance(artifact.mode, bool)
            or not isinstance(artifact.mode, int)
            or not 0 <= artifact.mode <= 0o777
        ):
            raise ValueError(
                "checkpoint artifact mode must be None or from 0o000 through 0o777"
            )
        destinations.add(resolved)
        normalized_artifacts.append(
            CheckpointArtifact(
                destination=resolved,
                writer=artifact.writer,
                mode=artifact.mode,
                preserve_permissions=artifact.preserve_permissions,
            )
        )

    retirements: set[Path] = set()
    normalized_retirements: list[Path] = []
    for retirement_path in tuple(plan.retire_after_commit):
        retirement = Path(retirement_path)
        resolved = resolve_confined_path(root_resolved, retirement, role="retirement")
        if resolved in destinations:
            raise ValueError(f"checkpoint retirement target is also published: {retirement}")
        if resolved in retirements:
            raise ValueError(f"duplicate checkpoint retirement target: {retirement}")
        try:
            retirement_mode = os.stat(
                retirement,
                follow_symlinks=False,
            ).st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(retirement_mode):
                raise ValueError(
                    f"checkpoint retirement target must be a regular file: {retirement}"
                )
        retirements.add(resolved)
        normalized_retirements.append(resolved)

    return CheckpointPlan(
        checkpoint_root=root_resolved,
        artifacts=tuple(normalized_artifacts),
        retire_after_commit=tuple(normalized_retirements),
    )


def build_trainer_checkpoint_plan(
    checkpoint_root: Path,
    *,
    epoch: int,
    save_policy: CheckpointSavePolicy,
    writers: TrainerCheckpointWriters,
    save_resumable: bool,
    save_best: bool,
) -> CheckpointPlan:
    """Build Mammoth-selected names, ordering, and retirement for one capture."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < -1:
        raise ValueError("checkpoint epoch must be an integer >= -1")
    if not isinstance(save_policy, CheckpointSavePolicy):
        raise TypeError("save_policy must be CheckpointSavePolicy")
    if not isinstance(writers, TrainerCheckpointWriters):
        raise TypeError("writers must be TrainerCheckpointWriters")
    if not isinstance(save_resumable, bool) or not isinstance(save_best, bool):
        raise TypeError("checkpoint selection flags must be booleans")
    if not save_resumable and not save_best:
        raise ValueError("checkpoint selection must include a resumable or best artifact")
    if save_best and writers.best is None:
        raise ValueError("save_best requires a project best-checkpoint writer")

    root = Path(checkpoint_root)
    artifacts: list[CheckpointArtifact] = []
    if save_best:
        assert writers.best is not None
        artifacts.append(
            CheckpointArtifact(
                destination=root / "best.safetensors",
                writer=writers.best,
            )
        )

    retirements: tuple[Path, ...] = ()
    if save_resumable:
        prefix = "epoch_" if save_policy.mode == "all" else "latest_epoch_"
        destination = root / f"{prefix}{epoch}.pt"
        artifacts.append(
            CheckpointArtifact(
                destination=destination,
                writer=writers.resumable,
            )
        )
        if save_policy.mode == "latest":
            retirements = tuple(
                path
                for path in sorted(root.glob("latest_epoch_*.pt"))
                if path != destination and _latest_checkpoint_epoch(path) is not None
            )

    return CheckpointPlan(
        checkpoint_root=root,
        artifacts=tuple(artifacts),
        retire_after_commit=retirements,
    )


def _latest_checkpoint_epoch(path: Path) -> int | None:
    """Parse one Mammoth-owned latest-checkpoint filename."""
    if path.suffix != ".pt" or not path.stem.startswith("latest_epoch_"):
        return None
    try:
        epoch = int(path.stem.removeprefix("latest_epoch_"))
    except ValueError:
        return None
    return epoch if epoch >= -1 else None


def resolve_confined_path(root: Path, path: Path, *, role: str) -> Path:
    """Resolve one target without following its final path component."""
    resolved = path.parent.resolve() / path.name
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"checkpoint {role} target is outside checkpoint_root: {path}")
    return resolved


def publish_checkpoint_plan(plan: CheckpointPlan) -> CheckpointPublication:
    """Prepare all artifacts, commit them in order, then retire exact old paths."""
    validated = validate_checkpoint_plan(plan)
    return publish_anchored_checkpoint_plan(anchor_checkpoint_plan(validated))


def anchor_checkpoint_plan(plan: CheckpointPlan) -> _AnchoredCheckpointPlan:
    """Bind one normalized plan to a durably created checkpoint root."""
    require_descriptor_relative_filesystem()
    root_identity = ensure_checkpoint_root(plan.checkpoint_root)
    return _AnchoredCheckpointPlan(
        checkpoint_root=plan.checkpoint_root,
        root_identity=root_identity,
        artifacts=plan.artifacts,
        retire_after_commit=plan.retire_after_commit,
    )


def publish_anchored_checkpoint_plan(plan: _AnchoredCheckpointPlan) -> CheckpointPublication:
    """Publish one plan without rebasing its submitted root identity."""
    prepared: list[PreparedArtifact] = []
    committed_count = 0
    try:
        for artifact in plan.artifacts:
            directory_descriptor = open_confined_parent(
                plan.checkpoint_root,
                artifact.destination,
                create=True,
                root_identity=plan.root_identity,
            )
            prepared.append(
                prepare_artifact_in_directory(
                    artifact.destination,
                    artifact.writer,
                    directory_descriptor=directory_descriptor,
                    mode=artifact.mode,
                    preserve_permissions=artifact.preserve_permissions,
                )
            )
        published: list[Path] = []
        for prepared_artifact in prepared:
            ensure_confined_path_unchanged(
                plan.checkpoint_root,
                prepared_artifact.destination,
                role="publication",
            )
            published.append(publish_prepared_artifact(prepared_artifact))
            committed_count += 1

        retired: list[Path] = []
        for path in plan.retire_after_commit:
            if retire_confined_path(
                plan.checkpoint_root,
                path,
                root_identity=plan.root_identity,
            ):
                retired.append(path)
        return CheckpointPublication(published=tuple(published), retired=tuple(retired))
    finally:
        for prepared_artifact in prepared[committed_count:]:
            discard_prepared_artifact(prepared_artifact)


def retire_confined_path(
    root: Path,
    path: Path,
    *,
    root_identity: tuple[int, int],
) -> bool:
    """Unlink one exact retirement path through a no-follow root-relative traversal."""
    ensure_confined_path_unchanged(root, path, role="retirement")
    try:
        directory_descriptor = open_confined_parent(
            root,
            path,
            create=False,
            root_identity=root_identity,
        )
    except FileNotFoundError:
        return False
    try:
        try:
            os.unlink(path.name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            return False
        sync_directory_descriptor(directory_descriptor)
        return True
    finally:
        with suppress(OSError):
            os.close(directory_descriptor)


def ensure_confined_path_unchanged(root: Path, path: Path, *, role: str) -> None:
    """Reject a normalized plan path whose parent now resolves elsewhere."""
    normalized = resolve_confined_path(root, path, role=role)
    if normalized != path:
        raise RuntimeError(
            f"checkpoint {role} parent changed before filesystem effect: {path.parent}"
        )


def open_confined_parent(
    root: Path,
    path: Path,
    *,
    create: bool,
    root_identity: tuple[int, int],
) -> int:
    """Open a target parent by walking from its checkpoint root without symlinks."""
    ensure_confined_path_unchanged(root, path, role="target")
    relative_parent = path.parent.relative_to(root)
    directory_descriptor = os.open(root, directory_open_flags())
    try:
        opened_root_stat = os.fstat(directory_descriptor)
        opened_root_identity = (opened_root_stat.st_dev, opened_root_stat.st_ino)
        if opened_root_identity != root_identity:
            raise RuntimeError(f"checkpoint root changed before filesystem effect: {root}")
        for component in relative_parent.parts:
            created = False
            if create:
                try:
                    os.mkdir(component, dir_fd=directory_descriptor)
                except FileExistsError:
                    pass
                else:
                    created = True
            child_descriptor = os.open(
                component,
                directory_open_flags(),
                dir_fd=directory_descriptor,
            )
            if created:
                try:
                    sync_directory_descriptor(directory_descriptor)
                except BaseException:
                    with suppress(OSError):
                        os.close(child_descriptor)
                    raise
            parent_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            try:
                os.close(parent_descriptor)
            except BaseException:
                with suppress(OSError):
                    os.close(parent_descriptor)
                raise
        return directory_descriptor
    except BaseException:
        with suppress(OSError):
            os.close(directory_descriptor)
        raise


def ensure_checkpoint_root(root: Path) -> tuple[int, int]:
    """Create a resolved checkpoint root and durably link each new directory."""
    missing_components: list[str] = []
    existing_parent = root
    while True:
        try:
            directory_descriptor = os.open(existing_parent, directory_open_flags())
        except FileNotFoundError:
            missing_components.append(existing_parent.name)
            existing_parent = existing_parent.parent
        else:
            break

    try:
        for component in reversed(missing_components):
            created = False
            try:
                os.mkdir(component, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            else:
                created = True
            child_descriptor = os.open(
                component,
                directory_open_flags(),
                dir_fd=directory_descriptor,
            )
            if created:
                try:
                    sync_directory_descriptor(directory_descriptor)
                except BaseException:
                    with suppress(OSError):
                        os.close(child_descriptor)
                    raise
            parent_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            try:
                os.close(parent_descriptor)
            except BaseException:
                with suppress(OSError):
                    os.close(parent_descriptor)
                raise
        root_stat = os.fstat(directory_descriptor)
        return root_stat.st_dev, root_stat.st_ino
    finally:
        with suppress(OSError):
            os.close(directory_descriptor)


def require_descriptor_relative_filesystem() -> None:
    """Reject platforms without the operations required for confined durability."""
    required = (os.mkdir, os.open, os.rename, os.stat, os.unlink)
    unsupported = [
        operation.__name__
        for operation in required
        if operation not in os.supports_dir_fd
    ]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            "ordered checkpoint publication requires POSIX descriptor-relative "
            f"filesystem operations; unavailable: {names}"
        )


def checkpoint_payload(registry: StateRegistry) -> dict[str, Any]:
    """Build one versioned registered-state checkpoint payload."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": registry.state_dict(),
    }


def restore_checkpoint(
    path: Path,
    registry: StateRegistry,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load one Mammoth checkpoint into an existing registry."""
    state = load_checkpoint_state(path, map_location=map_location)
    registry.load_state_dict(state, strict=strict)


def load_checkpoint_state(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load and validate one Mammoth registered-state checkpoint payload."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Mammoth checkpoint must contain a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Mammoth checkpoint schema: {payload.get('schema_version')!r}"
        )
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Mammoth checkpoint state must be a mapping")
    return state


def snapshot_to_cpu(value: Any) -> Any:
    """Clone tensors to CPU so worker serialization cannot race live mutation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [snapshot_to_cpu(item) for item in value]
    return value


def publish_torch_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    """Serialize a snapshot through the core same-directory atomic publisher."""
    return atomic_publish(path, lambda temporary: torch.save(payload, temporary))
