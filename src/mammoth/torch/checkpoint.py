"""Registered-state checkpoints with bounded asynchronous atomic publication.

The trainer registers framework objects here. Consuming projects retain
responsibility for deciding whether their model and checkpoint states match.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


@dataclass(frozen=True)
class CheckpointArtifact:
    """One opaque checkpoint artifact and its caller-owned serializer."""

    destination: Path
    writer: Callable[[Path], None]
    mode: int = 0o600
    preserve_permissions: bool = True


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

    @property
    def pending_count(self) -> int:
        """Return queued or running publications."""
        self._discard_completed()
        return len(self._pending)

    def publish(self, path: Path, payload: Mapping[str, Any]) -> Future[Path]:
        """Clone state to CPU and submit one bounded atomic publication."""
        if self._closed:
            raise RuntimeError("checkpoint publisher is closed")
        self._await_submission_slot()
        snapshot = snapshot_to_cpu(payload)
        future = self._executor.submit(publish_torch_payload, Path(path), snapshot)
        self._pending.append(future)
        return future

    def submit(self, plan: CheckpointPlan) -> Future[CheckpointPublication]:
        """Submit one caller-owned immutable ordered checkpoint plan."""
        if self._closed:
            raise RuntimeError("checkpoint publisher is closed")
        validated = validate_checkpoint_plan(plan)
        self._await_submission_slot()
        future = self._executor.submit(publish_checkpoint_plan, validated)
        self._pending.append(future)
        return future

    def flush(self) -> None:
        """Wait for and surface every pending publication result."""
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
        except (KeyboardInterrupt, SystemExit):
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
    if not plan.artifacts:
        raise ValueError("checkpoint plan must contain at least one artifact")
    root_resolved = checkpoint_root.resolve()

    destinations: set[Path] = set()
    normalized_artifacts: list[CheckpointArtifact] = []
    for artifact in plan.artifacts:
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
        if (
            isinstance(artifact.mode, bool)
            or not isinstance(artifact.mode, int)
            or not 0 <= artifact.mode <= 0o777
        ):
            raise ValueError("checkpoint artifact mode must be from 0o000 through 0o777")
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
    for retirement_path in plan.retire_after_commit:
        retirement = Path(retirement_path)
        resolved = resolve_confined_path(root_resolved, retirement, role="retirement")
        if resolved in destinations:
            raise ValueError(f"checkpoint retirement target is also published: {retirement}")
        if resolved in retirements:
            raise ValueError(f"duplicate checkpoint retirement target: {retirement}")
        retirements.add(resolved)
        normalized_retirements.append(resolved)

    return CheckpointPlan(
        checkpoint_root=root_resolved,
        artifacts=tuple(normalized_artifacts),
        retire_after_commit=tuple(normalized_retirements),
    )


def resolve_confined_path(root: Path, path: Path, *, role: str) -> Path:
    """Resolve one target without following its final path component."""
    resolved = path.parent.resolve() / path.name
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"checkpoint {role} target is outside checkpoint_root: {path}")
    return resolved


def publish_checkpoint_plan(plan: CheckpointPlan) -> CheckpointPublication:
    """Prepare all artifacts, commit them in order, then retire exact old paths."""
    validated = validate_checkpoint_plan(plan)
    validated.checkpoint_root.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedArtifact] = []
    committed_count = 0
    try:
        for artifact in validated.artifacts:
            directory_descriptor = open_confined_parent(
                validated.checkpoint_root,
                artifact.destination,
                create=True,
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
                validated.checkpoint_root,
                prepared_artifact.destination,
                role="publication",
            )
            published.append(publish_prepared_artifact(prepared_artifact))
            committed_count += 1

        retired: list[Path] = []
        for path in validated.retire_after_commit:
            if retire_confined_path(validated.checkpoint_root, path):
                retired.append(path)
        return CheckpointPublication(published=tuple(published), retired=tuple(retired))
    finally:
        for prepared_artifact in prepared[committed_count:]:
            discard_prepared_artifact(prepared_artifact)


def retire_confined_path(root: Path, path: Path) -> bool:
    """Unlink one exact retirement path through a no-follow root-relative traversal."""
    ensure_confined_path_unchanged(root, path, role="retirement")
    try:
        directory_descriptor = open_confined_parent(root, path, create=False)
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


def open_confined_parent(root: Path, path: Path, *, create: bool) -> int:
    """Open a target parent by walking from its checkpoint root without symlinks."""
    ensure_confined_path_unchanged(root, path, role="target")
    relative_parent = path.parent.relative_to(root)
    directory_descriptor = os.open(root, directory_open_flags())
    try:
        for component in relative_parent.parts:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, dir_fd=directory_descriptor)
            child_descriptor = os.open(
                component,
                directory_open_flags(),
                dir_fd=directory_descriptor,
            )
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
    registry.load_state_dict(state, strict=strict)


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
