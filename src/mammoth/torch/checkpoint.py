"""Registered-state checkpoints with bounded asynchronous atomic publication.

The trainer registers framework objects here. Consuming projects retain
responsibility for deciding whether their model and checkpoint states match.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import torch

from mammoth.core.artifacts import atomic_publish

CHECKPOINT_SCHEMA_VERSION = 1


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
        self._pending: deque[Future[Path]] = deque()
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
        self._discard_completed()
        while len(self._pending) >= self.max_pending:
            self._pending.popleft().result()
        snapshot = snapshot_to_cpu(payload)
        future = self._executor.submit(publish_torch_payload, Path(path), snapshot)
        self._pending.append(future)
        return future

    def flush(self) -> None:
        """Wait for and surface every pending publication result."""
        while self._pending:
            self._pending.popleft().result()

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
            self._pending.popleft().result()


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
