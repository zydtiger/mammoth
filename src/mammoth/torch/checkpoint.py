"""Registered-state checkpoints with bounded asynchronous atomic publication.

The trainer registers framework objects here. Consuming projects retain
responsibility for deciding whether their model and checkpoint states match.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Condition, Lock, RLock, Thread
from typing import Any, Literal, Protocol, cast

import torch

from mammoth.core.artifacts import (
    PreparedArtifact,
    atomic_publish,
    directory_open_flags,
    discard_prepared_artifact,
    inspect_artifact_descriptor,
    prepare_artifact_in_directory,
    publish_prepared_artifact,
    sync_directory_descriptor,
)
from mammoth.core.pipeline import (
    BackgroundPipelineError,
    BackgroundPipelineSubmission,
    BoundedBackgroundPipeline,
)

CHECKPOINT_SCHEMA_VERSION = 1
CheckpointReason = Literal["scheduled", "manual", "interrupted"]
CheckpointMode = Literal["all", "latest"]
CheckpointCaptureMode = Literal["auto", "cpu"]
CheckpointRole = Literal["epoch", "latest", "best"]
ResumableCheckpointRole = Literal["epoch", "latest"]
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

_LOGGER = logging.getLogger(__name__)
_RESUMABLE_CHECKPOINT_FILENAME = re.compile(
    r"(?:(?P<latest>latest_epoch_)|(?P<epoch>epoch_))(?P<number>-1|0|[1-9][0-9]*)\.pt\Z"
)


class _CudaEvent(Protocol):
    """Minimal CUDA-event behavior required by the checkpoint worker."""

    def record(self, stream: Any = None) -> None:
        """Record work completion on one CUDA stream."""

    def synchronize(self) -> None:
        """Wait until the recorded CUDA work completes."""


@dataclass(frozen=True, slots=True)
class _CheckpointCaptureDecision:
    """Describe the capture path selected for one generic checkpoint."""

    path: Literal["cpu", "gpu"]
    reason: str
    snapshot_bytes: int = 0
    available_bytes: int | None = None
    required_bytes: int | None = None


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
        invalid_components = (self.restored_components | self.reset_components).difference(
            _CHECKPOINT_COMPONENTS
        )
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
    role: CheckpointRole = "epoch"
    epoch: int = -1


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
    """Artifact receipts committed and paths retired by one completed plan."""

    published: tuple[PublishedCheckpoint, ...]
    retired: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PublishedCheckpoint:
    """Identity and exact-byte provenance for one committed checkpoint artifact."""

    path: Path
    role: CheckpointRole
    epoch: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"epoch", "latest", "best"}:
            raise ValueError("checkpoint role must be 'epoch', 'latest', or 'best'")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < -1:
            raise ValueError("published checkpoint epoch must be an integer >= -1")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("published checkpoint size_bytes must be non-negative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("published checkpoint sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ResumableCheckpointCandidate:
    """One filename-derived standard trainer checkpoint candidate.

    Discovery is intentionally ephemeral and syntactic: this value does not
    claim that its path is a regular file, symlink-safe, readable, or compatible
    with any model or trainer state.
    """

    path: Path
    role: ResumableCheckpointRole
    epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("checkpoint candidate path must be a Path")
        if self.role not in {"epoch", "latest"}:
            raise ValueError("checkpoint candidate role must be 'epoch' or 'latest'")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < -1:
            raise ValueError("checkpoint candidate epoch must be an integer >= -1")


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


@dataclass(frozen=True, slots=True)
class _CheckpointPublicationTask[ResultT]:
    """Bridge one public checkpoint future to generic background execution."""

    operation: Callable[[], ResultT]
    future: _CheckpointPublicationFuture[ResultT]


class _CheckpointPublicationFuture[ResultT](Future[ResultT]):
    """Finalize results before dispatching user callbacks through the pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self._callback_lock = RLock()
        self._user_callbacks: list[Callable[[Future[ResultT]], object]] = []
        self._callbacks_finalized = False

    def add_done_callback(
        self,
        fn: Callable[[Future[ResultT]], object],
    ) -> None:
        if not callable(fn):
            raise TypeError("done callback must be callable")
        with self._callback_lock:
            if not self._callbacks_finalized:
                self._user_callbacks.append(fn)
                return
        fn(self)

    def complete_result(
        self,
        result: ResultT,
    ) -> None:
        with self._callback_lock:
            with suppress(BaseException):
                super().set_result(result)
            callbacks = self._finalize_callbacks_locked()
        self._dispatch_callbacks(callbacks)

    def complete_exception(
        self,
        error: BaseException,
    ) -> None:
        with self._callback_lock:
            with suppress(BaseException):
                super().set_exception(error)
            callbacks = self._finalize_callbacks_locked()
        self._dispatch_callbacks(callbacks)

    def _finalize_callbacks_locked(
        self,
    ) -> tuple[Callable[[Future[ResultT]], object], ...]:
        self._callbacks_finalized = True
        callbacks = tuple(self._user_callbacks)
        self._user_callbacks.clear()
        return callbacks

    def _dispatch_callbacks(
        self,
        callbacks: tuple[Callable[[Future[ResultT]], object], ...],
    ) -> None:
        if callbacks:
            Thread(
                target=self._invoke_user_callbacks,
                args=(callbacks,),
                name="mammoth-checkpoint-future-callbacks",
                daemon=True,
            ).start()

    def _invoke_user_callbacks(
        self,
        callbacks: tuple[Callable[[Future[ResultT]], object], ...],
    ) -> None:
        for callback in callbacks:
            with suppress(BaseException):
                callback(self)


def _run_checkpoint_publication[ResultT](
    task: _CheckpointPublicationTask[ResultT],
) -> ResultT:
    """Run one checkpoint operation before generic state completes its future."""
    return task.operation()


class AsyncCheckpointPublisher:
    """Adapt checkpoint publication to Mammoth's bounded background pipeline."""

    def __init__(
        self,
        *,
        max_pending: int = 1,
        capture_mode: CheckpointCaptureMode = "auto",
        cuda_headroom_bytes: int = 0,
    ) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if capture_mode not in {"auto", "cpu"}:
            raise ValueError("checkpoint capture mode must be 'auto' or 'cpu'")
        if (
            isinstance(cuda_headroom_bytes, bool)
            or not isinstance(cuda_headroom_bytes, int)
            or cuda_headroom_bytes < 0
        ):
            raise ValueError("cuda_headroom_bytes must be a non-negative integer")
        self.max_pending = max_pending
        self.capture_mode = capture_mode
        self.cuda_headroom_bytes = cuda_headroom_bytes
        self._pipeline: BoundedBackgroundPipeline[
            _CheckpointPublicationTask[Any],
            Any,
        ] = BoundedBackgroundPipeline(
            _run_checkpoint_publication,
            max_pending=max_pending,
            thread_name_prefix="mammoth-checkpoint",
        )
        self._deferred_failures: deque[
            tuple[
                BackgroundPipelineSubmission[_CheckpointPublicationTask[Any], Any],
                BaseException,
            ]
        ] = deque()
        self._recorded_failure_submissions: set[
            BackgroundPipelineSubmission[_CheckpointPublicationTask[Any], Any]
        ] = set()
        self._deferred_interrupt: KeyboardInterrupt | SystemExit | None = None
        self._closing = False
        self._closed = False
        self._gpu_snapshot_pending = False
        self._gpu_snapshot_lock = Lock()
        self._lock = RLock()
        self._lifecycle_condition = Condition(self._lock)

    @property
    def pending_count(self) -> int:
        """Return queued or running publications."""
        with self._lock:
            if self._closing:
                raise RuntimeError("checkpoint publisher is closing")
            self._raise_deferred_failure()
            self._consume_completed()
            try:
                return self._pipeline.pending_count
            except BackgroundPipelineError as error:
                self._preserve_and_acknowledge_failure(error)
                self._raise_deferred_failure()
                raise error.cause from None

    def wait_for_submission_slot(self) -> None:
        """Apply queue backpressure before a caller captures its next snapshot."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("checkpoint publisher is closed")
            self._await_submission_slot()

    def publish(self, path: Path, payload: Mapping[str, Any]) -> Future[Path]:
        """Capture state and submit one bounded atomic publication."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("checkpoint publisher is closed")
            self._await_submission_slot()
            return self._publish_mapping(
                payload,
                lambda snapshot: partial(publish_torch_payload, Path(path), snapshot),
            )

    def publish_with_receipt(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        role: CheckpointRole,
        epoch: int,
    ) -> Future[CheckpointPublication]:
        """Publish one generic payload and report its exact committed bytes."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("checkpoint publisher is closed")
            self._await_submission_slot()
            return self._publish_mapping(
                payload,
                lambda snapshot: partial(
                    publish_torch_payload_with_receipt,
                    Path(path),
                    snapshot,
                    role=role,
                    epoch=epoch,
                ),
            )

    def submit(self, plan: CheckpointPlan) -> Future[CheckpointPublication]:
        """Submit one caller-owned immutable ordered checkpoint plan."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("checkpoint publisher is closed")
            validated = validate_checkpoint_plan(plan)
            anchored = anchor_checkpoint_plan(validated)
            self._await_submission_slot()
            return self._submit_operation(partial(publish_anchored_checkpoint_plan, anchored))

    def flush(self) -> None:
        """Wait for and surface every pending publication result."""
        with self._lifecycle_condition:
            while self._closing and not self._closed:
                self._lifecycle_condition.wait()
            if self._closed:
                return
            self._raise_deferred_interrupt()
            self._flush_pipeline(close=False)

    def close(self) -> None:
        """Flush and shut down the worker exactly once."""
        with self._lifecycle_condition:
            while self._closing and not self._closed:
                self._lifecycle_condition.wait()
            if self._closed:
                return
            self._closing = True
            deferred_interrupt = self._remember_deferred_interrupt()
        try:
            cleanup_error = self._flush_pipeline(close=True, raise_error=False)
        except BaseException:
            with self._lifecycle_condition:
                self._closing = False
                self._lifecycle_condition.notify_all()
            raise
        with self._lifecycle_condition:
            if deferred_interrupt is not None:
                try:
                    if cleanup_error is not None:
                        deferred_interrupt.add_note(
                            "Checkpoint cleanup failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        for note in getattr(cleanup_error, "__notes__", ()):
                            deferred_interrupt.add_note(f"Checkpoint cleanup detail: {note}")
                    self._deferred_interrupt = None
                    self._closing = False
                    self._closed = True
                    self._lifecycle_condition.notify_all()
                    raise deferred_interrupt
                except (KeyboardInterrupt, SystemExit) as delivery_interrupt:
                    if delivery_interrupt is deferred_interrupt:
                        raise
                    self._deferred_interrupt = deferred_interrupt
                    self._closing = False
                    self._closed = False
                    self._lifecycle_condition.notify_all()
                    raise
                except BaseException:
                    self._closing = False
                    self._lifecycle_condition.notify_all()
                    raise
            self._closing = False
            self._closed = True
            self._lifecycle_condition.notify_all()
            if cleanup_error is not None:
                raise cleanup_error

    def __enter__(self) -> AsyncCheckpointPublisher:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _await_submission_slot(self) -> None:
        self._raise_deferred_interrupt()
        self._raise_deferred_failure()
        try:
            self._pipeline.wait_for_submission_slot()
        except BackgroundPipelineError as error:
            self._preserve_and_acknowledge_failure(error)
            self._raise_deferred_failure()
            raise error.cause from None
        self._consume_completed()

    def _submit_operation[ResultT](
        self,
        operation: Callable[[], ResultT],
    ) -> Future[ResultT]:
        future = _CheckpointPublicationFuture[ResultT]()
        if not future.set_running_or_notify_cancel():
            raise RuntimeError("checkpoint publication future could not start")
        task = _CheckpointPublicationTask(operation, future)
        while True:
            try:
                self._pipeline.submit(
                    task,
                    on_done=lambda completed: self._complete_public_future(
                        future,
                        completed,
                    ),
                )
            except BackgroundPipelineError as error:
                self._preserve_and_acknowledge_failure(error)
                continue
            break
        return future

    def _publish_mapping[ResultT](
        self,
        payload: Mapping[str, Any],
        operation_for_snapshot: Callable[[Mapping[str, Any]], Callable[[], ResultT]],
    ) -> Future[ResultT]:
        decision = self._capture_decision(payload)
        _LOGGER.debug(
            "Checkpoint capture path=%s reason=%s snapshot_bytes=%d available_bytes=%s "
            "required_bytes=%s",
            decision.path,
            decision.reason,
            decision.snapshot_bytes,
            decision.available_bytes,
            decision.required_bytes,
        )
        if decision.path == "cpu":
            return self._submit_operation(operation_for_snapshot(snapshot_to_cpu(payload)))

        snapshot: Mapping[str, Any] | None = None
        try:
            snapshot = snapshot_to_gpu(payload)
            device = cuda_snapshot_device(snapshot)
            if device is None:
                raise RuntimeError("GPU checkpoint capture did not retain CUDA state")
            event_factory = cast(Callable[[], _CudaEvent], torch.cuda.Event)
            ready = event_factory()
            ready.record(torch.cuda.current_stream(device))
        except torch.OutOfMemoryError:
            snapshot = None
            self._record_gpu_fallback("gpu-snapshot-oom", decision)
            return self._submit_operation(operation_for_snapshot(snapshot_to_cpu(payload)))
        except BaseException:
            snapshot = None
            raise

        try:
            with self._gpu_snapshot_lock:
                self._gpu_snapshot_pending = True
            return self._submit_operation(
                self._gpu_transfer_operation(snapshot, ready, operation_for_snapshot)
            )
        except BaseException:
            with self._gpu_snapshot_lock:
                self._gpu_snapshot_pending = False
            raise

    def _capture_decision(self, payload: Mapping[str, Any]) -> _CheckpointCaptureDecision:
        if self.capture_mode == "cpu":
            return _CheckpointCaptureDecision("cpu", "configured-cpu-mode")
        with self._gpu_snapshot_lock:
            gpu_snapshot_pending = self._gpu_snapshot_pending
        if gpu_snapshot_pending:
            return _CheckpointCaptureDecision("cpu", "gpu-snapshot-pending")
        inspected = inspect_cuda_snapshot(payload)
        if inspected is None:
            return _CheckpointCaptureDecision("cpu", "unsupported-or-ambiguous-payload")
        snapshot_bytes, device = inspected
        if snapshot_bytes == 0:
            return _CheckpointCaptureDecision("cpu", "no-cuda-tensors")
        try:
            available_bytes, _ = torch.cuda.mem_get_info(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
        except RuntimeError:
            return _CheckpointCaptureDecision("cpu", "cuda-memory-query-failed", snapshot_bytes)
        headroom_bytes = max(
            self.cuda_headroom_bytes,
            snapshot_bytes,
            allocated_bytes // 2,
        )
        required_bytes = snapshot_bytes + headroom_bytes
        if available_bytes < required_bytes:
            return _CheckpointCaptureDecision(
                "cpu",
                "insufficient-cuda-headroom",
                snapshot_bytes,
                available_bytes,
                required_bytes,
            )
        return _CheckpointCaptureDecision(
            "gpu",
            "sufficient-cuda-headroom",
            snapshot_bytes,
            available_bytes,
            required_bytes,
        )

    def _record_gpu_fallback(
        self,
        reason: str,
        decision: _CheckpointCaptureDecision,
    ) -> None:
        _LOGGER.debug(
            "Checkpoint capture path=cpu reason=%s snapshot_bytes=%d available_bytes=%s "
            "required_bytes=%s",
            reason,
            decision.snapshot_bytes,
            decision.available_bytes,
            decision.required_bytes,
        )

    def _gpu_transfer_operation[ResultT](
        self,
        snapshot: Mapping[str, Any],
        ready: _CudaEvent,
        operation_for_snapshot: Callable[[Mapping[str, Any]], Callable[[], ResultT]],
    ) -> Callable[[], ResultT]:
        def operation() -> ResultT:
            nonlocal snapshot
            try:
                ready.synchronize()
                cpu_snapshot = snapshot_to_cpu(snapshot)
            finally:
                snapshot = {}
                with self._gpu_snapshot_lock:
                    self._gpu_snapshot_pending = False
            return operation_for_snapshot(cpu_snapshot)()

        return operation

    @staticmethod
    def _complete_public_future[ResultT](
        future: _CheckpointPublicationFuture[ResultT],
        submission: BackgroundPipelineSubmission[
            _CheckpointPublicationTask[ResultT],
            ResultT,
        ],
    ) -> None:
        try:
            result = submission.result()
        except BaseException as error:
            future.complete_exception(error)
            return
        future.complete_result(result)

    def _consume_completed(self) -> None:
        try:
            completed = self._pipeline.take_completed()
        except BackgroundPipelineError as error:
            self._preserve_and_acknowledge_failure(error)
            self._raise_deferred_failure()
            raise error.cause from None
        for result in completed:
            self._acknowledge(result.submission)

    def _flush_pipeline(
        self,
        *,
        close: bool,
        raise_error: bool = True,
    ) -> BaseException | None:
        error_entries = list(self._deferred_failures)
        while True:
            try:
                completed = self._pipeline.close() if close else self._pipeline.flush()
            except BackgroundPipelineError as error:
                if self._preserve_and_acknowledge_failure(error):
                    error_entries.append(self._deferred_failures[-1])
                continue
            for result in completed:
                self._acknowledge(result.submission)
            break
        for submission, _cause in error_entries:
            if not self._pipeline.owns(submission):
                self._recorded_failure_submissions.discard(submission)
        errors = [cause for _submission, cause in error_entries]
        if not errors:
            self._deferred_failures.clear()
            return None
        primary_index = next(
            (
                index
                for index, error in enumerate(errors)
                if isinstance(error, (KeyboardInterrupt, SystemExit))
            ),
            0,
        )
        first_error = errors[primary_index]
        for index, additional_error in enumerate(errors):
            if index == primary_index:
                continue
            try:
                first_error.add_note(
                    "Additional checkpoint publication failure: "
                    f"{type(additional_error).__name__}: {additional_error}"
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                continue
        try:
            self._deferred_failures.clear()
            if raise_error:
                raise first_error
            return first_error
        except (KeyboardInterrupt, SystemExit) as delivery_interrupt:
            if raise_error and delivery_interrupt is first_error:
                raise
            self._deferred_failures.extend(error_entries)
            raise

    def _acknowledge(
        self,
        submission: BackgroundPipelineSubmission[
            _CheckpointPublicationTask[Any],
            Any,
        ],
    ) -> None:
        submission.input.future.exception()
        self._pipeline.acknowledge(submission)

    def _raise_deferred_interrupt(self) -> None:
        deferred_interrupt = self._remember_deferred_interrupt()
        if deferred_interrupt is not None:
            self._deferred_interrupt = None
            raise deferred_interrupt

    def _remember_deferred_interrupt(self) -> KeyboardInterrupt | SystemExit | None:
        if self._deferred_interrupt is None:
            self._deferred_interrupt = self._pipeline.take_deferred_interrupt()
        return self._deferred_interrupt

    def _raise_deferred_failure(self) -> None:
        if self._deferred_failures:
            submission, cause = self._deferred_failures.popleft()
            if not self._pipeline.owns(submission):
                self._recorded_failure_submissions.discard(submission)
            raise cause

    def _preserve_and_acknowledge_failure(
        self,
        error: BackgroundPipelineError[_CheckpointPublicationTask[Any], Any],
    ) -> bool:
        submission = error.submission
        newly_recorded = submission not in self._recorded_failure_submissions
        if newly_recorded:
            self._recorded_failure_submissions.add(submission)
            self._deferred_failures.append((submission, error.cause))
        self._acknowledge(submission)
        self._recorded_failure_submissions.discard(submission)
        return newly_recorded


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
        if artifact.role not in {"epoch", "latest", "best"}:
            raise ValueError("checkpoint artifact role must be 'epoch', 'latest', or 'best'")
        if (
            isinstance(artifact.epoch, bool)
            or not isinstance(artifact.epoch, int)
            or artifact.epoch < -1
        ):
            raise ValueError("checkpoint artifact epoch must be an integer >= -1")
        if artifact.mode is not None and (
            isinstance(artifact.mode, bool)
            or not isinstance(artifact.mode, int)
            or not 0 <= artifact.mode <= 0o777
        ):
            raise ValueError("checkpoint artifact mode must be None or from 0o000 through 0o777")
        destinations.add(resolved)
        normalized_artifacts.append(
            CheckpointArtifact(
                destination=resolved,
                writer=artifact.writer,
                role=artifact.role,
                epoch=artifact.epoch,
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
                role="best",
                epoch=epoch,
            )
        )

    retirements: tuple[Path, ...] = ()
    if save_resumable:
        role: ResumableCheckpointRole = "epoch" if save_policy.mode == "all" else "latest"
        destination = root / resumable_checkpoint_filename(role, epoch)
        artifacts.append(
            CheckpointArtifact(
                destination=destination,
                writer=writers.resumable,
                role=role,
                epoch=epoch,
            )
        )
        if save_policy.mode == "latest":
            retirements = tuple(
                candidate.path
                for candidate in discover_resumable_checkpoints(root)
                if candidate.role == "latest" and candidate.path != destination
            )

    return CheckpointPlan(
        checkpoint_root=root,
        artifacts=tuple(artifacts),
        retire_after_commit=retirements,
    )


def resumable_checkpoint_filename(role: ResumableCheckpointRole, epoch: int) -> str:
    """Return Mammoth's canonical filename for one resumable trainer checkpoint."""
    if role not in {"epoch", "latest"}:
        raise ValueError("resumable checkpoint role must be 'epoch' or 'latest'")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < -1:
        raise ValueError("checkpoint epoch must be an integer >= -1")
    prefix = "epoch_" if role == "epoch" else "latest_epoch_"
    return f"{prefix}{epoch}.pt"


def parse_resumable_checkpoint(path: Path) -> ResumableCheckpointCandidate | None:
    """Parse one standard resumable checkpoint path without opening it."""
    checkpoint_path = Path(path)
    if checkpoint_path.parent != Path("."):
        return None
    match = _RESUMABLE_CHECKPOINT_FILENAME.fullmatch(checkpoint_path.name)
    if match is None:
        return None
    role: ResumableCheckpointRole = "latest" if match["latest"] is not None else "epoch"
    return ResumableCheckpointCandidate(
        path=checkpoint_path,
        role=role,
        epoch=int(match["number"]),
    )


def discover_resumable_checkpoints(
    checkpoint_root: Path,
) -> tuple[ResumableCheckpointCandidate, ...]:
    """Return direct-child standard candidates in deterministic resume precedence.

    Matching regular files, directories, and symlinks are returned alike. Callers
    own every filesystem-safety, payload-readability, and compatibility decision.
    """
    try:
        children = tuple(Path(checkpoint_root).iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return ()
    candidates = [
        ResumableCheckpointCandidate(child, candidate.role, candidate.epoch)
        for child in children
        if (candidate := parse_resumable_checkpoint(Path(child.name))) is not None
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.epoch,
                candidate.role != "latest",
                candidate.path.name,
            ),
        )
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
    receipts: list[PublishedCheckpoint] = []
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
                    inspect_serialized=receipt_inspector(artifact, receipts),
                )
            )
        for prepared_artifact in prepared:
            ensure_confined_path_unchanged(
                plan.checkpoint_root,
                prepared_artifact.destination,
                role="publication",
            )
            publish_prepared_artifact(prepared_artifact)
            committed_count += 1

        retired: list[Path] = []
        for path in plan.retire_after_commit:
            if retire_confined_path(
                plan.checkpoint_root,
                path,
                root_identity=plan.root_identity,
            ):
                retired.append(path)
        return CheckpointPublication(published=tuple(receipts), retired=tuple(retired))
    finally:
        for prepared_artifact in prepared[committed_count:]:
            discard_prepared_artifact(prepared_artifact)


def receipt_inspector(
    artifact: CheckpointArtifact,
    receipts: list[PublishedCheckpoint],
) -> Callable[[int], object]:
    """Capture validated regular-file bytes before final permissions and rename."""

    def inspect(serialized_descriptor: int) -> None:
        receipts.append(
            checkpoint_receipt_for_descriptor(
                serialized_descriptor,
                destination=artifact.destination,
                role=artifact.role,
                epoch=artifact.epoch,
            )
        )

    return inspect


def checkpoint_receipt_for_descriptor(
    descriptor: int,
    *,
    destination: Path,
    role: CheckpointRole,
    epoch: int,
) -> PublishedCheckpoint:
    """Hash a validated open artifact before permissions can prevent reading."""
    receipt = inspect_artifact_descriptor(descriptor, path=destination)
    return PublishedCheckpoint(
        path=destination,
        role=role,
        epoch=epoch,
        size_bytes=receipt.size_bytes,
        sha256=receipt.sha256,
    )


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
        operation.__name__ for operation in required if operation not in os.supports_dir_fd
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


def clone_checkpoint_value(value: Any) -> Any:
    """Recursively clone nested checkpoint state to detached, contiguous CPU tensors.

    Unlike :func:`snapshot_to_cpu`, non-tensor, non-container values are deep
    copied rather than returned by reference, so a checkpoint policy can safely
    retain the result alongside live, still-mutating training objects.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {deepcopy(key): clone_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_checkpoint_value(item) for item in value)
    return deepcopy(value)


def inspect_cuda_snapshot(value: Any) -> tuple[int, torch.device] | None:
    """Return safe CUDA snapshot storage or reject ambiguous checkpoint state."""
    cuda_device: torch.device | None = None
    snapshot_bytes = 0
    seen_containers: set[int] = set()
    seen_storages: set[tuple[str, int]] = set()

    def inspect(item: Any) -> bool:
        nonlocal cuda_device, snapshot_bytes
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_quantized or not item.is_contiguous():
                return False
            try:
                storage = item.untyped_storage()
                storage_key = (str(item.device), storage.data_ptr())
            except RuntimeError:
                return False
            if storage_key in seen_storages:
                return False
            seen_storages.add(storage_key)
            if item.device.type == "cuda":
                if cuda_device is None:
                    cuda_device = item.device
                elif item.device != cuda_device:
                    return False
                snapshot_bytes += item.numel() * item.element_size()
            return True
        if isinstance(item, (Mapping, tuple, list)):
            identity = id(item)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            values = item.values() if isinstance(item, Mapping) else item
            return all(inspect(child) for child in values)
        return item is None or isinstance(item, (bool, int, float, str, bytes))

    if not inspect(value):
        return None
    return snapshot_bytes, cuda_device or torch.device("cpu")


def snapshot_to_gpu(value: Any) -> Any:
    """Clone supported CUDA state before later training mutations can race it."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            return value.detach().clone()
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_to_gpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot_to_gpu(item) for item in value)
    if isinstance(value, list):
        return [snapshot_to_gpu(item) for item in value]
    return value


def cuda_snapshot_device(value: Any) -> torch.device | None:
    """Find the sole CUDA device in a snapshot already validated for capture."""
    if isinstance(value, torch.Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, Mapping):
        for item in value.values():
            if (device := cuda_snapshot_device(item)) is not None:
                return device
    elif isinstance(value, (tuple, list)):
        for item in value:
            if (device := cuda_snapshot_device(item)) is not None:
                return device
    return None


def publish_torch_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    """Serialize a snapshot through the core same-directory atomic publisher."""
    return atomic_publish(path, lambda temporary: torch.save(payload, temporary))


def publish_torch_payload_with_receipt(
    path: Path,
    payload: Mapping[str, Any],
    *,
    role: CheckpointRole,
    epoch: int,
) -> CheckpointPublication:
    """Atomically publish one generic payload with pre-rename byte provenance."""
    destination = Path(path).absolute()
    receipt: PublishedCheckpoint | None = None

    def inspect(serialized_descriptor: int) -> None:
        nonlocal receipt
        receipt = checkpoint_receipt_for_descriptor(
            serialized_descriptor,
            destination=destination,
            role=role,
            epoch=epoch,
        )

    atomic_publish(
        destination,
        lambda temporary: torch.save(payload, temporary),
        inspect_serialized=inspect,
    )
    assert receipt is not None
    return CheckpointPublication(published=(receipt,), retired=())
