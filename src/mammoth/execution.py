"""Framework-neutral direct execution sessions composed from core and logging.

This module turns immutable :mod:`mammoth.core` execution identities and
:mod:`mammoth.logging` observability into a context-managed direct-process
lifecycle.  Optional framework integrations, including :mod:`mammoth.torch`,
compose this session without adding framework dependencies to ``mammoth.core``.
"""

from __future__ import annotations

import logging
import os
import signal as signal_module
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Self

from mammoth.core import (
    BackgroundPipelineError,
    BoundedBackgroundPipeline,
    ExecutionContext,
    LogicalRunLease,
    claim_logical_run_lease,
    create_execution_context,
    execution_id_from_environment,
    join_execution_context,
    latest_execution_id,
    sanitize_command,
    sanitize_metadata_fields,
    sanitize_reference,
    validate_execution_id,
    validate_resume_checkpoint_sha256,
    validate_run_name,
)
from mammoth.core.execution import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    PHASE_ENV,
    RUN_NAME_ENV,
)
from mammoth.logging import (
    ExecutionLogging,
    ObservationSink,
    RunObserver,
    create_execution_logging,
)

__all__ = ["ExecutionSession", "ExecutionSpec"]


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Immutable expected facts for strict direct execution creation or attachment.

    :class:`ExecutionSession` sanitizes this specification before publishing or
    comparing metadata.  Framework adapters may add their own provenance before
    calling the lower-level core creation API, but the caller-owned values stay
    detached and immutable here.
    """

    run_dir: Path
    run_name: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    config_reference: str | Path = ""
    execution_id: str | None = None
    resume_checkpoint: str | Path | None = None
    resume_checkpoint_sha256: str | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach mutable inputs and validate the complete resume-fact group."""
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        validate_run_name(self.run_name)
        if not isinstance(self.invocation_kind, str) or not self.invocation_kind:
            raise ValueError("invocation_kind must be a non-empty string")
        if isinstance(self.intended_phases, str) or not isinstance(
            self.intended_phases, Sequence
        ):
            raise ValueError("intended_phases must be a sequence of phase names")
        phases = tuple(self.intended_phases)
        if not phases or any(not isinstance(phase, str) or not phase for phase in phases):
            raise ValueError("intended_phases must contain non-empty phase names")
        if len(set(phases)) != len(phases):
            raise ValueError("intended_phases must not contain duplicate phase names")
        object.__setattr__(self, "intended_phases", phases)
        if not isinstance(self.runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        object.__setattr__(self, "runtime", _freeze_runtime_mapping(self.runtime))
        _validate_resume_coordinate("starting_epoch", self.starting_epoch)
        _validate_resume_coordinate("starting_global_step", self.starting_global_step)
        _validate_resume_facts(
            resume_checkpoint=self.resume_checkpoint,
            resume_checkpoint_sha256=self.resume_checkpoint_sha256,
            parent_execution_id=self.parent_execution_id,
            starting_epoch=self.starting_epoch,
            starting_global_step=self.starting_global_step,
        )


@dataclass(frozen=True, slots=True)
class _JoinEnvironment:
    """Canonical workflow-child values required by strict direct attachment."""

    execution_id: str
    run_name: str
    invocation_kind: str
    phase: str


class ExecutionSession:
    """Own one direct process lifecycle and its framework-neutral resources.

    Direct callers use :meth:`create` or :meth:`attach` for a single process.
    Framework adapters may use :meth:`from_established` after they have handled
    framework-specific coordination, then retain their own resources around
    this session's observer, pipeline, lifecycle, and logging ownership.
    """

    def __init__(
        self,
        context: ExecutionContext,
        execution_logging: ExecutionLogging,
        *,
        close_callbacks: Sequence[tuple[str, Callable[[], None]]] = (),
    ) -> None:
        """Create a lifecycle owner for already-established execution resources."""
        if execution_logging.context != context:
            raise ValueError("execution logging must belong to the session context")
        self.context = context
        self.execution_logging = execution_logging
        self.observer = execution_logging.observer
        self.event_writer = execution_logging.event_writer
        self._close_callbacks = tuple(close_callbacks)
        self._phase: str | None = None
        self._phase_terminal = False
        self._phase_outcome: Literal["completed", "failed", "interrupted", "skipped"] | None = None
        self._process_started_at: float | None = None
        self._phase_started_at: float | None = None
        self._owned_resources = ExitStack()
        self._owned_observers = ExitStack()
        self._owned_pipelines = ExitStack()
        self._owned_resources.callback(self._owned_observers.close)
        self._owned_resources.callback(self._owned_pipelines.close)
        self._owned_resource_errors: list[tuple[str, BaseException]] = []
        self._resource_lock = RLock()
        self._closed = False

    @classmethod
    def create(
        cls,
        spec: ExecutionSpec,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> Self:
        """Create a single-process execution, logging bundle, and leased session."""
        if not isinstance(spec, ExecutionSpec):
            raise TypeError("spec must be an ExecutionSpec")
        inherited_id = execution_id_from_environment()
        if inherited_id is not None:
            raise ValueError(f"{EXECUTION_ID_ENV} must not be set when creating an execution")
        lease: LogicalRunLease | None = None
        logging_bundle: ExecutionLogging | None = None
        try:
            lease = claim_logical_run_lease(spec.run_dir)
            context = create_execution_context(
                spec.run_dir,
                run_name=spec.run_name,
                invocation_kind=spec.invocation_kind,
                intended_phases=spec.intended_phases,
                world_size=1,
                execution_mode="single",
                command=sanitize_command(tuple(sys.argv)),
                config_reference=(
                    "" if spec.config_reference == "" else sanitize_reference(spec.config_reference)
                ),
                execution_id=spec.execution_id,
                previous_execution_id=latest_execution_id(spec.run_dir),
                resume_checkpoint=(
                    sanitize_reference(spec.resume_checkpoint)
                    if spec.resume_checkpoint is not None
                    else None
                ),
                resume_checkpoint_sha256=spec.resume_checkpoint_sha256,
                parent_execution_id=spec.parent_execution_id,
                starting_epoch=spec.starting_epoch,
                starting_global_step=spec.starting_global_step,
                runtime=sanitize_metadata_fields(dict(spec.runtime)),
            )
            logging_bundle = create_execution_logging(
                context,
                rank=0,
                world_size=1,
                additional_sinks=additional_sinks,
                text_level=text_level,
            )
        except BaseException as error:
            cleanup_errors: list[tuple[str, BaseException]] = []
            if lease is not None:
                try:
                    lease.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(("logical-run lease", cleanup_error))
            _attach_cleanup_errors(error, cleanup_errors)
            raise
        return cls(
            context,
            logging_bundle,
            close_callbacks=(("logical-run lease", lease.close),),
        )

    @classmethod
    def attach(
        cls,
        expected: ExecutionSpec,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> Self:
        """Strictly attach one single-process workflow child without a lease."""
        if not isinstance(expected, ExecutionSpec):
            raise TypeError("expected must be an ExecutionSpec")
        environment = _canonical_join_environment(expected)
        context = join_execution_context(
            expected.run_dir,
            environment.execution_id,
            expected_run_name=expected.run_name,
        )
        _validate_attached_context(
            context,
            expected,
            environment,
            world_size=1,
            execution_mode="single",
        )
        logging_bundle = create_execution_logging(
            context,
            rank=0,
            world_size=1,
            additional_sinks=additional_sinks,
            text_level=text_level,
        )
        return cls(context, logging_bundle)

    @classmethod
    def from_established(
        cls,
        context: ExecutionContext,
        execution_logging: ExecutionLogging,
        *,
        close_callbacks: Sequence[tuple[str, Callable[[], None]]] = (),
    ) -> Self:
        """Wrap framework-established context and logging without importing it."""
        return cls(context, execution_logging, close_callbacks=close_callbacks)

    @property
    def phase(self) -> str | None:
        """Return the active or most recently completed phase name."""
        return self._phase

    def create_observer(
        self,
        sinks: Sequence[ObservationSink] = (),
        *,
        observer_factory: Callable[[Sequence[ObservationSink]], RunObserver] = RunObserver,
    ) -> RunObserver:
        """Create and own one sink-neutral observer for caller output."""
        with self._resource_lock:
            self._require_open()
            observer = observer_factory(sinks)
            self._register_owned_resource(self._owned_observers, "observer", observer.close)
            return observer

    def create_background_pipeline[InputT, ResultT](
        self,
        worker: Callable[[InputT], ResultT],
        *,
        max_pending: int = 1,
        thread_name_prefix: str = "mammoth-background",
        pipeline_factory: Callable[..., BoundedBackgroundPipeline[InputT, ResultT]] = (
            BoundedBackgroundPipeline
        ),
    ) -> BoundedBackgroundPipeline[InputT, ResultT]:
        """Create a pipeline that closes before owned observers."""
        self._require_open()
        pipeline = pipeline_factory(
            worker,
            max_pending=max_pending,
            thread_name_prefix=thread_name_prefix,
        )

        def close_pipeline() -> None:
            cleanup_errors: list[BaseException] = []

            def acknowledge_submission(submission: Any) -> None:
                while pipeline.owns(submission):
                    try:
                        pipeline.acknowledge(submission)
                    except BaseException as error:
                        cleanup_errors.append(error)

            while True:
                try:
                    completed = pipeline.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                    if isinstance(error, BackgroundPipelineError):
                        acknowledge_submission(error.submission)
                    continue
                for result in completed:
                    acknowledge_submission(result.submission)
                break
            if cleanup_errors:
                first_error = cleanup_errors[0]
                for later_error in cleanup_errors[1:]:
                    first_error.add_note(
                        "Later background pipeline cleanup failure: "
                        f"{type(later_error).__name__}: {later_error}"
                    )
                raise first_error

        try:
            with self._resource_lock:
                self._require_open()
                self._register_owned_resource(
                    self._owned_pipelines,
                    "background pipeline",
                    close_pipeline,
                )
        except BaseException as registration_error:
            try:
                close_pipeline()
            except BaseException as cleanup_error:
                registration_error.add_note(
                    "Unregistered background pipeline cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                for note in getattr(cleanup_error, "__notes__", ()):
                    registration_error.add_note(
                        f"Unregistered background pipeline cleanup detail: {note}"
                    )
            raise
        return pipeline

    def start_phase(self, phase: str) -> None:
        """Start one phase and, on first use, this process lifecycle."""
        with self._resource_lock:
            self._require_open()
            if not isinstance(phase, str) or not phase:
                raise ValueError("phase must be a non-empty string")
            if self._phase is not None and not self._phase_terminal:
                raise RuntimeError(f"Execution phase {self._phase!r} is still active")
            now = time.monotonic()
            if self._process_started_at is None:
                self._process_started_at = now
                self.observer.emit("process_started", phase=phase)
            self._phase = phase
            self._phase_terminal = False
            self._phase_outcome = None
            self._phase_started_at = now
            self.observer.emit("phase_started", phase=phase)

    @contextmanager
    def phase_scope(self, phase: str) -> Iterator[Self]:
        """Own one phase's success, failure, and interruption transition."""
        self.start_phase(phase)
        try:
            yield self
        except BaseException as error:
            self.fail_phase(error, interrupted=isinstance(error, KeyboardInterrupt))
            raise
        else:
            self.complete_phase()

    def complete_phase(self, *, message: str | None = None) -> None:
        """Mark the active phase successful."""
        with self._resource_lock:
            self._require_open()
            phase = self._active_phase()
            fields: dict[str, Any] = {
                "phase": phase,
                "duration_seconds": self._phase_duration(),
            }
            if message is not None:
                fields["message"] = message
            self.observer.emit("phase_completed", **fields)
            self._phase_terminal = True
            self._phase_outcome = "completed"

    def fail_phase(self, error: BaseException, *, interrupted: bool = False) -> None:
        """Mark the active phase failed or interrupted."""
        with self._resource_lock:
            self._require_open()
            self._fail_phase(error, interrupted=interrupted)

    def skip_phase(self, message: str) -> None:
        """Mark the active phase skipped."""
        with self._resource_lock:
            self._require_open()
            phase = self._active_phase()
            self.observer.emit(
                "phase_skipped",
                phase=phase,
                duration_seconds=self._phase_duration(),
                message=message,
            )
            self._phase_terminal = True
            self._phase_outcome = "skipped"

    def close(
        self,
        *,
        error: BaseException | None = None,
        exit_code: int | None = None,
        signal: int | str | None = None,
        message: str | None = None,
        before_close: Callable[[], None] | None = None,
        prior_cleanup_errors: Sequence[tuple[str, BaseException]] = (),
    ) -> None:
        """Close owned resources, lifecycle logging, and registered finalizers once."""
        with self._resource_lock:
            if self._closed:
                return
            self._closed = True
        cleanup_errors = list(prior_cleanup_errors)
        self._owned_resource_errors = cleanup_errors
        self._owned_resources.close()
        if before_close is not None:
            try:
                before_close()
            except BaseException as callback_error:
                cleanup_errors.append(("presentation lease", callback_error))
        terminal_error = error or _first_cleanup_error(cleanup_errors)
        try:
            if self._phase is not None:
                if not self._phase_terminal:
                    lifecycle_error = terminal_error or RuntimeError(
                        "Execution session closed before recording a phase outcome"
                    )
                    self._fail_phase(
                        lifecycle_error,
                        interrupted=isinstance(lifecycle_error, KeyboardInterrupt),
                    )
                effective_exit_code = _process_exit_code(
                    terminal_error,
                    requested=exit_code,
                    phase_outcome=self._phase_outcome,
                    cleanup_failed=bool(cleanup_errors),
                )
                fields: dict[str, Any] = {
                    "phase": self._phase,
                    "duration_seconds": self._process_duration(),
                    "exit_code": effective_exit_code,
                }
                effective_signal = signal
                if effective_signal is None and isinstance(terminal_error, KeyboardInterrupt):
                    effective_signal = signal_module.SIGINT
                if effective_signal is not None:
                    fields["signal"] = effective_signal
                if message is not None:
                    fields["message"] = message
                elif terminal_error is not None:
                    fields["message"] = _error_text(terminal_error)
                self.observer.emit("process_completed", **fields)
        except BaseException as lifecycle_error:
            cleanup_errors.append(("execution lifecycle", lifecycle_error))
        try:
            self.execution_logging.close()
        except BaseException as logging_error:
            cleanup_errors.append(("execution logging", logging_error))
        for label, callback in self._close_callbacks:
            try:
                callback()
            except BaseException as callback_error:
                cleanup_errors.append((label, callback_error))
        if error is not None:
            _attach_cleanup_errors(error, cleanup_errors)
        elif cleanup_errors and exit_code in {None, 0}:
            first_label, first_error = cleanup_errors[0]
            _attach_cleanup_errors(first_error, cleanup_errors[1:])
            first_error.add_note(f"Cleanup stage: {first_label}")
            raise first_error

    def __enter__(self) -> Self:
        """Return this open execution session."""
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session without replacing a workload exception."""
        del exc_type, traceback
        self.close(error=exc_value)

    def _fail_phase(self, error: BaseException, *, interrupted: bool) -> None:
        """Record phase failure while the caller holds lifecycle ownership."""
        phase = self._active_phase()
        self.observer.emit(
            "phase_failed",
            phase=phase,
            duration_seconds=self._phase_duration(),
            message=_error_text(error),
            status="interrupted" if interrupted else "failed",
            error_type=type(error).__name__,
        )
        self._phase_terminal = True
        self._phase_outcome = "interrupted" if interrupted else "failed"

    def _register_owned_resource(
        self,
        stack: ExitStack,
        label: str,
        close: Callable[[], None],
    ) -> None:
        """Register one close callback on the session's reverse-order stack."""
        stack.callback(self._close_owned_resource, label, close)

    def _close_owned_resource(self, label: str, close: Callable[[], None]) -> None:
        """Capture one owned-resource failure while allowing later cleanup."""
        try:
            close()
        except BaseException as error:
            self._owned_resource_errors.append((label, error))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Execution session is already closed")

    def _active_phase(self) -> str:
        if self._phase is None:
            raise RuntimeError("No execution phase has been started")
        if self._phase_terminal:
            raise RuntimeError(f"Execution phase {self._phase!r} is already terminal")
        return self._phase

    def _phase_duration(self) -> float:
        if self._phase_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._phase_started_at)

    def _process_duration(self) -> float:
        if self._process_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._process_started_at)


def _canonical_join_environment(expected: ExecutionSpec) -> _JoinEnvironment:
    """Validate the four canonical workflow-child variables for single-process attach."""
    execution_id = execution_id_from_environment()
    if execution_id is None:
        raise ValueError(f"{EXECUTION_ID_ENV} is required when attaching an execution")
    values: dict[str, str] = {}
    for name in (RUN_NAME_ENV, INVOCATION_KIND_ENV, PHASE_ENV):
        value = os.environ.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} is required when attaching an execution")
        values[name] = value
    environment = _JoinEnvironment(
        execution_id=execution_id,
        run_name=values[RUN_NAME_ENV],
        invocation_kind=values[INVOCATION_KIND_ENV],
        phase=values[PHASE_ENV],
    )
    if environment.run_name != expected.run_name:
        raise ValueError(
            f"{RUN_NAME_ENV}={environment.run_name!r} does not match expected run "
            f"{expected.run_name!r}"
        )
    if environment.invocation_kind != expected.invocation_kind:
        raise ValueError(
            f"{INVOCATION_KIND_ENV}={environment.invocation_kind!r} does not match "
            f"expected invocation {expected.invocation_kind!r}"
        )
    if environment.phase not in expected.intended_phases:
        raise ValueError(
            f"{PHASE_ENV}={environment.phase!r} is not one of the expected phases"
        )
    if expected.execution_id is not None and environment.execution_id != expected.execution_id:
        raise ValueError(
            f"{EXECUTION_ID_ENV}={environment.execution_id!r} does not match expected "
            f"execution {expected.execution_id!r}"
        )
    return environment


def _validate_attached_context(
    context: ExecutionContext,
    expected: ExecutionSpec,
    environment: _JoinEnvironment,
    *,
    world_size: int,
    execution_mode: Literal["single", "distributed"],
) -> None:
    """Require exact identity, topology, metadata, and resume facts on attachment."""
    metadata = context.metadata
    if context.run_dir.resolve() != expected.run_dir.resolve():
        raise ValueError("Attached execution resolved to an unexpected run directory")
    if metadata.execution_id != environment.execution_id:
        raise ValueError("Canonical execution ID does not match immutable metadata")
    if metadata.run_name != expected.run_name or metadata.run_name != environment.run_name:
        raise ValueError("Canonical run name does not match immutable metadata")
    if (
        metadata.invocation_kind != expected.invocation_kind
        or metadata.invocation_kind != environment.invocation_kind
    ):
        raise ValueError("Canonical invocation kind does not match immutable metadata")
    if metadata.world_size != world_size:
        raise ValueError(
            f"Execution {metadata.execution_id!r} has world_size={metadata.world_size}, "
            f"but this direct session has world_size={world_size}"
        )
    if metadata.execution_mode != execution_mode:
        raise ValueError(
            f"Execution {metadata.execution_id!r} uses {metadata.execution_mode!r}, "
            f"but this direct session uses {execution_mode!r}"
        )
    missing_phases = set(expected.intended_phases).difference(metadata.intended_phases)
    if missing_phases:
        raise ValueError(
            f"Execution {metadata.execution_id!r} omits phases: "
            f"{', '.join(sorted(missing_phases))}"
        )
    if environment.phase not in metadata.intended_phases:
        raise ValueError(
            f"Canonical phase {environment.phase!r} is absent from immutable metadata"
        )
    expected_config_reference = (
        "" if expected.config_reference == "" else sanitize_reference(expected.config_reference)
    )
    if metadata.config_reference != expected_config_reference:
        raise ValueError("Attached execution has an unexpected config reference")
    expected_runtime = sanitize_metadata_fields(dict(expected.runtime))
    actual_runtime = sanitize_metadata_fields(dict(metadata.runtime or {}))
    if actual_runtime != expected_runtime:
        raise ValueError("Attached execution has unexpected runtime metadata")
    _validate_resume_metadata(metadata, expected)


def _validate_resume_metadata(metadata: Any, expected: ExecutionSpec) -> None:
    """Compare all nullable resume facts without a fresh-execution shortcut."""
    expected_facts = {
        "resume_checkpoint": (
            sanitize_reference(expected.resume_checkpoint)
            if expected.resume_checkpoint is not None
            else None
        ),
        "resume_checkpoint_sha256": expected.resume_checkpoint_sha256,
        "parent_execution_id": expected.parent_execution_id,
        "starting_epoch": expected.starting_epoch,
        "starting_global_step": expected.starting_global_step,
    }
    for field_name, expected_value in expected_facts.items():
        actual_value = getattr(metadata, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"Execution {metadata.execution_id!r} resume field {field_name!r} "
                f"does not match: metadata={actual_value!r}, expected={expected_value!r}."
            )


def _validate_resume_coordinate(name: str, value: int | None) -> None:
    """Reject non-integral resume coordinates before exact metadata comparison."""
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}.")


def _validate_resume_facts(
    *,
    resume_checkpoint: str | Path | None,
    resume_checkpoint_sha256: str | None,
    parent_execution_id: str | None,
    starting_epoch: int | None,
    starting_global_step: int | None,
) -> None:
    """Require resume provenance to be wholly absent or fully attestable."""
    if resume_checkpoint is None:
        populated = {
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "parent_execution_id": parent_execution_id,
            "starting_epoch": starting_epoch,
            "starting_global_step": starting_global_step,
        }
        unexpected = sorted(name for name, value in populated.items() if value is not None)
        if unexpected:
            raise ValueError(
                "resume facts require resume_checkpoint: " + ", ".join(unexpected)
            )
        return
    try:
        checkpoint_reference = os.fspath(resume_checkpoint)
    except TypeError as error:
        raise ValueError("resume_checkpoint must be a non-empty path or reference") from error
    if not isinstance(checkpoint_reference, str) or not checkpoint_reference:
        raise ValueError("resume_checkpoint must be a non-empty path or reference")
    if resume_checkpoint_sha256 is None:
        raise ValueError("resume_checkpoint requires resume_checkpoint_sha256.")
    validate_resume_checkpoint_sha256(resume_checkpoint_sha256)
    if starting_epoch is None:
        raise ValueError("resume_checkpoint requires starting_epoch.")
    if parent_execution_id is not None:
        validate_execution_id(parent_execution_id)


def _freeze_runtime_mapping(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively detach JSON-shaped runtime metadata from caller-owned values."""
    return MappingProxyType(
        {name: _freeze_runtime_value(value) for name, value in runtime.items()}
    )


def _freeze_runtime_value(value: Any) -> Any:
    """Freeze nested mappings and sequences retained by :class:`ExecutionSpec`."""
    if isinstance(value, Mapping):
        return _freeze_runtime_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | os.PathLike):
        return tuple(_freeze_runtime_value(item) for item in value)
    return value


def _error_text(error: BaseException) -> str:
    """Render a bounded exception identity for lifecycle event metadata."""
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _first_cleanup_error(
    errors: Sequence[tuple[str, BaseException]],
) -> BaseException | None:
    """Return the first captured cleanup failure, if any."""
    return errors[0][1] if errors else None


def _attach_cleanup_errors(
    primary_error: BaseException,
    errors: Sequence[tuple[str, BaseException]],
) -> None:
    """Retain cleanup failures as notes without replacing the primary error."""
    for label, error in errors:
        primary_error.add_note(f"{label} cleanup also failed: {_error_text(error)}")
        for note in getattr(error, "__notes__", ()):
            primary_error.add_note(f"{label} cleanup detail: {note}")


def _process_exit_code(
    error: BaseException | None,
    *,
    requested: int | None,
    phase_outcome: str | None,
    cleanup_failed: bool,
) -> int:
    """Derive a non-success process code without erasing caller precedence."""
    if requested is not None:
        exit_code = requested
    elif isinstance(error, KeyboardInterrupt):
        exit_code = 130
    elif isinstance(error, SystemExit):
        exit_code = error.code if isinstance(error.code, int) else 1
    elif error is not None:
        exit_code = 1
    else:
        exit_code = 0
    if exit_code == 0 and (cleanup_failed or phase_outcome in {"failed", "interrupted"}):
        return 1
    return exit_code
