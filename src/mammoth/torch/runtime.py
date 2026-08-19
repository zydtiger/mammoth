"""Own PyTorch single-process and standard DDP execution state.

CLI entrypoints construct this runtime before project code. The generic trainer
consumes its device and rank identity. The runtime composes the framework-neutral
direct execution session after it has established an immutable attempt and one
process-owned logging stream per rank.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Literal, cast

import torch
import torch.distributed as dist

from mammoth.core import (
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
)
from mammoth.core.execution import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    PHASE_ENV,
    RUN_NAME_ENV,
)
from mammoth.execution import ExecutionSession as NeutralExecutionSession
from mammoth.execution import ExecutionSpec as ExecutionSpec
from mammoth.logging import (
    ExecutionLogging,
    ObservationSink,
    RunObserver,
    create_execution_logging,
)
from mammoth.torch.device import resolve_device
from mammoth.torch.scheduling import weighted_partition_counts, weighted_partition_indices
from mammoth.torch.trainer import Trainer

Strategy = Literal["single", "ddp"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Framework-level process-group and device policy."""

    strategy: Strategy = "single"
    device: str = "auto"
    backend: str | None = None
    init_method: str = "env://"
    timeout_seconds: float = 1800.0
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None
    workload_weights: tuple[float, ...] | None = None
    strict_launch_environment: bool = False
    require_global_local_rank_match: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {"single", "ddp"}:
            raise ValueError(f"Unsupported torch runtime strategy: {self.strategy!r}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        if self.backend is not None and not self.backend:
            raise ValueError("backend must be a non-empty string when provided")
        if not self.init_method:
            raise ValueError("init_method must be a non-empty string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        for name, value in (
            ("rank", self.rank),
            ("local_rank", self.local_rank),
            ("world_size", self.world_size),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer when provided")
        if self.world_size == 0:
            raise ValueError("world_size must be positive when provided")
        if (self.rank is None) != (self.world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if self.rank is not None and self.world_size is not None and self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.workload_weights is not None:
            weights = tuple(self.workload_weights)
            if not weights or any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
                for weight in weights
            ):
                raise ValueError("workload_weights must contain positive finite numbers")
            object.__setattr__(self, "workload_weights", tuple(float(weight) for weight in weights))
        for name, value in (
            ("strict_launch_environment", self.strict_launch_environment),
            ("require_global_local_rank_match", self.require_global_local_rank_match),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class _PrimaryResult:
    execution_id: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _PrimaryValue:
    value: Any
    error: str | None


@dataclass(frozen=True, slots=True)
class _StartupStatus:
    rank: int
    error: str | None


@dataclass(frozen=True, slots=True)
class _JoinEnvironment:
    """Canonical workflow-child environment shared by every attaching rank."""

    execution_id: str
    run_name: str
    invocation_kind: str
    phase: str


@dataclass(frozen=True, slots=True)
class _AttachmentIntent:
    """Rank-local strict attachment inputs that must agree before validation."""

    environment: _JoinEnvironment
    run_dir: str
    run_name: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    config_reference: str
    execution_id: str | None
    resume_checkpoint: str | None
    resume_checkpoint_sha256: str | None
    parent_execution_id: str | None
    starting_epoch: int | None
    starting_global_step: int | None
    runtime: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CreationIntent:
    """Normalized strict-creation inputs that must agree before publication."""

    run_dir: str
    run_name: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    command: tuple[str, ...]
    config_reference: str
    execution_id: str | None
    resume_checkpoint: str | None
    resume_checkpoint_sha256: str | None
    parent_execution_id: str | None
    starting_epoch: int | None
    starting_global_step: int | None
    runtime: dict[str, Any]


class Runtime:
    """Own one process's PyTorch identity, collectives, and execution IO."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.strategy = self.config.strategy
        self._owns_process_group = False
        self._logical_run_lease: LogicalRunLease | None = None
        self.execution_logging: ExecutionLogging | None = None
        self.execution_context: ExecutionContext | None = None
        self._execution_session: ExecutionSession | None = None
        self._state_lock = RLock()
        self._closed = False

        if self.strategy == "single":
            if self.config.strict_launch_environment and _environment_world_size() > 1:
                raise RuntimeError(
                    "Single-process strategy cannot run inside a multi-process launch"
                )
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = resolve_device(self.config.device)
            self.backend = None
            self.workload_weights = self._resolve_workload_weights()
            return

        self.rank, self.local_rank, self.world_size = _distributed_identity(self.config)
        if self.world_size < 2:
            raise RuntimeError("DDP strategy requires a world size of at least two")
        if (
            self.config.require_global_local_rank_match
            and self.rank != self.local_rank
        ):
            raise RuntimeError(
                "This runtime requires global rank and local rank to match; "
                f"got rank={self.rank}, local_rank={self.local_rank}"
            )
        self.workload_weights = self._resolve_workload_weights()
        self.device = _distributed_device(self.config.device, self.local_rank)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.backend = self.config.backend or ("nccl" if self.device.type == "cuda" else "gloo")
        if dist.is_initialized():
            self._validate_initialized_group()
        else:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is unavailable")
            dist.init_process_group(
                backend=self.backend,
                init_method=self.config.init_method,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=float(self.config.timeout_seconds)),
            )
            self._owns_process_group = True

    @property
    def enabled(self) -> bool:
        """Return whether standard DDP collectives are active."""
        return self.strategy == "ddp"

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns shared artifact publication."""
        return self.rank == 0

    def broadcast_object(self, value: Any, *, source_rank: int = 0) -> Any:
        """Broadcast one picklable object from ``source_rank``."""
        if not self.enabled:
            return value
        objects = [value if self.rank == source_rank else None]
        dist.broadcast_object_list(objects, src=source_rank)
        return objects[0]

    def broadcast_bool(self, value: bool, *, source_rank: int = 0) -> bool:
        """Broadcast one boolean decision from ``source_rank``."""
        result = self.broadcast_object(value, source_rank=source_rank)
        if not isinstance(result, bool):
            raise RuntimeError("Boolean broadcast returned an invalid payload")
        return result

    def gather_object(
        self,
        value: Any,
        *,
        destination_rank: int = 0,
    ) -> list[Any] | None:
        """Gather one object per rank on ``destination_rank``."""
        if not self.enabled:
            return [value]
        gathered: list[Any] | None = (
            [None] * self.world_size if self.rank == destination_rank else None
        )
        dist.gather_object(value, gathered, dst=destination_rank)
        return gathered

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Return one gathered object per rank on every process."""
        if not self.enabled:
            return (value,)
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return tuple(gathered)

    def shared_string_union(self, values: Iterable[str]) -> tuple[str, ...]:
        """Return one sorted string union shared by every participating rank."""
        local_values = tuple(sorted(set(values)))
        gathered = self.all_gather_object(local_values)
        if not all(
            isinstance(rank_values, tuple)
            and all(isinstance(value, str) for value in rank_values)
            for rank_values in gathered
        ):
            raise RuntimeError("Distributed string union received an invalid payload")
        return tuple(sorted({value for rank_values in gathered for value in rank_values}))

    def local_partition_count(
        self,
        total_count: int,
        *,
        require_nonempty: bool = False,
    ) -> int:
        """Return this rank's caller-weighted share of ``total_count``."""
        counts = weighted_partition_counts(
            total_count,
            self.workload_weights,
            require_nonempty=require_nonempty,
        )
        return counts[self.rank]

    def local_partition_indices(self, total_count: int) -> range:
        """Return this rank's caller-weighted contiguous item range."""
        return weighted_partition_indices(total_count, self.rank, self.workload_weights)

    def scatter_object(
        self,
        values: Sequence[Any] | None,
        *,
        source_rank: int = 0,
    ) -> Any:
        """Scatter one object per rank from ``source_rank``."""
        if not self.enabled:
            if values is None or len(values) != 1:
                raise ValueError("Single-process object scatter requires exactly one value")
            return values[0]
        if self.rank == source_rank and (values is None or len(values) != self.world_size):
            raise ValueError(f"Source scatter requires {self.world_size} values")
        output: list[Any] = [None]
        source_values = list(values) if self.rank == source_rank and values is not None else None
        dist.scatter_object_list(output, source_values, src=source_rank)
        return output[0]

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum a tensor in place across all ranks and return it."""
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """Take an elementwise maximum in place across all ranks and return it."""
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    def barrier(self) -> None:
        """Wait until every active DDP process reaches this point."""
        if not self.enabled:
            return
        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def coordinate_primary[T](self, operation_name: str, operation: Callable[[], T]) -> T:
        """Run one fallible operation on rank zero and broadcast its result."""
        result: _PrimaryValue | None = None
        if self.is_primary:
            try:
                result = _PrimaryValue(value=operation(), error=None)
            except Exception as error:
                result = _PrimaryValue(value=None, error=_error_text(error))
        result = self.broadcast_object(result)
        if not isinstance(result, _PrimaryValue):
            raise RuntimeError(f"Primary operation {operation_name!r} returned invalid data")
        if result.error is not None:
            raise RuntimeError(f"Primary operation {operation_name!r} failed: {result.error}")
        return cast(T, result.value)

    def startup_consensus(self, stage: str, local_error: BaseException | str | None) -> None:
        """Make every rank raise the same bounded startup failure decision."""
        error_text = (
            None
            if local_error is None
            else local_error
            if isinstance(local_error, str)
            else _error_text(local_error)
        )
        statuses = self.all_gather_object(_StartupStatus(rank=self.rank, error=error_text))
        if not all(isinstance(status, _StartupStatus) for status in statuses):
            raise RuntimeError(f"Distributed startup stage {stage!r} returned invalid status data")
        failures = [status for status in statuses if status.error is not None]
        if failures:
            details = "; ".join(f"rank {status.rank}: {status.error}" for status in failures)
            raise RuntimeError(f"Distributed startup stage {stage!r} failed: {details}")

    def start_execution_logging(
        self,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> ExecutionLogging:
        """Open rank-local logging after execution establishment and validation."""
        with self._state_lock:
            return self._start_execution_logging(
                additional_sinks=additional_sinks,
                text_level=text_level,
            )

    def _start_execution_logging(
        self,
        *,
        additional_sinks: Sequence[ObservationSink],
        text_level: int,
    ) -> ExecutionLogging:
        """Open logging while the runtime terminal-state lock is held."""
        self._require_open()
        if self.execution_logging is not None:
            raise RuntimeError("This torch runtime has already started execution logging")
        context = self.execution_context
        if context is None:
            raise RuntimeError("Establish an execution before starting execution logging")
        logging_bundle: ExecutionLogging | None = None
        local_error: BaseException | None = None
        try:
            logging_bundle = create_execution_logging(
                context,
                rank=self.rank,
                world_size=self.world_size,
                additional_sinks=additional_sinks,
                text_level=text_level,
            )
        except BaseException as error:
            local_error = error
        try:
            self.startup_consensus("execution logging", local_error)
        except BaseException as startup_error:
            self._terminate_failed_logging_startup(logging_bundle, startup_error)
            raise
        if logging_bundle is None:
            missing_bundle_error = RuntimeError(
                "Execution logging startup succeeded without a local bundle"
            )
            self._terminate_failed_logging_startup(None, missing_bundle_error)
            raise missing_bundle_error
        self.execution_context = context
        self.execution_logging = logging_bundle
        return logging_bundle

    def create_execution_session(self) -> ExecutionSession:
        """Create the generic process/phase lifecycle owner for this runtime."""
        with self._state_lock:
            self._require_open()
            if self._execution_session is not None:
                raise RuntimeError("This torch runtime already has an execution session")
            if self.execution_logging is None:
                raise RuntimeError("Start execution logging before creating a session")
            session = ExecutionSession(self)
            self._execution_session = session
            return session

    def create_execution(self, spec: ExecutionSpec) -> ExecutionContext:
        """Create a new execution and never attach to an inherited attempt.

        Rank zero claims the logical-run lease, resolves adjacency while that
        lease is held, publishes immutable metadata, and shares the new ID.
        Every rank then joins and validates the published attempt by consensus.
        """
        with self._state_lock:
            self._require_open()
            if not isinstance(spec, ExecutionSpec):
                raise TypeError("spec must be an ExecutionSpec")
            if self.execution_context is not None:
                raise RuntimeError("This torch runtime has already established an execution")
            context = self._create_execution(spec)
            self.execution_context = context
            return context

    def attach_execution(self, expected: ExecutionSpec) -> ExecutionContext:
        """Attach to exactly one canonical workflow execution without leasing it."""
        with self._state_lock:
            self._require_open()
            if not isinstance(expected, ExecutionSpec):
                raise TypeError("expected must be an ExecutionSpec")
            if self.execution_context is not None:
                raise RuntimeError("This torch runtime has already established an execution")
            context = self._attach_execution(expected)
            self.execution_context = context
            return context

    def close_process_group(self) -> None:
        """Destroy the default process group only when this runtime created it."""
        if not self._owns_process_group:
            return
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            self._owns_process_group = False

    def close(self) -> None:
        """Close logging, release the primary lease, and destroy an owned group."""
        with self._state_lock:
            if self._closed:
                return
            if self._execution_session is not None:
                self._execution_session.close()
                return
            self._closed = True
            first_error: BaseException | None = None
            if self.execution_logging is not None:
                try:
                    self.execution_logging.close()
                except BaseException as error:
                    first_error = error
            try:
                self._release_logical_run_lease()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            try:
                self.close_process_group()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                raise first_error

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _require_open(self) -> None:
        """Reject new runtime resources after terminal cleanup."""
        if self._closed:
            raise RuntimeError("This torch runtime is closed")

    def _terminate_failed_logging_startup(
        self,
        logging_bundle: ExecutionLogging | None,
        startup_error: BaseException,
    ) -> None:
        """Make a failed logging transition terminal without hiding its cause."""
        cleanup_errors: list[tuple[str, BaseException]] = []
        if logging_bundle is not None:
            try:
                logging_bundle.close()
            except BaseException as error:
                cleanup_errors.append(("execution logging", error))
        try:
            self._release_logical_run_lease()
        except BaseException as error:
            cleanup_errors.append(("logical-run lease", error))
        try:
            self.close_process_group()
        except BaseException as error:
            cleanup_errors.append(("process group", error))
        self._closed = True
        for resource, cleanup_failure in cleanup_errors:
            startup_error.add_note(
                f"Later {resource} cleanup failure: {_error_text(cleanup_failure)}"
            )

    def _create_execution(self, spec: ExecutionSpec) -> ExecutionContext:
        """Coordinate strict creation while retaining the primary producer lease."""
        inherited_id = os.environ.get(EXECUTION_ID_ENV)
        self.startup_consensus(
            "execution creation preflight",
            None
            if inherited_id is None
            else ValueError(f"{EXECUTION_ID_ENV} must not be set when creating an execution"),
        )

        intent: _CreationIntent | None = None
        intent_error: BaseException | None = None
        try:
            intent = _CreationIntent(
                run_dir=str(spec.run_dir.resolve()),
                run_name=spec.run_name,
                invocation_kind=spec.invocation_kind,
                intended_phases=spec.intended_phases,
                command=sanitize_command(tuple(sys.argv)),
                config_reference=(
                    ""
                    if spec.config_reference == ""
                    else sanitize_reference(spec.config_reference)
                ),
                execution_id=spec.execution_id,
                resume_checkpoint=(
                    sanitize_reference(spec.resume_checkpoint)
                    if spec.resume_checkpoint is not None
                    else None
                ),
                resume_checkpoint_sha256=spec.resume_checkpoint_sha256,
                parent_execution_id=spec.parent_execution_id,
                starting_epoch=spec.starting_epoch,
                starting_global_step=spec.starting_global_step,
                runtime=self._expected_runtime_metadata(spec),
            )
        except BaseException as error:
            intent_error = error
        self.startup_consensus("execution creation intent", intent_error)
        if intent is None:
            raise RuntimeError("Execution creation intent validation lost its result")
        gathered_intents = self.all_gather_object(intent)
        if any(item != intent for item in gathered_intents):
            baseline = gathered_intents[0]
            field_names = (
                "run_dir",
                "run_name",
                "invocation_kind",
                "intended_phases",
                "command",
                "config_reference",
                "execution_id",
                "resume_checkpoint",
                "resume_checkpoint_sha256",
                "parent_execution_id",
                "starting_epoch",
                "starting_global_step",
                "runtime",
            )
            differences = [
                f"rank {rank}: "
                + ", ".join(
                    name
                    for name in field_names
                    if getattr(rank_intent, name) != getattr(baseline, name)
                )
                for rank, rank_intent in enumerate(gathered_intents)
                if rank_intent != baseline
            ]
            raise RuntimeError(
                "execution creation expectations are inconsistent across ranks; "
                + "; ".join(differences)
            )

        primary_result: _PrimaryResult | None = None
        if self.is_primary:
            try:
                run_dir = Path(intent.run_dir)
                self._logical_run_lease = claim_logical_run_lease(run_dir)
                previous_execution_id = latest_execution_id(run_dir)
                primary_context = create_execution_context(
                    run_dir,
                    run_name=intent.run_name,
                    invocation_kind=intent.invocation_kind,
                    intended_phases=intent.intended_phases,
                    world_size=self.world_size,
                    execution_mode="distributed" if self.enabled else "single",
                    command=intent.command,
                    config_reference=intent.config_reference,
                    execution_id=intent.execution_id,
                    previous_execution_id=previous_execution_id,
                    resume_checkpoint=intent.resume_checkpoint,
                    resume_checkpoint_sha256=intent.resume_checkpoint_sha256,
                    parent_execution_id=intent.parent_execution_id,
                    starting_epoch=intent.starting_epoch,
                    starting_global_step=intent.starting_global_step,
                    runtime=intent.runtime,
                )
                primary_result = _PrimaryResult(
                    execution_id=primary_context.metadata.execution_id,
                    error=None,
                )
            except BaseException as error:
                error_text = _error_text(error)
                try:
                    self._release_logical_run_lease()
                except BaseException as cleanup_error:
                    error_text += f"; lease cleanup failed: {_error_text(cleanup_error)}"
                primary_result = _PrimaryResult(execution_id=None, error=error_text)

        try:
            primary_result = self.broadcast_object(primary_result)
        except BaseException:
            self._release_logical_run_lease()
            raise
        if not isinstance(primary_result, _PrimaryResult):
            self._release_logical_run_lease()
            raise RuntimeError("Execution creation returned invalid primary data")
        if primary_result.error is not None or primary_result.execution_id is None:
            self._release_logical_run_lease()
            raise RuntimeError(
                "Execution creation failed on rank 0: "
                f"{primary_result.error or 'missing execution ID'}"
            )

        context: ExecutionContext | None = None
        local_error: BaseException | None = None
        try:
            context = join_execution_context(
                spec.run_dir,
                primary_result.execution_id,
                expected_run_name=spec.run_name,
            )
            self._validate_created_context(context, spec, intent)
        except BaseException as error:
            local_error = error
        try:
            self.startup_consensus("execution creation validation", local_error)
        except BaseException:
            self._release_logical_run_lease()
            raise
        if context is None:
            self._release_logical_run_lease()
            raise RuntimeError("Execution creation succeeded without a local context")
        return context

    def _attach_execution(self, expected: ExecutionSpec) -> ExecutionContext:
        """Join and exactly validate one canonical environment-selected attempt."""
        environment = self._canonical_join_environment(expected)
        context: ExecutionContext | None = None
        local_error: BaseException | None = None
        try:
            context = join_execution_context(
                expected.run_dir,
                environment.execution_id,
                expected_run_name=expected.run_name,
            )
            self._validate_attached_context(context, expected, environment)
        except BaseException as error:
            local_error = error
        self.startup_consensus("execution attachment", local_error)
        if context is None:
            raise RuntimeError("Execution attachment succeeded without a local context")
        return context

    def _canonical_join_environment(self, expected: ExecutionSpec) -> _JoinEnvironment:
        """Validate required canonical variables and their cross-rank consistency."""
        local_environment: _JoinEnvironment | None = None
        local_error: BaseException | None = None
        try:
            execution_id = execution_id_from_environment()
            if execution_id is None:
                raise ValueError(f"{EXECUTION_ID_ENV} is required when attaching an execution")
            values: dict[str, str] = {}
            for name in (RUN_NAME_ENV, INVOCATION_KIND_ENV, PHASE_ENV):
                value = os.environ.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} is required when attaching an execution")
                values[name] = value
            local_environment = _JoinEnvironment(
                execution_id=execution_id,
                run_name=values[RUN_NAME_ENV],
                invocation_kind=values[INVOCATION_KIND_ENV],
                phase=values[PHASE_ENV],
            )
        except BaseException as error:
            local_error = error
        self.startup_consensus("execution attachment environment", local_error)
        if local_environment is None:
            raise RuntimeError("Execution attachment environment validation lost its result")

        gathered = self.all_gather_object(local_environment)
        if any(item != local_environment for item in gathered):
            raise RuntimeError(
                "Canonical MAMMOTH execution environment is inconsistent across ranks"
            )
        intent: _AttachmentIntent | None = None
        intent_error: BaseException | None = None
        try:
            intent = _AttachmentIntent(
                environment=local_environment,
                run_dir=str(expected.run_dir.resolve()),
                run_name=expected.run_name,
                invocation_kind=expected.invocation_kind,
                intended_phases=expected.intended_phases,
                config_reference=(
                    ""
                    if expected.config_reference == ""
                    else sanitize_reference(expected.config_reference)
                ),
                execution_id=expected.execution_id,
                resume_checkpoint=(
                    sanitize_reference(expected.resume_checkpoint)
                    if expected.resume_checkpoint is not None
                    else None
                ),
                resume_checkpoint_sha256=expected.resume_checkpoint_sha256,
                parent_execution_id=expected.parent_execution_id,
                starting_epoch=expected.starting_epoch,
                starting_global_step=expected.starting_global_step,
                runtime=sanitize_metadata_fields(dict(expected.runtime)),
            )
        except BaseException as error:
            intent_error = error
        self.startup_consensus("execution attachment intent", intent_error)
        if intent is None:
            raise RuntimeError("Execution attachment intent validation lost its result")
        gathered_intents = self.all_gather_object(intent)
        if any(item != intent for item in gathered_intents):
            baseline = gathered_intents[0]
            field_names = (
                "environment",
                "run_dir",
                "run_name",
                "invocation_kind",
                "intended_phases",
                "config_reference",
                "execution_id",
                "resume_checkpoint",
                "resume_checkpoint_sha256",
                "parent_execution_id",
                "starting_epoch",
                "starting_global_step",
                "runtime",
            )
            differences = [
                f"rank {rank}: "
                + ", ".join(
                    name
                    for name in field_names
                    if getattr(rank_intent, name) != getattr(baseline, name)
                )
                for rank, rank_intent in enumerate(gathered_intents)
                if rank_intent != baseline
            ]
            raise RuntimeError(
                "execution attachment expectations are inconsistent across ranks; "
                + "; ".join(differences)
            )
        if local_environment.run_name != expected.run_name:
            raise ValueError(
                f"{RUN_NAME_ENV}={local_environment.run_name!r} does not match "
                f"expected run {expected.run_name!r}"
            )
        if local_environment.invocation_kind != expected.invocation_kind:
            raise ValueError(
                f"{INVOCATION_KIND_ENV}={local_environment.invocation_kind!r} does not "
                f"match expected invocation {expected.invocation_kind!r}"
            )
        if local_environment.phase not in expected.intended_phases:
            raise ValueError(
                f"{PHASE_ENV}={local_environment.phase!r} is not one of the expected phases"
            )
        if expected.execution_id is not None and (
            local_environment.execution_id != expected.execution_id
        ):
            raise ValueError(
                f"{EXECUTION_ID_ENV}={local_environment.execution_id!r} does not match "
                f"expected execution {expected.execution_id!r}"
            )
        return local_environment

    def _validate_created_context(
        self,
        context: ExecutionContext,
        spec: ExecutionSpec,
        intent: _CreationIntent,
    ) -> None:
        """Verify metadata published by strict creation against local intent."""
        metadata = context.metadata
        expected_mode = "distributed" if self.enabled else "single"
        if context.run_dir.resolve() != spec.run_dir.resolve():
            raise ValueError("Created execution resolved to an unexpected run directory")
        if metadata.run_name != spec.run_name:
            raise ValueError("Created execution has an unexpected run name")
        if metadata.invocation_kind != spec.invocation_kind:
            raise ValueError("Created execution has an unexpected invocation kind")
        if metadata.intended_phases != spec.intended_phases:
            raise ValueError("Created execution has unexpected intended phases")
        if metadata.world_size != self.world_size or metadata.execution_mode != expected_mode:
            raise ValueError("Created execution topology does not match the torch runtime")
        if metadata.command != intent.command:
            raise ValueError("Created execution did not record the agreed invocation command")
        if metadata.config_reference != intent.config_reference:
            raise ValueError("Created execution has an unexpected config reference")
        if metadata.to_dict()["runtime"] != intent.runtime:
            raise ValueError("Created execution has unexpected runtime provenance")
        if spec.execution_id is not None and metadata.execution_id != spec.execution_id:
            raise ValueError("Created execution has an unexpected execution ID")
        self._validate_resume_metadata(metadata, spec)

    def _validate_attached_context(
        self,
        context: ExecutionContext,
        expected: ExecutionSpec,
        environment: _JoinEnvironment,
    ) -> None:
        """Require exact topology, identity, phase, invocation, and resume facts."""
        metadata = context.metadata
        expected_mode = "distributed" if self.enabled else "single"
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
        if metadata.world_size != self.world_size:
            raise ValueError(
                f"Execution {metadata.execution_id!r} has world_size={metadata.world_size}, "
                f"but this runtime has world_size={self.world_size}"
            )
        if metadata.execution_mode != expected_mode:
            raise ValueError(
                f"Execution {metadata.execution_id!r} uses {metadata.execution_mode!r}, "
                f"but this runtime uses {expected_mode!r}"
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
            ""
            if expected.config_reference == ""
            else sanitize_reference(expected.config_reference)
        )
        if metadata.config_reference != expected_config_reference:
            raise ValueError("Attached execution has an unexpected config reference")
        expected_runtime = sanitize_metadata_fields(dict(expected.runtime))
        actual_runtime = sanitize_metadata_fields(dict(metadata.runtime or {}))
        if actual_runtime != expected_runtime:
            raise ValueError("Attached execution has unexpected runtime metadata")
        self._validate_resume_metadata(metadata, expected)

    def _expected_runtime_metadata(self, spec: ExecutionSpec) -> dict[str, Any]:
        """Return the sanitized caller and framework provenance stored at creation."""
        runtime_metadata = dict(spec.runtime)
        runtime_metadata.update(self.provenance())
        return sanitize_metadata_fields(runtime_metadata)

    @staticmethod
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

    def _validate_initialized_group(self) -> None:
        """Reject a preinitialized default group that disagrees with this runtime."""
        actual_rank = dist.get_rank()
        actual_world_size = dist.get_world_size()
        actual_backend = str(dist.get_backend())
        if (actual_rank, actual_world_size) != (self.rank, self.world_size):
            raise RuntimeError(
                "Initialized process group identity disagrees with the torch runtime: "
                f"group=({actual_rank}, {actual_world_size}), "
                f"runtime=({self.rank}, {self.world_size})"
            )
        if self.config.backend is not None and actual_backend != self.config.backend:
            raise RuntimeError(
                f"Initialized process group backend {actual_backend!r} does not match "
                f"requested backend {self.config.backend!r}"
            )
        self.backend = actual_backend

    def _resolve_workload_weights(self) -> tuple[float, ...]:
        weights = self.config.workload_weights
        if weights is None:
            return (1.0,) * self.world_size
        if len(weights) != self.world_size:
            raise RuntimeError(
                "workload_weights must contain one value per rank; "
                f"got {len(weights)} weights for world_size={self.world_size}"
            )
        return weights

    def _release_logical_run_lease(self) -> None:
        lease = self._logical_run_lease
        if lease is None:
            return
        self._logical_run_lease = None
        lease.close()

    def provenance(self) -> dict[str, Any]:
        """Return allowlisted framework facts safe for immutable execution metadata."""
        return {
            "framework": "pytorch",
            "framework_version": str(torch.__version__),
            "strategy": self.strategy,
            "backend": self.backend,
            "device_type": self.device.type,
        }


class ExecutionSession:
    """Compose neutral execution lifecycle with Torch trainer ownership.

    ``Runtime`` constructs this compatibility adapter after it has completed
    device and distributed startup.  The contained neutral session owns logging,
    lifecycle events, generic observers, and pipelines; this adapter keeps the
    optional Torch-only ``Trainer`` factory and closes those trainers first.
    """

    def __init__(self, runtime: Runtime) -> None:
        logging_bundle = runtime.execution_logging
        if logging_bundle is None:
            raise RuntimeError("Execution logging is required for an execution session")
        self.runtime = runtime
        self._neutral = NeutralExecutionSession.from_established(
            logging_bundle.context,
            logging_bundle,
            close_callbacks=(
                ("logical-run lease", lambda: runtime._release_logical_run_lease()),
                ("process group", lambda: runtime.close_process_group()),
                ("torch runtime", self._mark_runtime_closed),
            ),
        )
        self.context = self._neutral.context
        self.execution_logging = self._neutral.execution_logging
        self.observer = self._neutral.observer
        self.event_writer = self._neutral.event_writer
        self._owned_trainers = ExitStack()
        self._owned_resource_errors: list[tuple[str, BaseException]] = []
        self._resource_lock = RLock()
        self._closed = False

    @property
    def phase(self) -> str | None:
        """Return the active or most recently completed neutral phase name."""
        return self._neutral.phase

    @property
    def _phase_terminal(self) -> bool:
        """Expose the legacy internal state used by existing test diagnostics."""
        return self._neutral._phase_terminal

    def create_observer(
        self,
        sinks: Sequence[ObservationSink] = (),
    ) -> RunObserver:
        """Create a neutral observer owned by the contained lifecycle session."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
        return self._neutral.create_observer(sinks, observer_factory=RunObserver)

    def create_trainer(self, **kwargs: Any) -> Trainer:
        """Create and own one generic Torch trainer while borrowing its inputs."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
            trainer = Trainer(**kwargs)
            self._register_owned_resource(self._owned_trainers, "trainer", trainer.close)
            return trainer

    def create_background_pipeline[InputT, ResultT](
        self,
        worker: Callable[[InputT], ResultT],
        *,
        max_pending: int = 1,
        thread_name_prefix: str = "mammoth-background",
    ) -> BoundedBackgroundPipeline[InputT, ResultT]:
        """Create a neutral pipeline that closes after trainers and before observers."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
        return self._neutral.create_background_pipeline(
            worker,
            max_pending=max_pending,
            thread_name_prefix=thread_name_prefix,
            pipeline_factory=BoundedBackgroundPipeline,
        )

    def start_phase(self, phase: str) -> None:
        """Delegate process and phase start lifecycle emission to the neutral session."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
            self._neutral.start_phase(phase)

    @contextmanager
    def phase_scope(self, phase: str) -> Iterator[ExecutionSession]:
        """Own one phase transition while preserving the Torch compatibility type."""
        self.start_phase(phase)
        try:
            yield self
        except BaseException as error:
            self.fail_phase(error, interrupted=isinstance(error, KeyboardInterrupt))
            raise
        else:
            self.complete_phase()

    def complete_phase(self, *, message: str | None = None) -> None:
        """Mark the active neutral phase successful."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
            self._neutral.complete_phase(message=message)

    def fail_phase(self, error: BaseException, *, interrupted: bool = False) -> None:
        """Mark the active neutral phase failed or interrupted."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
            self._neutral.fail_phase(error, interrupted=interrupted)

    def skip_phase(self, message: str) -> None:
        """Mark the active neutral phase skipped."""
        with self.runtime._state_lock, self._resource_lock:
            self._require_open()
            self._neutral.skip_phase(message)

    def close(
        self,
        *,
        error: BaseException | None = None,
        exit_code: int | None = None,
        signal: int | str | None = None,
        message: str | None = None,
        before_close: Callable[[], None] | None = None,
    ) -> None:
        """Close Torch trainers before the neutral lifecycle and runtime finalizers."""
        with self.runtime._state_lock, self._resource_lock:
            if self._closed:
                return
            self._closed = True
            cleanup_errors: list[tuple[str, BaseException]] = []
            self._owned_resource_errors = cleanup_errors
            self._owned_trainers.close()
            self._neutral.close(
                error=error,
                exit_code=exit_code,
                signal=signal,
                message=message,
                before_close=before_close,
                prior_cleanup_errors=cleanup_errors,
            )

    def __enter__(self) -> ExecutionSession:
        """Return this open Torch compatibility session."""
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

    def _mark_runtime_closed(self) -> None:
        """Make the runtime terminal after neutral lifecycle finalization."""
        self.runtime._closed = True

    def _register_owned_resource(
        self,
        stack: ExitStack,
        label: str,
        close: Callable[[], None],
    ) -> None:
        """Register one Torch-only close callback in reverse creation order."""
        stack.callback(self._close_owned_resource, label, close)

    def _close_owned_resource(self, label: str, close: Callable[[], None]) -> None:
        """Capture a trainer cleanup error without skipping later cleanup stages."""
        try:
            close()
        except BaseException as error:
            self._owned_resource_errors.append((label, error))

    def _require_open(self) -> None:
        if self._closed or self.runtime._closed:
            raise RuntimeError("Execution session is already closed")


def initialize_runtime(
    config: RuntimeConfig | None = None,
) -> Runtime:
    """Initialize and return one generic PyTorch execution runtime."""
    return Runtime(config)


def _distributed_identity(config: RuntimeConfig) -> tuple[int, int, int]:
    if config.rank is not None and config.world_size is not None:
        rank = config.rank
        world_size = config.world_size
    elif dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = _environment_integer("RANK")
        world_size = _environment_integer("WORLD_SIZE")
    local_rank = (
        config.local_rank
        if config.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", str(rank)))
    )
    if rank >= world_size:
        raise RuntimeError(f"RANK={rank} must be smaller than WORLD_SIZE={world_size}")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    return rank, local_rank, world_size


def _distributed_device(value: str, local_rank: int) -> torch.device:
    if value == "auto":
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    device = resolve_device(value)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if device.index is None else device.index)
        if device.index != local_rank:
            raise RuntimeError(
                f"Configured CUDA device index {device.index} disagrees with "
                f"LOCAL_RANK={local_rank}"
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is unavailable; torch sees "
                f"{torch.cuda.device_count()} CUDA devices"
            )
    return device


def _environment_integer(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"DDP strategy requires {name} in the environment")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from error
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative, got {value}")
    return value


def _environment_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise RuntimeError(f"WORLD_SIZE must be an integer, got {raw!r}") from error
    if world_size <= 0:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")
    return world_size


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
