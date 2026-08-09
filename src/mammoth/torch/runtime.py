"""Own generic single-process and standard PyTorch DDP execution state.

CLI entrypoints construct this runtime before project code. The generic trainer
consumes its device and rank identity, while execution logging uses it to create
or join one immutable attempt and open one process-owned stream per rank.
"""

from __future__ import annotations

import logging
import math
import os
import signal as signal_module
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Literal, cast

import torch
import torch.distributed as dist

from mammoth.core import (
    BackgroundPipelineError,
    BoundedBackgroundPipeline,
    ExecutionContext,
    LogicalRunLease,
    claim_logical_run_lease,
    create_execution_context,
    execution_id_from_environment,
    join_execution_context,
    normalize_execution_id_environment_aliases,
    sanitize_reference,
    validate_resume_checkpoint_sha256,
)
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
class ExecutionRequest:
    """Project-neutral facts needed to establish one immutable execution.

    Consumers may declare temporary execution-ID environment aliases here; the
    runtime passes them to Mammoth's generic resolver without persisting them.
    """

    run_dir: Path
    run_name: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    command: tuple[str, ...]
    config_reference: str | Path = ""
    execution_id: str | None = None
    previous_execution_id: str | None = None
    resume_checkpoint: str | Path | None = None
    resume_checkpoint_sha256: str | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    execution_id_environment_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "intended_phases", tuple(self.intended_phases))
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "runtime", dict(self.runtime))
        if self.resume_checkpoint is not None:
            if self.resume_checkpoint_sha256 is None:
                raise ValueError("resume_checkpoint requires resume_checkpoint_sha256.")
            if self.starting_epoch is None:
                raise ValueError("resume_checkpoint requires starting_epoch.")
        if self.resume_checkpoint_sha256 is not None:
            validate_resume_checkpoint_sha256(self.resume_checkpoint_sha256)
        object.__setattr__(
            self,
            "execution_id_environment_aliases",
            normalize_execution_id_environment_aliases(self.execution_id_environment_aliases),
        )


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


class Runtime:
    """Own one process's generic PyTorch identity, collectives, and execution IO."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.strategy = self.config.strategy
        self._owns_process_group = False
        self._logical_run_lease: LogicalRunLease | None = None
        self.execution_logging: ExecutionLogging | None = None
        self.execution_context: ExecutionContext | None = None
        self._execution_session: ExecutionSession | None = None
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

    def start_execution(
        self,
        request: ExecutionRequest,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> ExecutionLogging:
        """Create or join one attempt and establish rank-local logging by consensus."""
        if self.execution_logging is not None:
            raise RuntimeError("This torch runtime has already started an execution")
        self.establish_execution(request)
        return self.start_execution_logging(
            additional_sinks=additional_sinks,
            text_level=text_level,
        )

    def start_execution_logging(
        self,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> ExecutionLogging:
        """Open rank-local logging after execution establishment and validation."""
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
        except BaseException:
            if logging_bundle is not None:
                logging_bundle.close()
            self._release_logical_run_lease()
            raise
        if logging_bundle is None:
            self._release_logical_run_lease()
            raise RuntimeError("Execution logging startup succeeded without a local bundle")
        self.execution_context = context
        self.execution_logging = logging_bundle
        return logging_bundle

    def create_execution_session(self) -> ExecutionSession:
        """Create the generic process/phase lifecycle owner for this runtime."""
        if self._execution_session is not None:
            raise RuntimeError("This torch runtime already has an execution session")
        if self.execution_logging is None:
            raise RuntimeError("Start execution logging before creating a session")
        session = ExecutionSession(self)
        self._execution_session = session
        return session

    def establish_execution(self, request: ExecutionRequest) -> ExecutionContext:
        """Create or join and validate one immutable execution across all ranks."""
        if self.execution_context is not None:
            raise RuntimeError("This torch runtime has already established an execution")
        context = self._establish_execution(request)
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
        if self._closed:
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

    def _establish_execution(self, request: ExecutionRequest) -> ExecutionContext:
        """Publish on rank zero, then make every rank join and validate one attempt."""
        primary_result: _PrimaryResult | None = None
        if self.is_primary:
            try:
                provided_execution_id = execution_id_from_environment(
                    aliases=request.execution_id_environment_aliases
                )
                if provided_execution_id is not None:
                    primary_context = join_execution_context(
                        request.run_dir,
                        provided_execution_id,
                        expected_run_name=request.run_name,
                    )
                else:
                    self._logical_run_lease = claim_logical_run_lease(request.run_dir)
                    runtime_metadata = dict(request.runtime)
                    runtime_metadata.update(self.provenance())
                    primary_context = create_execution_context(
                        request.run_dir,
                        run_name=request.run_name,
                        invocation_kind=request.invocation_kind,
                        intended_phases=request.intended_phases,
                        world_size=self.world_size,
                        execution_mode="distributed" if self.enabled else "single",
                        command=request.command,
                        config_reference=request.config_reference,
                        execution_id=request.execution_id,
                        previous_execution_id=request.previous_execution_id,
                        resume_checkpoint=request.resume_checkpoint,
                        resume_checkpoint_sha256=request.resume_checkpoint_sha256,
                        parent_execution_id=request.parent_execution_id,
                        starting_epoch=request.starting_epoch,
                        starting_global_step=request.starting_global_step,
                        runtime=runtime_metadata,
                    )
                primary_result = _PrimaryResult(
                    execution_id=primary_context.metadata.execution_id,
                    error=None,
                )
            except Exception as error:
                self._release_logical_run_lease()
                primary_result = _PrimaryResult(execution_id=None, error=_error_text(error))

        primary_result = self.broadcast_object(primary_result)
        if not isinstance(primary_result, _PrimaryResult):
            self._release_logical_run_lease()
            raise RuntimeError("Execution startup returned invalid primary data")
        if primary_result.error is not None or primary_result.execution_id is None:
            self._release_logical_run_lease()
            raise RuntimeError(
                "Execution initialization failed on rank 0: "
                f"{primary_result.error or 'missing execution ID'}"
            )

        context: ExecutionContext | None = None
        local_error: BaseException | None = None
        try:
            context = join_execution_context(
                request.run_dir,
                primary_result.execution_id,
                expected_run_name=request.run_name,
            )
            self._validate_context(context, request)
        except BaseException as error:
            local_error = error
        try:
            self.startup_consensus("execution join", local_error)
        except BaseException:
            self._release_logical_run_lease()
            raise
        if context is None:
            self._release_logical_run_lease()
            raise RuntimeError("Execution join succeeded without a local context")
        return context

    def _validate_context(
        self,
        context: ExecutionContext,
        request: ExecutionRequest,
    ) -> None:
        metadata = context.metadata
        expected_mode = "distributed" if self.enabled else "single"
        workflow_child = metadata.invocation_kind == "workflow"
        if not workflow_child and metadata.world_size != self.world_size:
            raise ValueError(
                f"Execution {metadata.execution_id!r} has world_size={metadata.world_size}, "
                f"but this runtime has world_size={self.world_size}"
            )
        if not workflow_child and metadata.execution_mode != expected_mode:
            raise ValueError(
                f"Execution {metadata.execution_id!r} uses {metadata.execution_mode!r}, "
                f"but this runtime uses {expected_mode!r}"
            )
        missing_phases = set(request.intended_phases).difference(metadata.intended_phases)
        if missing_phases:
            raise ValueError(
                f"Execution {metadata.execution_id!r} omits phases: "
                f"{', '.join(sorted(missing_phases))}"
            )
        if request.resume_checkpoint is None:
            return
        expected_facts = {
            "resume_checkpoint": sanitize_reference(request.resume_checkpoint),
            "resume_checkpoint_sha256": request.resume_checkpoint_sha256,
            "parent_execution_id": request.parent_execution_id,
            "starting_epoch": request.starting_epoch,
            "starting_global_step": request.starting_global_step,
        }
        for field_name, expected_value in expected_facts.items():
            actual_value = getattr(metadata, field_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"Execution {metadata.execution_id!r} resume field {field_name!r} "
                    f"does not match: metadata={actual_value!r}, request={expected_value!r}."
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
    """Own process/phase lifecycle plus resources created through this session."""

    def __init__(self, runtime: Runtime) -> None:
        logging_bundle = runtime.execution_logging
        if logging_bundle is None:
            raise RuntimeError("Execution logging is required for an execution session")
        self.runtime = runtime
        self.context = logging_bundle.context
        self.execution_logging = logging_bundle
        self.observer = logging_bundle.observer
        self.event_writer = logging_bundle.event_writer
        self._phase: str | None = None
        self._phase_terminal = False
        self._phase_outcome: Literal["completed", "failed", "interrupted", "skipped"] | None = None
        self._process_started_at: float | None = None
        self._phase_started_at: float | None = None
        self._owned_resources = ExitStack()
        self._owned_observers = ExitStack()
        self._owned_pipelines = ExitStack()
        self._owned_trainers = ExitStack()
        self._owned_resources.callback(self._owned_observers.close)
        self._owned_resources.callback(self._owned_pipelines.close)
        self._owned_resources.callback(self._owned_trainers.close)
        self._owned_resource_errors: list[tuple[str, BaseException]] = []
        self._resource_lock = RLock()
        self._closed = False

    @property
    def phase(self) -> str | None:
        """Return the active or most recently completed phase name."""
        return self._phase

    def create_observer(
        self,
        sinks: Sequence[ObservationSink] = (),
    ) -> RunObserver:
        """Create and own one sink-neutral observer for project training output."""
        self._require_open()
        observer = RunObserver(sinks)
        self._register_owned_resource(self._owned_observers, "observer", observer.close)
        return observer

    def create_trainer(self, **kwargs: Any) -> Trainer:
        """Create and own one generic Trainer while borrowing all supplied inputs."""
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
        """Create a pipeline that closes after trainers and before observers."""
        self._require_open()
        pipeline = BoundedBackgroundPipeline(
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
        if self._closed:
            raise RuntimeError("Cannot start a phase after execution-session closure")
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
    def phase_scope(self, phase: str) -> Iterator[ExecutionSession]:
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

    def skip_phase(self, message: str) -> None:
        """Mark the active phase skipped."""
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
    ) -> None:
        """Close owned resources, lifecycle logging, leases, and owned runtime state."""
        with self._resource_lock:
            if self._closed:
                return
            self._closed = True
        cleanup_errors: list[tuple[str, BaseException]] = []
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
                    self.fail_phase(
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
            if self.runtime.execution_logging is not None:
                self.runtime.execution_logging.close()
        except BaseException as logging_error:
            cleanup_errors.append(("execution logging", logging_error))
        try:
            self.runtime._release_logical_run_lease()
        except BaseException as lease_error:
            cleanup_errors.append(("logical-run lease", lease_error))
        try:
            self.runtime.close_process_group()
        except BaseException as process_group_error:
            cleanup_errors.append(("process group", process_group_error))
        self.runtime._closed = True
        if error is not None:
            _attach_cleanup_errors(error, cleanup_errors)
        elif cleanup_errors and exit_code in {None, 0}:
            first_label, first_error = cleanup_errors[0]
            _attach_cleanup_errors(first_error, cleanup_errors[1:])
            first_error.add_note(f"Cleanup stage: {first_label}")
            raise first_error

    def __enter__(self) -> ExecutionSession:
        """Return this open execution session."""
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session without replacing the workload exception."""
        del exc_type, traceback
        self.close(error=exc_value)

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
