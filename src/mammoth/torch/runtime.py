"""Own generic single-process and standard PyTorch DDP execution state.

CLI entrypoints construct this runtime before project code. The generic trainer
consumes its device and rank identity, while execution logging uses it to create
or join one immutable attempt and open one process-owned stream per rank.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist

from mammoth.core import (
    ExecutionContext,
    LogicalRunLease,
    claim_logical_run_lease,
    create_execution_context,
    execution_id_from_environment,
    join_execution_context,
)
from mammoth.logging import ExecutionLogging, ObservationSink, create_execution_logging

Strategy = Literal["single", "ddp"]


@dataclass(frozen=True, slots=True)
class TorchRuntimeConfig:
    """Framework-level process-group and device policy."""

    strategy: Strategy = "single"
    device: str = "auto"
    backend: str | None = None
    init_method: str = "env://"
    timeout_seconds: float = 1800.0
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None

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


@dataclass(frozen=True, slots=True)
class TorchExecutionRequest:
    """Project-neutral facts needed to establish one immutable execution."""

    run_dir: Path
    run_name: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    command: tuple[str, ...]
    config_reference: str | Path = ""
    execution_id: str | None = None
    previous_execution_id: str | None = None
    resume_checkpoint: str | Path | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "intended_phases", tuple(self.intended_phases))
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "runtime", dict(self.runtime))


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


class TorchExecutionRuntime:
    """Own one process's generic PyTorch identity, collectives, and execution IO."""

    def __init__(self, config: TorchRuntimeConfig | None = None) -> None:
        self.config = config or TorchRuntimeConfig()
        self.strategy = self.config.strategy
        self._owns_process_group = False
        self._logical_run_lease: LogicalRunLease | None = None
        self.execution_logging: ExecutionLogging | None = None
        self.execution_context: ExecutionContext | None = None
        self._closed = False

        if self.strategy == "single":
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = resolve_device(self.config.device)
            self.backend = None
            return

        self.rank, self.local_rank, self.world_size = _distributed_identity(self.config)
        if self.world_size < 2:
            raise RuntimeError("DDP strategy requires a world size of at least two")
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
        request: TorchExecutionRequest,
        *,
        additional_sinks: Sequence[ObservationSink] = (),
        text_level: int = logging.INFO,
    ) -> ExecutionLogging:
        """Create or join one attempt and establish rank-local logging by consensus."""
        if self.execution_logging is not None:
            raise RuntimeError("This torch runtime has already started an execution")
        context = self.establish_execution(request)
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

    def establish_execution(self, request: TorchExecutionRequest) -> ExecutionContext:
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

    def __enter__(self) -> TorchExecutionRuntime:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _establish_execution(self, request: TorchExecutionRequest) -> ExecutionContext:
        """Publish on rank zero, then make every rank join and validate one attempt."""
        primary_result: _PrimaryResult | None = None
        if self.is_primary:
            try:
                provided_execution_id = execution_id_from_environment()
                if provided_execution_id is not None:
                    primary_context = join_execution_context(
                        request.run_dir,
                        provided_execution_id,
                        expected_run_name=request.run_name,
                    )
                    self._validate_context(primary_context, request)
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
        request: TorchExecutionRequest,
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


def initialize_torch_runtime(
    config: TorchRuntimeConfig | None = None,
) -> TorchExecutionRuntime:
    """Initialize and return one generic PyTorch execution runtime."""
    return TorchExecutionRuntime(config)


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` or one explicit torch device string."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def _distributed_identity(config: TorchRuntimeConfig) -> tuple[int, int, int]:
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


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
