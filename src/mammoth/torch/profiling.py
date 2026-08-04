"""Model-independent callable profiling for constructed PyTorch workloads.

Consuming projects supply a zero-argument callable and any explicitly named
component modules. This module owns synchronized timing, allocator evidence,
operation profiling, reversible runtime controls, and versioned reports without
assuming a model factory, input contract, or output meaning.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.hooks import RemovableHandle

from mammoth.core import atomic_write_json
from mammoth.torch.runtime import resolve_device

PROFILE_SCHEMA_VERSION = 1

type MatmulPrecision = Literal["highest", "high", "medium"]
type JsonScalar = str | int | float | bool | None
type FrozenJson = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]
type OutputSummarizer = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Project-neutral policy for profiling one callable workload."""

    device: str = "auto"
    warmup_iterations: int = 3
    measured_iterations: int = 10
    profiler_iterations: int = 1
    work_units_per_iteration: float = 1.0
    work_unit: str = "item"
    profile_operations: bool = True
    record_shapes: bool = True
    profile_memory: bool = True
    with_flops: bool = True
    row_limit: int = 20
    sort_by: str | None = None
    chrome_trace: Path | None = None

    def __post_init__(self) -> None:
        _nonnegative_integer("warmup_iterations", self.warmup_iterations)
        _positive_integer("measured_iterations", self.measured_iterations)
        _positive_integer("profiler_iterations", self.profiler_iterations)
        _positive_integer("row_limit", self.row_limit)
        if (
            isinstance(self.work_units_per_iteration, bool)
            or not isinstance(self.work_units_per_iteration, int | float)
            or not math.isfinite(self.work_units_per_iteration)
            or self.work_units_per_iteration <= 0
        ):
            raise ValueError("work_units_per_iteration must be positive and finite")
        if not isinstance(self.work_unit, str) or not self.work_unit.strip():
            raise ValueError("work_unit must be a non-empty string")
        if self.sort_by is not None and not self.sort_by.strip():
            raise ValueError("sort_by must be non-empty when provided")
        if self.chrome_trace is not None:
            object.__setattr__(self, "chrome_trace", Path(self.chrome_trace))
            if not self.profile_operations:
                raise ValueError("chrome_trace requires profile_operations=True")


@dataclass(frozen=True, slots=True)
class TorchRuntimeOptions:
    """Optional process-global Torch settings applied only during profiling.

    When both matmul precision APIs are requested, the legacy TF32 boolean has
    precedence and is expressed through the newer precision API. This avoids a
    PyTorch mixed-API state whose precision getter raises at report time.
    """

    matmul_precision: MatmulPrecision | None = None
    cuda_matmul_allow_tf32: bool | None = None
    cudnn_allow_tf32: bool | None = None
    cudnn_benchmark: bool | None = None
    cudnn_deterministic: bool | None = None
    deterministic_algorithms: bool | None = None
    deterministic_warn_only: bool = False

    def __post_init__(self) -> None:
        if self.matmul_precision not in {None, "highest", "high", "medium"}:
            raise ValueError(f"Unsupported matmul precision: {self.matmul_precision!r}")


@dataclass(frozen=True, slots=True)
class TorchRuntimeState:
    """Effective Torch backend state used for one profile."""

    matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool


@dataclass(frozen=True, slots=True)
class ProfileTiming:
    """Synchronized wall and optional CUDA-device time for one invocation."""

    wall_ms: float
    device_ms: float | None


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution summary for repeated latency samples in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class ThroughputSummary:
    """Caller-labelled throughput derived from synchronized wall time."""

    work_units_per_iteration: float
    work_unit: str
    work_units_per_second: float


@dataclass(frozen=True, slots=True)
class CudaMemoryStats:
    """CUDA allocator state and peaks captured after steady-state measurement."""

    device_name: str
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class OperationProfile:
    """Normalized aggregate for one operation emitted by ``torch.profiler``."""

    key: str
    count: int
    cpu_time_total_us: float
    self_cpu_time_total_us: float
    device_time_total_us: float
    self_device_time_total_us: float
    cpu_memory_usage_bytes: int
    self_cpu_memory_usage_bytes: int
    device_memory_usage_bytes: int
    self_device_memory_usage_bytes: int
    flops: int
    input_shapes: tuple[tuple[int | str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Immutable versioned evidence returned by :func:`profile_callable`."""

    schema_version: int
    device: str
    cold_start: ProfileTiming
    wall_latency: LatencySummary
    device_latency: LatencySummary | None
    throughput: ThroughputSummary
    memory: CudaMemoryStats | None
    output_summary: FrozenJson
    operations_profiled: bool
    top_operations: tuple[OperationProfile, ...]
    runtime: TorchRuntimeState
    warmup_iterations: int
    measured_iterations: int
    profiler_iterations: int
    trace_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of this report."""
        return {
            "schema_version": self.schema_version,
            "device": self.device,
            "cold_start": asdict(self.cold_start),
            "wall_latency": asdict(self.wall_latency),
            "device_latency": (
                None if self.device_latency is None else asdict(self.device_latency)
            ),
            "throughput": asdict(self.throughput),
            "memory": None if self.memory is None else asdict(self.memory),
            "output_summary": _thaw_json(self.output_summary),
            "operations_profiled": self.operations_profiled,
            "top_operations": [asdict(operation) for operation in self.top_operations],
            "runtime": asdict(self.runtime),
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "profiler_iterations": self.profiler_iterations,
            "trace_path": self.trace_path,
        }


def profile_callable(
    workload: Callable[[], Any],
    *,
    config: ProfileConfig | None = None,
    components: Mapping[str, torch.nn.Module] | None = None,
    summarize_output: OutputSummarizer | None = None,
    runtime_options: TorchRuntimeOptions | None = None,
) -> ProfileReport:
    """Profile an arbitrary caller-owned PyTorch workload.

    The callable owns model mode, inference/autograd, autocast, inputs, keyword
    arguments, and output semantics. Its first result is summarized outside the
    timed region; subsequent results are discarded.
    """
    if not callable(workload):
        raise TypeError("workload must be callable")
    selected_config = config or ProfileConfig()
    selected_components = _validate_components(components or {})
    device = resolve_device(selected_config.device)
    options = runtime_options or TorchRuntimeOptions()

    with torch_runtime_options(options):
        runtime = current_torch_runtime_state()
        output, cold_start = _measure_invocation(device, workload)
        summary = (
            summarize_output(output)
            if summarize_output is not None
            else summarize_output_value(output)
        )
        frozen_summary = _freeze_json(summary, context="output summary")

        for _ in range(selected_config.warmup_iterations):
            workload()
        _synchronize(device)

        _reset_memory_stats(device)
        wall_samples: list[float] = []
        device_samples: list[float] = []
        for _ in range(selected_config.measured_iterations):
            _, timing = _measure_invocation(device, workload)
            wall_samples.append(timing.wall_ms)
            if timing.device_ms is not None:
                device_samples.append(timing.device_ms)
        memory = _collect_memory_stats(device)

        top_operations: tuple[OperationProfile, ...] = ()
        trace_path: str | None = None
        if selected_config.profile_operations:
            profiler = _profile_operations(
                workload,
                device=device,
                iterations=selected_config.profiler_iterations,
                components=selected_components,
                config=selected_config,
            )
            top_operations = _top_operations(
                profiler,
                device=device,
                row_limit=selected_config.row_limit,
                sort_by=selected_config.sort_by,
                record_shapes=selected_config.record_shapes,
            )
            if selected_config.chrome_trace is not None:
                selected_config.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(selected_config.chrome_trace))
                trace_path = str(selected_config.chrome_trace)

    wall_latency = summarize_latency(wall_samples)
    mean_seconds = wall_latency.mean_ms / 1000.0
    throughput = ThroughputSummary(
        work_units_per_iteration=float(selected_config.work_units_per_iteration),
        work_unit=selected_config.work_unit,
        work_units_per_second=(
            0.0
            if mean_seconds == 0.0
            else float(selected_config.work_units_per_iteration) / mean_seconds
        ),
    )
    return ProfileReport(
        schema_version=PROFILE_SCHEMA_VERSION,
        device=str(device),
        cold_start=cold_start,
        wall_latency=wall_latency,
        device_latency=summarize_latency(device_samples) if device_samples else None,
        throughput=throughput,
        memory=memory,
        output_summary=frozen_summary,
        operations_profiled=selected_config.profile_operations,
        top_operations=top_operations,
        runtime=runtime,
        warmup_iterations=selected_config.warmup_iterations,
        measured_iterations=selected_config.measured_iterations,
        profiler_iterations=selected_config.profiler_iterations,
        trace_path=trace_path,
    )


def write_profile_report(path: Path, report: ProfileReport) -> Path:
    """Atomically publish one profile report as deterministic JSON."""
    if not isinstance(report, ProfileReport):
        raise TypeError("report must be a ProfileReport")
    return atomic_write_json(Path(path), report.to_dict())


def summarize_latency(values: list[float]) -> LatencySummary:
    """Return a finite latency distribution for one nonempty sample list."""
    if not values:
        raise ValueError("latency samples must be non-empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    ordered = sorted(values)
    return LatencySummary(
        mean_ms=float(statistics.fmean(ordered)),
        median_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        min_ms=float(ordered[0]),
        max_ms=float(ordered[-1]),
    )


def summarize_output_value(value: Any) -> Mapping[str, Any]:
    """Return project-neutral metadata for tensors and nested containers."""
    return _summarize_value(value)


@contextmanager
def torch_runtime_options(options: TorchRuntimeOptions) -> Iterator[None]:
    """Apply Torch backend settings and restore all prior values on exit."""
    if not isinstance(options, TorchRuntimeOptions):
        raise TypeError("options must be TorchRuntimeOptions")
    previous = current_torch_runtime_state()
    matmul_precision = _normalized_matmul_precision(options, previous)
    try:
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)
        if options.cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = options.cudnn_allow_tf32
        if options.cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = options.cudnn_benchmark
        if options.cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = options.cudnn_deterministic
        if options.deterministic_algorithms is not None:
            torch.use_deterministic_algorithms(
                options.deterministic_algorithms,
                warn_only=options.deterministic_warn_only,
            )
        yield
    finally:
        torch.set_float32_matmul_precision(previous.matmul_precision)
        torch.backends.cudnn.allow_tf32 = previous.cudnn_allow_tf32
        torch.backends.cudnn.benchmark = previous.cudnn_benchmark
        torch.backends.cudnn.deterministic = previous.cudnn_deterministic
        torch.use_deterministic_algorithms(
            previous.deterministic_algorithms,
            warn_only=previous.deterministic_warn_only,
        )


def current_torch_runtime_state() -> TorchRuntimeState:
    """Capture the process-global Torch settings relevant to profiling."""
    return TorchRuntimeState(
        matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _measure_invocation(
    device: torch.device,
    workload: Callable[[], Any],
) -> tuple[Any, ProfileTiming]:
    use_cuda = device.type == "cuda"
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    if use_cuda:
        _synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start_event.record(torch.cuda.current_stream(device))
    started = time.perf_counter()
    result = workload()
    device_ms: float | None = None
    if start_event is not None and end_event is not None:
        end_event.record(torch.cuda.current_stream(device))
        _synchronize(device)
        device_ms = float(start_event.elapsed_time(end_event))
    wall_ms = (time.perf_counter() - started) * 1000.0
    return result, ProfileTiming(wall_ms=wall_ms, device_ms=device_ms)


def _profile_operations(
    workload: Callable[[], Any],
    *,
    device: torch.device,
    iterations: int,
    components: Mapping[str, torch.nn.Module],
    config: ProfileConfig,
) -> profile:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profiler_context = profile(
        activities=activities,
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_flops=config.with_flops,
    )
    with profiler_context, _component_ranges(components), record_function("profiled_workload"):
        for _ in range(iterations):
            workload()
        _synchronize(device)
    return profiler_context


@contextmanager
def _component_ranges(components: Mapping[str, torch.nn.Module]) -> Iterator[None]:
    handles: list[RemovableHandle] = []
    active_ranges: list[list[Any]] = []
    for name, module in components.items():
        stack: list[Any] = []
        active_ranges.append(stack)

        def pre_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            *,
            range_name: str = name,
            range_stack: list[Any] = stack,
        ) -> None:
            context = record_function(f"component:{range_name}")
            context.__enter__()  # type: ignore[no-untyped-call]
            range_stack.append(context)

        def post_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            _output: Any,
            *,
            range_stack: list[Any] = stack,
        ) -> None:
            if range_stack:
                range_stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    try:
        yield
    finally:
        for stack in active_ranges:
            while stack:
                stack.pop().__exit__(None, None, None)
        for handle in handles:
            handle.remove()


def _top_operations(
    profiler: profile,
    *,
    device: torch.device,
    row_limit: int,
    sort_by: str | None,
    record_shapes: bool = False,
) -> tuple[OperationProfile, ...]:
    selected_sort = sort_by or ("device_time_total" if device.type == "cuda" else "cpu_time_total")
    rows = list(
        profiler.key_averages(group_by_input_shape=True)
        if record_shapes
        else profiler.key_averages()
    )
    rows.sort(key=lambda row: _row_sort_value(row, selected_sort), reverse=True)
    return tuple(
        OperationProfile(
            key=str(row.key),
            count=int(getattr(row, "count", 0)),
            cpu_time_total_us=_row_number(row, "cpu_time_total"),
            self_cpu_time_total_us=_row_number(row, "self_cpu_time_total"),
            device_time_total_us=_row_number(row, "device_time_total", "cuda_time_total"),
            self_device_time_total_us=_row_number(
                row, "self_device_time_total", "self_cuda_time_total"
            ),
            cpu_memory_usage_bytes=int(getattr(row, "cpu_memory_usage", 0)),
            self_cpu_memory_usage_bytes=int(getattr(row, "self_cpu_memory_usage", 0)),
            device_memory_usage_bytes=int(
                getattr(row, "device_memory_usage", getattr(row, "cuda_memory_usage", 0))
            ),
            self_device_memory_usage_bytes=int(
                getattr(
                    row,
                    "self_device_memory_usage",
                    getattr(row, "self_cuda_memory_usage", 0),
                )
            ),
            flops=int(getattr(row, "flops", 0) or 0),
            input_shapes=_input_shapes(row),
        )
        for row in rows[:row_limit]
    )


def _normalized_matmul_precision(
    options: TorchRuntimeOptions,
    previous: TorchRuntimeState,
) -> MatmulPrecision | None:
    precision = options.matmul_precision
    allow_tf32 = options.cuda_matmul_allow_tf32
    if allow_tf32 is None:
        return precision
    if not allow_tf32:
        return "highest"
    if precision == "medium" or (precision is None and previous.matmul_precision == "medium"):
        return "medium"
    return "high"


def _input_shapes(row: Any) -> tuple[tuple[int | str, ...], ...]:
    raw_shapes = getattr(row, "input_shapes", ())
    if not isinstance(raw_shapes, tuple | list):
        return ()
    shapes: list[tuple[int | str, ...]] = []
    for raw_shape in raw_shapes:
        if not isinstance(raw_shape, tuple | list):
            continue
        shapes.append(
            tuple(
                int(dimension)
                if isinstance(dimension, int) and not isinstance(dimension, bool)
                else str(dimension)
                for dimension in raw_shape
            )
        )
    return tuple(shapes)


def _row_number(row: Any, name: str, fallback: str | None = None) -> float:
    value = getattr(row, name, None)
    if value is None and fallback is not None:
        value = getattr(row, fallback, 0.0)
    return float(value or 0.0)


def _row_sort_value(row: Any, name: str) -> float:
    fallbacks = {
        "device_time_total": "cuda_time_total",
        "self_device_time_total": "self_cuda_time_total",
        "device_memory_usage": "cuda_memory_usage",
        "self_device_memory_usage": "self_cuda_memory_usage",
        "cuda_time_total": "device_time_total",
        "self_cuda_time_total": "self_device_time_total",
        "cuda_memory_usage": "device_memory_usage",
        "self_cuda_memory_usage": "self_device_memory_usage",
    }
    return _row_number(row, name, fallbacks.get(name))


def _reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _collect_memory_stats(device: torch.device) -> CudaMemoryStats | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return CudaMemoryStats(
        device_name=torch.cuda.get_device_name(index),
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        max_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        max_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_components(
    components: Mapping[str, torch.nn.Module],
) -> Mapping[str, torch.nn.Module]:
    validated: dict[str, torch.nn.Module] = {}
    for name, module in components.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component names must be non-empty strings")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"component {name!r} must be a torch.nn.Module")
        validated[name] = module
    return MappingProxyType(validated)


def _summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        summary: dict[str, Any] = {
            "kind": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "layout": str(detached.layout),
            "numel": detached.numel(),
            "requires_grad": value.requires_grad,
        }
        if detached.layout == torch.strided:
            summary["stride"] = list(detached.stride())
        if detached.layout == torch.strided and detached.numel() > 0 and (
            torch.is_floating_point(detached) or torch.is_complex(detached)
        ):
            finite = bool(torch.isfinite(detached).all().item())
            summary["finite"] = finite
            if finite:
                magnitude = detached.abs().to(dtype=torch.float64)
                summary["mean_abs"] = float(magnitude.mean().item())
                summary["l1_sum"] = float(magnitude.sum().item())
        return summary
    if isinstance(value, Mapping):
        values: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key)
            if normalized in values:
                raise ValueError(f"output mapping keys collide after string conversion: {key!r}")
            values[normalized] = _summarize_value(item)
        return {"kind": "mapping", "values": values}
    if isinstance(value, tuple | list):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "values": [_summarize_value(item) for item in value],
        }
    if isinstance(value, str | int | bool) or value is None:
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        return {
            "kind": "float",
            "value": value if math.isfinite(value) else None,
            "finite": math.isfinite(value),
        }
    return {"kind": type(value).__name__}


def _freeze_json(value: Any, *, context: str) -> FrozenJson:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            frozen[key] = _freeze_json(item, context=context)
        return MappingProxyType(frozen)
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item, context=context) for item in value)
    raise TypeError(f"{context} contains unsupported value {type(value).__name__}")


def _thaw_json(value: FrozenJson) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _positive_integer(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
