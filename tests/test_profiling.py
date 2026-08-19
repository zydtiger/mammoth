"""Tests for model-independent callable PyTorch profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import mammoth.torch.profiling as profiling_module
from mammoth.torch import (
    NamedPhaseProfiler,
    ProfileConfig,
    ProfileTiming,
    TorchRuntimeOptions,
    collect_cuda_memory_stats,
    current_torch_runtime_state,
    normalize_operation_profiles,
    profile_callable,
    reset_cuda_peak_memory_stats,
    resolve_profiler_sort_key,
    summarize_latency,
    summarize_output_value,
    synchronize_device,
    tensor_metadata,
    write_profile_report,
)


class KeywordModel(torch.nn.Module):
    """Small module whose call shape differs from a single-tensor model."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def forward(self, features: torch.Tensor, *, bias: torch.Tensor) -> torch.Tensor:
        return self.projection(features) + bias


def test_named_phase_profiler_measures_dependent_cpu_regions_and_profiler_ranges() -> None:
    phases = NamedPhaseProfiler("cpu")

    with profile(activities=[ProfilerActivity.CPU]) as operation_profiler:
        prepared = phases.measure("prepare inputs", lambda: torch.arange(4, dtype=torch.float32))
        result = phases.measure("consume inputs", lambda: prepared.square().sum())

    summaries = phases.summaries()
    assert result.item() == 14.0
    assert list(summaries) == ["prepare inputs", "consume inputs"]
    assert summaries["prepare inputs"].wall_latency.mean_ms >= 0.0
    assert summaries["prepare inputs"].device_latency is None
    with pytest.raises(TypeError):
        summaries["new"] = summaries["prepare inputs"]  # type: ignore[index]
    operation_rows = normalize_operation_profiles(operation_profiler, device="cpu")
    operation_keys = {row.key for row in operation_rows}
    assert {"prepare inputs", "consume inputs"} <= operation_keys


def test_named_phase_profiler_aggregates_completed_samples_and_excludes_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(
        [
            ("warmup", ProfileTiming(wall_ms=99.0, device_ms=None)),
            ("first", ProfileTiming(wall_ms=1.0, device_ms=None)),
            ("second", ProfileTiming(wall_ms=3.0, device_ms=None)),
        ]
    )
    monkeypatch.setattr(
        profiling_module,
        "_measure_invocation",
        lambda _device, _workload: next(values),
    )
    phases = NamedPhaseProfiler("cpu")

    assert phases.measure("forward", lambda: "unused", record=False) == "warmup"
    assert phases.measure("forward", lambda: "unused") == "first"
    assert phases.measure("forward", lambda: "unused") == "second"

    summary = phases.summaries()["forward"].wall_latency
    assert summary.mean_ms == 2.0
    assert summary.median_ms == 2.0
    assert summary.p95_ms == pytest.approx(2.9)
    assert summary.min_ms == 1.0
    assert summary.max_ms == 3.0


def test_named_phase_profiler_rejects_invalid_names_and_discards_failed_measurements() -> None:
    phases = NamedPhaseProfiler("cpu")
    with pytest.raises(ValueError, match="phase"):
        phases.measure("  ", lambda: None)
    with pytest.raises(RuntimeError, match="failed phase"):
        phases.measure("forward", lambda: (_ for _ in ()).throw(RuntimeError("failed phase")))
    assert phases.summaries() == {}


def test_public_cpu_allocator_helpers_do_not_require_cuda() -> None:
    synchronize_device("cpu")
    reset_cuda_peak_memory_stats("cpu")
    assert collect_cuda_memory_stats("cpu") is None


def test_profile_callable_accepts_project_owned_call_and_components() -> None:
    model = KeywordModel()
    features = torch.ones(4, 3)
    bias = torch.tensor([0.25, -0.25])
    calls = 0

    def workload() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"scores": model(features, bias=bias), "coordinates": (1, "sample")}

    report = profile_callable(
        workload,
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=1,
            measured_iterations=2,
            profiler_iterations=1,
            work_units_per_iteration=4,
            work_unit="sample",
        ),
        components={"projection": model.projection},
    )

    assert calls == 5
    assert report.schema_version == 1
    assert report.device == "cpu"
    assert report.cold_start.device_ms is None
    assert report.device_latency is None
    assert report.memory is None
    assert report.wall_latency.mean_ms >= 0
    assert report.throughput.work_unit == "sample"
    assert report.throughput.work_units_per_iteration == 4
    assert report.throughput.work_units_per_second >= 0
    assert report.operations_profiled is True
    assert any(row.key == "component:projection" for row in report.top_operations)
    payload = report.to_dict()
    assert payload["output_summary"]["kind"] == "mapping"
    assert payload["output_summary"]["values"]["scores"]["shape"] == [4, 2]
    with pytest.raises(TypeError):
        report.output_summary["changed"] = True  # type: ignore[index]


def test_profile_callable_supports_custom_summary_trace_and_json(tmp_path: Path) -> None:
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = torch.ones(3, 2)
    trace_path = tmp_path / "traces" / "profile.json"

    report = profile_callable(
        lambda: left @ right,
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=1,
            chrome_trace=trace_path,
        ),
        summarize_output=lambda output: {
            "kind": "matrix_product",
            "sum": float(output.sum().item()),
        },
    )
    report_path = write_profile_report(tmp_path / "reports" / "profile.json", report)

    assert trace_path.is_file()
    assert report.trace_path == str(trace_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["operations_profiled"] is True
    assert payload["output_summary"] == {"kind": "matrix_product", "sum": 30.0}
    assert payload["top_operations"]
    assert any(
        operation["input_shapes"] == [[2, 3], [3, 2]]
        for operation in payload["top_operations"]
    )


def test_default_output_summary_handles_nested_tensors_without_argmax() -> None:
    summary = summarize_output_value(
        {
            "embedding": torch.tensor([[1.0, 2.0]]),
            "auxiliary": [torch.tensor([True, False]), None],
        }
    )

    values = summary["values"]
    assert values["embedding"]["kind"] == "tensor"
    assert values["embedding"]["mean_abs"] == 1.5
    assert "argmax" not in values["embedding"]
    assert values["auxiliary"]["values"][0]["dtype"] == "torch.bool"


def test_default_output_summary_handles_non_strided_tensor() -> None:
    sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [1]]),
        values=torch.tensor([2.0]),
        size=(2, 2),
        check_invariants=True,
    )

    summary = summarize_output_value(sparse)

    assert summary["layout"] == "torch.sparse_coo"
    assert summary["shape"] == [2, 2]
    assert "stride" not in summary


def test_tensor_metadata_reports_layout_without_stats_by_default() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()

    metadata = tensor_metadata(tensor)

    assert metadata["shape"] == [3, 2]
    assert metadata["dtype"] == "torch.float32"
    assert metadata["device"] == "cpu"
    assert metadata["is_contiguous"] is False
    assert metadata["is_channels_last"] is False
    assert metadata["numel"] == 6
    assert "min" not in metadata


def test_tensor_metadata_includes_stats_for_floating_point_tensors() -> None:
    tensor = torch.tensor([-1.0, 0.0, 2.0, 3.0])

    metadata = tensor_metadata(tensor, include_stats=True)

    assert metadata["min"] == -1.0
    assert metadata["max"] == 3.0
    assert metadata["mean"] == 1.0
    assert metadata["positive_fraction"] == 0.5


def test_tensor_metadata_casts_bool_and_integer_tensors_for_stats() -> None:
    bool_tensor = torch.tensor([True, False, True])
    int_tensor = torch.tensor([1, 2, 3])

    bool_metadata = tensor_metadata(bool_tensor, include_stats=True)
    int_metadata = tensor_metadata(int_tensor, include_stats=True)

    assert bool_metadata["mean"] == pytest.approx(2 / 3)
    assert int_metadata["mean"] == 2.0


def test_tensor_metadata_detects_channels_last_layout() -> None:
    tensor = torch.zeros(1, 3, 4, 4).to(memory_format=torch.channels_last)

    metadata = tensor_metadata(tensor)

    assert metadata["is_channels_last"] is True


def test_tensor_metadata_skips_stats_for_empty_tensors() -> None:
    metadata = tensor_metadata(torch.empty(0), include_stats=True)

    assert "min" not in metadata


def test_tensor_metadata_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        tensor_metadata([1, 2, 3])  # type: ignore[arg-type]


def test_runtime_options_restore_state_after_success_and_failure() -> None:
    before = current_torch_runtime_state()
    options = TorchRuntimeOptions(
        cudnn_benchmark=not before.cudnn_benchmark,
        cudnn_deterministic=not before.cudnn_deterministic,
    )
    observed: list[tuple[bool, bool]] = []

    def observe_runtime() -> torch.Tensor:
        observed.append(
            (
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
            )
        )
        return torch.ones(1)

    profile_callable(
        observe_runtime,
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=1,
            profile_operations=False,
        ),
        runtime_options=options,
    )

    assert observed == [
        (not before.cudnn_benchmark, not before.cudnn_deterministic),
        (not before.cudnn_benchmark, not before.cudnn_deterministic),
    ]
    assert current_torch_runtime_state() == before

    def fail() -> torch.Tensor:
        assert torch.backends.cudnn.benchmark is not before.cudnn_benchmark
        assert torch.backends.cudnn.deterministic is not before.cudnn_deterministic
        raise RuntimeError("profile failed")

    with pytest.raises(RuntimeError, match="profile failed"):
        profile_callable(fail, runtime_options=options)
    assert current_torch_runtime_state() == before


@pytest.mark.parametrize(
    ("matmul_precision", "allow_tf32", "expected_precision"),
    [
        ("highest", False, "highest"),
        ("high", False, "highest"),
        ("medium", False, "highest"),
        ("highest", True, "high"),
        ("high", True, "high"),
        ("medium", True, "medium"),
    ],
)
def test_mixed_matmul_apis_use_one_effective_precision_without_getter_failure(
    matmul_precision: str,
    allow_tf32: bool,
    expected_precision: str,
) -> None:
    before = current_torch_runtime_state()
    report = profile_callable(
        lambda: torch.ones(1),
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=1,
            profile_operations=False,
        ),
        runtime_options=TorchRuntimeOptions(
            matmul_precision=matmul_precision,  # type: ignore[arg-type]
            cuda_matmul_allow_tf32=allow_tf32,
        ),
    )

    assert report.runtime.matmul_precision == expected_precision
    assert report.runtime.cuda_matmul_allow_tf32 is allow_tf32
    assert current_torch_runtime_state() == before


@pytest.mark.parametrize(
    ("allow_tf32", "expected_precision"),
    [(False, "highest"), (True, "medium")],
)
def test_legacy_tf32_only_uses_new_api_and_restores_medium_state(
    allow_tf32: bool,
    expected_precision: str,
) -> None:
    original = current_torch_runtime_state()
    try:
        torch.set_float32_matmul_precision("medium")
        before = current_torch_runtime_state()
        report = profile_callable(
            lambda: torch.ones(1),
            config=ProfileConfig(
                device="cpu",
                warmup_iterations=0,
                measured_iterations=1,
                profiler_iterations=1,
                profile_operations=False,
            ),
            runtime_options=TorchRuntimeOptions(cuda_matmul_allow_tf32=allow_tf32),
        )

        assert report.runtime.matmul_precision == expected_precision
        assert report.runtime.cuda_matmul_allow_tf32 is allow_tf32
        assert current_torch_runtime_state() == before
    finally:
        torch.set_float32_matmul_precision(original.matmul_precision)

def test_disabled_operation_profile_is_explicit_in_report() -> None:
    report = profile_callable(
        lambda: torch.ones(1),
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=3,
            profile_operations=False,
        ),
    )

    assert report.operations_profiled is False
    assert report.top_operations == ()
    assert report.to_dict()["operations_profiled"] is False


def test_component_hooks_are_removed_when_profiled_workload_fails() -> None:
    module = torch.nn.Linear(2, 2)
    calls = 0

    def workload() -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("operation profile failed")
        return module(torch.ones(1, 2))

    with pytest.raises(RuntimeError, match="operation profile failed"):
        profile_callable(
            workload,
            config=ProfileConfig(
                device="cpu",
                warmup_iterations=0,
                measured_iterations=1,
                profiler_iterations=1,
            ),
            components={"linear": module},
        )

    assert not module._forward_pre_hooks
    assert not module._forward_hooks


def test_latency_summary_uses_interpolated_percentiles() -> None:
    summary = summarize_latency([4.0, 1.0, 3.0, 2.0])

    assert summary.mean_ms == 2.5
    assert summary.median_ms == 2.5
    assert summary.p95_ms == pytest.approx(3.85)
    assert summary.min_ms == 1.0
    assert summary.max_ms == 4.0


def test_legacy_cuda_time_field_is_used_for_operation_sorting() -> None:
    class Row:
        def __init__(self, key: str, cuda_time_total: float) -> None:
            self.key = key
            self.cuda_time_total = cuda_time_total

    class Profiler:
        def key_averages(self) -> list[Row]:
            return [Row("slow", 20.0), Row("fast", 2.0)]

    rows = normalize_operation_profiles(
        Profiler(),
        device="cuda",
        row_limit=2,
        sort_by=None,
    )

    assert [row.key for row in rows] == ["slow", "fast"]


def test_current_cuda_time_field_is_used_for_legacy_requested_sort() -> None:
    class Row:
        def __init__(self, key: str, device_time_total: float) -> None:
            self.key = key
            self.device_time_total = device_time_total

    class Profiler:
        def key_averages(self) -> list[Row]:
            return [Row("slow", 20.0), Row("fast", 2.0)]

    rows = normalize_operation_profiles(
        Profiler(),
        device="cuda",
        row_limit=2,
        sort_by="cuda_time_total",
    )

    assert [row.key for row in rows] == ["slow", "fast"]


def test_resolve_profiler_sort_key_maps_legacy_cuda_names_when_using_cuda() -> None:
    assert resolve_profiler_sort_key("cuda_time_total", True) == "device_time_total"
    assert resolve_profiler_sort_key("self_cuda_time_total", True) == "self_device_time_total"


def test_resolve_profiler_sort_key_leaves_other_names_and_cpu_requests_untouched() -> None:
    assert resolve_profiler_sort_key("cpu_time_total", True) == "cpu_time_total"
    assert resolve_profiler_sort_key("cuda_time_total", False) == "cuda_time_total"
    assert resolve_profiler_sort_key("device_time_total", True) == "device_time_total"


def test_resolve_profiler_sort_key_rejects_empty_request() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_profiler_sort_key("", True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_iterations": -1}, "warmup_iterations"),
        ({"measured_iterations": 0}, "measured_iterations"),
        ({"profiler_iterations": 0}, "profiler_iterations"),
        ({"work_units_per_iteration": float("nan")}, "work_units_per_iteration"),
        ({"work_units_per_iteration": 0}, "work_units_per_iteration"),
        ({"work_unit": ""}, "work_unit"),
        ({"row_limit": 0}, "row_limit"),
        ({"sort_by": ""}, "sort_by"),
        ({"profile_operations": False, "chrome_trace": Path("trace.json")}, "chrome_trace"),
    ],
)
def test_profile_config_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProfileConfig(**kwargs)


def test_profile_callable_rejects_invalid_components_and_summary() -> None:
    config = ProfileConfig(
        device="cpu",
        warmup_iterations=0,
        measured_iterations=1,
        profiler_iterations=1,
        profile_operations=False,
    )
    with pytest.raises(ValueError, match="component names"):
        profile_callable(lambda: torch.ones(1), config=config, components={"": torch.nn.Identity()})
    with pytest.raises(TypeError, match="torch.nn.Module"):
        profile_callable(lambda: torch.ones(1), config=config, components={"bad": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        profile_callable(
            lambda: torch.ones(1),
            config=config,
            summarize_output=lambda _output: {"invalid": float("inf")},
        )


def test_profile_callable_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA device requested"):
        profile_callable(lambda: torch.ones(1), config=ProfileConfig(device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_profile_callable_collects_cuda_timing_and_memory() -> None:
    tensor = torch.ones(8, 8, device="cuda")

    report = profile_callable(
        lambda: tensor @ tensor,
        config=ProfileConfig(
            device="cuda",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=1,
        ),
    )

    assert report.cold_start.device_ms is not None
    assert report.device_latency is not None
    assert report.memory is not None
    assert report.memory.max_allocated_bytes > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_named_phase_profiler_and_public_cuda_allocator_helpers() -> None:
    phases = NamedPhaseProfiler("cuda")
    tensor = torch.ones(8, 8, device="cuda")

    reset_cuda_peak_memory_stats("cuda")
    result = phases.measure("matrix product", lambda: tensor @ tensor)
    synchronize_device("cuda")
    memory = collect_cuda_memory_stats("cuda")

    assert result.device.type == "cuda"
    assert phases.summaries()["matrix product"].device_latency is not None
    assert memory is not None
    assert memory.max_allocated_bytes > 0
