"""Tests for model-independent callable PyTorch profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import mammoth.torch.profiling as profiling_module
from mammoth.torch import (
    ProfileConfig,
    TorchRuntimeOptions,
    current_torch_runtime_state,
    profile_callable,
    summarize_latency,
    summarize_output_value,
    write_profile_report,
)


class KeywordModel(torch.nn.Module):
    """Small module whose call shape differs from a single-tensor model."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def forward(self, features: torch.Tensor, *, bias: torch.Tensor) -> torch.Tensor:
        return self.projection(features) + bias


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


def test_runtime_options_restore_state_after_success_and_failure() -> None:
    before = current_torch_runtime_state()
    options = TorchRuntimeOptions(cudnn_benchmark=not before.cudnn_benchmark)
    observed: list[bool] = []

    profile_callable(
        lambda: observed.append(torch.backends.cudnn.benchmark) or torch.ones(1),
        config=ProfileConfig(
            device="cpu",
            warmup_iterations=0,
            measured_iterations=1,
            profiler_iterations=1,
            profile_operations=False,
        ),
        runtime_options=options,
    )

    assert observed == [not before.cudnn_benchmark, not before.cudnn_benchmark]
    assert current_torch_runtime_state() == before

    def fail() -> torch.Tensor:
        assert torch.backends.cudnn.benchmark is not before.cudnn_benchmark
        raise RuntimeError("profile failed")

    with pytest.raises(RuntimeError, match="profile failed"):
        profile_callable(fail, runtime_options=options)
    assert current_torch_runtime_state() == before


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

    rows = profiling_module._top_operations(  # type: ignore[attr-defined]
        Profiler(),  # type: ignore[arg-type]
        device=torch.device("cuda"),
        row_limit=2,
        sort_by=None,
    )

    assert [row.key for row in rows] == ["slow", "fast"]


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
