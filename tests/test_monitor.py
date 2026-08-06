from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import RunLayout, create_execution_context
from mammoth.core.events import ExecutionEventWriter
from mammoth.monitor import (
    ExecutionMonitor,
    RunMonitor,
    discover_executions,
    execution_lineage,
    render_snapshot,
    sample_viewer_telemetry,
    select_execution,
)
from mammoth.monitor.dashboard import braille_line_chart, dashboard_layout
from mammoth.monitor.psutil_telemetry import (
    GpuTelemetry,
    PsutilViewerTelemetry,
    PsutilViewerTelemetrySampler,
    sample_psutil_viewer_telemetry,
)
from mammoth.monitor.textual_ui import MonitorApp


def create_context(
    layout: RunLayout,
    execution_id: str,
    created_at: str,
    *,
    previous_execution_id: str | None = None,
    parent_execution_id: str | None = None,
    world_size: int = 1,
    starting_epoch: int | None = None,
    starting_global_step: int | None = None,
):
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("opaque",),
        world_size=world_size,
        execution_mode="single" if world_size == 1 else "distributed",
        command=("python", "job.py"),
        execution_id=execution_id,
        created_at=created_at,
        previous_execution_id=previous_execution_id,
        parent_execution_id=parent_execution_id,
        starting_epoch=starting_epoch,
        starting_global_step=starting_global_step,
    )


def test_discovery_selects_latest_valid_execution_and_isolates_warning(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    first = create_context(layout, "first", "2026-01-01T00:00:00Z")
    second = create_context(layout, "second", "2026-01-02T00:00:00Z")
    invalid = layout.executions_dir / "invalid"
    invalid.mkdir()
    (invalid / "execution.json").write_text("{}")

    contexts, warnings = discover_executions(layout)

    assert contexts == [first, second]
    assert len(warnings) == 1
    assert "invalid" in warnings[0]
    assert select_execution(layout) == second
    assert select_execution(layout, "first") == first


def test_lineage_follows_only_explicit_links_without_cycles(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    create_context(layout, "root", "2026-01-01T00:00:00Z")
    parent = create_context(
        layout,
        "parent",
        "2026-01-02T00:00:00Z",
        previous_execution_id="root",
    )
    current = create_context(
        layout,
        "current",
        "2026-01-03T00:00:00Z",
        previous_execution_id="parent",
        parent_execution_id="root",
    )

    assert execution_lineage(layout, current.metadata) == ("parent", "root")
    assert execution_lineage(layout, parent.metadata) == ("root",)


def test_run_monitor_exposes_execution_history_and_lineage_metrics(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    first = create_context(layout, "first", "2026-01-01T00:00:00Z")
    second = create_context(
        layout,
        "second",
        "2026-01-02T00:00:00Z",
        parent_execution_id="first",
    )
    with ExecutionEventWriter.for_process(first, rank=0) as writer:
        writer.emit_progress(
            phase="train",
            task_id="epoch",
            completed=1,
            total=2,
            coordinates={"epoch": 0},
            display_metrics={"loss": 2.0},
        )
    with ExecutionEventWriter.for_process(second, rank=0) as writer:
        writer.emit_progress(
            phase="train",
            task_id="epoch",
            completed=1,
            total=2,
            epoch=1,
            optimizer_step=4,
            display_metrics={"loss": 1.0, "learning_rate": 0.01},
        )

    snapshot = RunMonitor(layout).poll()

    assert [attempt.execution_id for attempt in snapshot.executions] == ["first", "second"]
    assert snapshot.selected_execution_id == "second"
    assert snapshot.selected.current_coordinates == {"epoch": 1, "optimizer_step": 4}
    assert [point.value for point in snapshot.metric_history["loss"]] == [2.0, 1.0]
    assert [point.value for point in snapshot.metric_history["learning_rate"]] == [0.01]


def test_run_monitor_can_select_an_earlier_execution(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    create_context(layout, "first", "2026-01-01T00:00:00Z")
    create_context(layout, "second", "2026-01-02T00:00:00Z")

    snapshot = RunMonitor(layout).poll(selected_execution_id="first")

    assert snapshot.selected_execution_id == "first"


def test_monitor_folds_generic_tasks_metrics_throughput_and_eta(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit("process_started", phase="opaque")
        writer.emit("phase_started", phase="opaque")
        writer.emit("task_started", phase="opaque", task_id="unit-1")
        writer.emit_progress(
            phase="opaque",
            task_id="unit-1",
            completed=4,
            total=10,
            throughput=2.0,
            coordinates={"round": 3},
            display_metrics={"arbitrary/score": 0.75},
        )
    snapshot = ExecutionMonitor(layout, "attempt").poll()
    task = next(iter(snapshot.tasks.values()))

    assert snapshot.status == "running"
    assert snapshot.phases == {"opaque": "running"}
    assert task.completed == 4
    assert task.eta_seconds == 3.0
    assert task.coordinates == {"round": 3}
    assert snapshot.metric_history["arbitrary/score"][-1].value == 0.75

    rendered = render_snapshot(
        snapshot,
        now=datetime(2026, 1, 1, tzinfo=UTC),
        stale_after_seconds=10**9,
    )
    assert "rank-0/opaque/unit-1: running 4/10 eta=3s" in rendered
    assert "arbitrary/score: 0.75 (1 points)" in rendered
    assert "\x1b[" not in rendered


def test_monitor_preserves_valid_prefix_and_isolates_malformed_stream(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
    stream = context.execution_dir / "rank-0.jsonl"
    with stream.open("ab") as handle:
        handle.write(b"{}\n")

    snapshot = ExecutionMonitor(layout, "attempt").poll()

    assert snapshot.status == "running"
    assert len(snapshot.warnings) == 1
    assert "line 2" in snapshot.warnings[0]


def test_monitor_cli_preserves_public_command_shape(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")

    result = CliRunner().invoke(
        app,
        [
            "monitor",
            "run",
            "--entry",
            str(tmp_path),
            "--execution",
            "attempt",
            "--telemetry",
        ]
    )

    assert result.exit_code == 0
    assert "Run: run" in result.output
    assert "Execution: attempt" in result.output
    assert '"host_role": "viewer"' in result.output


def test_local_telemetry_never_claims_execution_host_provenance() -> None:
    sample = sample_viewer_telemetry()
    assert sample.host_role == "viewer"
    assert sample.hostname


def test_dashboard_renders_identity_progress_metrics_and_viewer_telemetry(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "complete-execution-id", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit("process_started", phase="train")
        writer.emit_progress(
            phase="train",
            task_id="epoch-2",
            completed=4,
            total=10,
            throughput=2.0,
            coordinates={"epoch": 2, "optimizer_step": 12},
            display_metrics={"loss": 0.5, "learning_rate": 0.001},
        )
    snapshot = RunMonitor(layout).poll()
    host = PsutilViewerTelemetry(
        host_role="viewer",
        hostname="viewer-host",
        sampled_at="2026-01-01T00:00:00Z",
        cpu_percent=25.0,
        memory_percent=50.0,
        load_average_1m=1.5,
        cpu_frequency_mhz=4200.0,
        memory_used_bytes=16 * 1024**3,
        memory_total_bytes=32 * 1024**3,
        cpu_model_name="Example CPU",
        ram_ddr_generation="DDR5",
        ram_speed="5,600 MT/s",
        cpu_power_w=105.5,
        gpus=(GpuTelemetry(0, "Example GPU", 75.0, 2800.0, 275.25),),
    )
    console = Console(width=120, record=True, color_system=None)

    console.print(
        dashboard_layout(
            snapshot,
            host=host,
            detail=True,
            compact=False,
            now=datetime.now(UTC),
        )
    )
    rendered = console.export_text()

    assert "complete-execution-id" in rendered
    assert "4/10" in rendered
    assert "ETA 3s" in rendered
    assert "Epoch 2/--" in rendered
    assert "Optimizer 12/--" in rendered
    assert "loss" in rendered
    assert "Learning rate" in rendered
    assert "HOST RESOURCES · viewer-host" in rendered
    assert "VIEWER HOST RESOURCES" not in rendered
    assert "CPU · Example CPU" in rendered
    assert "Util 25.0%" in rendered
    assert "Power 105.5 W · Core 4,200 MHz" in rendered
    assert "DDR5 · 5,600 MT/s" in rendered
    assert "GPU 0 · Example GPU" in rendered
    assert "Util 75.0%" in rendered
    assert "Power 275.2 W · Core 2,800 MHz" in rendered
    assert "Load" not in rendered
    assert "Sampled" not in rendered


def test_braille_chart_restores_multi_row_terminal_geometry() -> None:
    values = tuple(float(value) for value in (5, 4, 3, 2, 3, 1, 2, 0))

    wide = braille_line_chart(values, 24, height=4)
    compact = braille_line_chart(values, 16, height=3, aggregation="last")

    assert len(wide.splitlines()) == 4
    assert len(compact.splitlines()) == 3
    assert any(0x2800 <= ord(character) <= 0x28FF for character in wide)


def test_dashboard_shortens_batch_and_microbatch_throughput_units(tmp_path: Path) -> None:
    for unit in ("batch", "microbatch"):
        layout = RunLayout(tmp_path / unit, "run").prepare()
        context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
        with ExecutionEventWriter.for_process(context, rank=0) as writer:
            writer.emit_progress(
                phase="train",
                task_id="train",
                completed=1,
                total=2,
                unit=unit,
                throughput=2.0,
            )
        console = Console(width=120, record=True, color_system=None)

        console.print(
            dashboard_layout(
                RunMonitor(layout).poll(),
                host=None,
                detail=False,
                compact=False,
            )
        )
        rendered = console.export_text()

        assert "2.0 b/s" in rendered
        assert f"{unit}/s" not in rendered


def test_dashboard_shows_segmentation_test_patch_rate_as_batches(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit_progress(
            phase="test/segmentation",
            task_id="segment",
            completed=1,
            total=2,
            unit="patches",
            throughput=2.0,
        )
    snapshot = RunMonitor(layout).poll()

    for compact, width in ((False, 120), (True, 80)):
        console = Console(width=width, record=True, color_system=None)
        console.print(
            dashboard_layout(
                snapshot,
                host=None,
                detail=False,
                compact=compact,
            )
        )
        rendered = console.export_text()

        assert "2.0 b/s" in rendered
        assert "patches/s" not in rendered


def test_dashboard_omits_secondary_metrics_without_loss_or_learning_rate(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit_progress(
            phase="train",
            task_id="train",
            completed=1,
            total=2,
            display_metrics={"mean_iou": 0.75},
        )
    console = Console(width=120, record=True, color_system=None)

    console.print(
        dashboard_layout(
            RunMonitor(layout).poll(),
            host=None,
            detail=False,
            compact=False,
        )
    )
    rendered = console.export_text()

    assert "TRAINING TRENDS" not in rendered
    assert "mean_iou" not in rendered


def test_dashboard_restores_legacy_wide_and_compact_information_hierarchy(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "a-production-run-name").prepare()
    first_id = "20260101T000000000000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    second_id = "20260102T000000000000Z-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    first = create_context(layout, first_id, "2026-01-01T00:00:00Z")
    second = create_context(
        layout,
        second_id,
        "2026-01-02T00:00:00Z",
        parent_execution_id=first_id,
        starting_epoch=2,
        starting_global_step=16,
    )
    for context, start in ((first, 0), (second, 16)):
        ticks = count()
        with ExecutionEventWriter.for_process(
            context,
            rank=0,
            monotonic_clock=lambda ticks=ticks: float(next(ticks) * 2),
        ) as writer:
            writer.emit("process_started", phase="train")
            writer.emit("phase_started", phase="train")
            writer.emit("task_started", phase="train", task_id="train-epoch-3")
            for index in range(16):
                step = start + index + 1
                writer.emit(
                    "progress",
                    phase="train",
                    task_id="train-epoch-3",
                    completed=index + 1,
                    total=2000,
                    unit="microbatch",
                    throughput=6.5,
                    epoch=3,
                    epoch_total=10,
                    optimizer_step=step,
                    optimizer_step_total=100,
                    display_metrics={
                        "loss": 2.0 / (index + 1),
                        "learning_rate": index / 1000,
                        "mean_iou": index / 16,
                    },
                )
    snapshot = RunMonitor(layout).poll()
    host = PsutilViewerTelemetry(
        host_role="viewer",
        hostname="viewer-host",
        sampled_at="2026-01-02T00:05:00Z",
        cpu_percent=25.0,
        memory_percent=50.0,
        load_average_1m=1.5,
        cpu_frequency_mhz=4200.0,
        memory_used_bytes=16 * 1024**3,
        memory_total_bytes=32 * 1024**3,
        cpu_model_name="Example CPU",
        ram_ddr_generation="DDR5",
        ram_speed="5,600 MT/s",
        cpu_power_w=105.5,
        gpus=(
            GpuTelemetry(0, "GPU Zero", 25.0, 2400.0, 200.0),
            GpuTelemetry(1, "GPU One", 75.0, 2800.0, 275.25),
        ),
    )
    now = datetime(2026, 1, 2, 0, 10, tzinfo=UTC)

    wide_console = Console(width=140, record=True, color_system=None)
    wide_console.print(
        dashboard_layout(
            snapshot,
            host=host,
            detail=False,
            compact=False,
            now=now,
        )
    )
    wide = wide_console.export_text()

    assert "SELECTED ATTEMPT · bbbbbbbb" in wide
    assert second_id not in wide
    assert "Epoch 3/10" in wide
    assert "Optimizer 32/100" in wide
    assert "Global step" not in wide
    assert "TRAINING TRENDS" in wide
    assert "Training losses · batch" in wide
    assert "Learning rate" in wide
    assert "mean_iou" not in wide
    assert "Metric" not in wide
    assert "Trend" not in wide
    assert any(0x2800 <= ord(character) <= 0x28FF for character in wide)
    assert "Overall" in wide
    assert "16/2,000" in wide
    assert "6.5 b/s" in wide
    assert "microbatch/s" not in wide
    assert "HOST RESOURCES · viewer-host" in wide
    assert "16.0/32.0 GiB (50.0%)" in wide
    assert "CPU · Example CPU" in wide
    assert "DDR5 · 5,600 MT/s" in wide
    assert "GPU 0 · GPU Zero" in wide
    assert "GPU 1 · GPU One" in wide
    assert "Power 105.5 W · Core 4,200 MHz" in wide
    assert "Power 200.0 W · Core 2,400 MHz" in wide
    assert "Power 275.2 W · Core 2,800 MHz" in wide
    wide_lines = wide.splitlines()
    for identity in ("CPU · Example CPU", "RAM", "GPU 0 · GPU Zero", "GPU 1 · GPU One"):
        identity_index = next(
            index for index, line in enumerate(wide_lines) if line.strip() == identity
        )
        assert "Util" in wide_lines[identity_index + 1] or identity == "RAM"
        assert "Core" in wide_lines[identity_index + 1] or identity == "RAM"
    assert "RANKS" in wide
    assert "ATTEMPT HISTORY" in wide
    assert "aaaaaaaa -> ep2" in wide

    compact_console = Console(width=80, record=True, color_system=None)
    compact_console.print(
        dashboard_layout(
            snapshot,
            host=host,
            detail=False,
            compact=True,
            now=now,
        )
    )
    compact = compact_console.export_text()

    assert "SELECTED ATTEMPT · bbbbbbbb" in compact
    assert second_id not in compact
    assert "overall: 16/2,000 microbatch · 6.5 b/s" in compact
    assert "Overall ━" not in compact
    assert max(len(line) for line in compact.splitlines()) <= 80


def test_dashboard_keeps_parent_overview_and_active_leaf_rank_progress(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "distributed-run").prepare()
    context = create_context(
        layout,
        "20260101T000000000000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "2026-01-01T00:00:00Z",
        world_size=2,
    )
    for rank, child_completed, child_total in (
        (0, 9_876_543, 10_000_000),
        (1, 3_000_000, 4_000_000),
    ):
        with ExecutionEventWriter.for_process(context, rank=rank) as writer:
            writer.emit("process_started", phase="work")
            writer.emit("task_started", phase="work", task_id="pipeline")
            writer.emit(
                "progress",
                phase="work",
                task_id="pipeline",
                completed=5,
                total=41,
                unit="items",
            )
            writer.emit(
                "task_started",
                phase="work",
                task_id="current-item",
                parent_task_id="pipeline",
            )
            writer.emit(
                "progress",
                phase="work",
                task_id="current-item",
                parent_task_id="pipeline",
                completed=child_completed,
                total=child_total,
                unit="parts",
            )
            writer.emit("heartbeat", phase="work", task_id="pipeline")

    snapshot = RunMonitor(layout).poll()
    console = Console(width=140, record=True, color_system=None)
    console.print(
        dashboard_layout(
            snapshot,
            host=None,
            detail=False,
            compact=False,
            now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        )
    )
    rendered = console.export_text()
    selected_attempt, ranks = rendered.split("RANKS", maxsplit=1)
    ranks, _history = ranks.split("ATTEMPT HISTORY", maxsplit=1)

    assert "work / pipeline" in selected_attempt
    assert "5/41" in selected_attempt
    assert ranks.count("current-item") == 2
    assert "9,876,543/10,000,000" in ranks
    assert "3,000,000/4,000,000" in ranks


def test_resume_history_prunes_superseded_parent_metric_tail(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    parent = create_context(layout, "parent", "2026-01-01T00:00:00Z")
    child = create_context(
        layout,
        "child",
        "2026-01-02T00:00:00Z",
        parent_execution_id="parent",
        starting_global_step=3,
    )
    parent_ticks = count()
    with ExecutionEventWriter.for_process(
        parent,
        rank=0,
        monotonic_clock=lambda: float(next(parent_ticks) * 2),
    ) as writer:
        for step in (1, 2, 3, 4):
            writer.emit(
                "progress",
                phase="train",
                task_id="train",
                completed=step,
                total=10,
                global_step=step,
                display_metrics={"loss": float(step)},
            )
    child_ticks = count()
    with ExecutionEventWriter.for_process(
        child,
        rank=0,
        monotonic_clock=lambda: float(next(child_ticks) * 2),
    ) as writer:
        for step in (3, 4):
            writer.emit(
                "progress",
                phase="train",
                task_id="train",
                completed=step,
                total=10,
                global_step=step,
                display_metrics={"loss": float(step * 10)},
            )

    snapshot = RunMonitor(layout).poll()

    assert [point.value for point in snapshot.metric_history["loss"]] == [
        1.0,
        2.0,
        30.0,
        40.0,
    ]


def test_new_process_generation_clears_stale_task_progress_and_infers_terminal(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("process_started", phase="train")
        writer.emit("task_started", phase="train", task_id="train")
        writer.emit(
            "progress",
            phase="train",
            task_id="train",
            completed=8,
            total=10,
        )
        writer.emit("process_completed", phase="train", exit_code=0)
        writer.emit("process_started", phase="train")
        writer.emit("task_started", phase="train", task_id="train")
        writer.emit("process_completed", phase="train", exit_code=0)

    snapshot = ExecutionMonitor(layout).poll()

    assert snapshot.status == "completed"
    assert snapshot.current_task is not None
    assert snapshot.current_task.completed == 0
    assert snapshot.current_task.total is None


def test_psutil_telemetry_isolates_optional_sampler_failures(
    monkeypatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr("mammoth.monitor.psutil_telemetry.psutil.cpu_percent", fail)
    monkeypatch.setattr("mammoth.monitor.psutil_telemetry.psutil.virtual_memory", fail)

    sample = sample_psutil_viewer_telemetry()

    assert sample.host_role == "viewer"
    assert sample.cpu_percent is None
    assert sample.memory_percent is None


def test_psutil_telemetry_restores_sudo_dimm_probe_and_all_gpu_rows() -> None:
    commands: list[tuple[str, ...]] = []

    def run_command(
        command,
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout, check
        normalized = tuple(command)
        commands.append(normalized)
        if normalized[:2] == ("sudo", "-n"):
            return subprocess.CompletedProcess(normalized, 1, "", "password required")
        if normalized[:2] == ("sudo", "dmidecode"):
            return subprocess.CompletedProcess(
                normalized,
                0,
                """
                Memory Device
                    Type: DDR5
                    Speed: 5600 MT/s
                    Configured Memory Speed: 5200 MT/s
                """,
                "",
            )
        if normalized[0] == "sensors":
            return subprocess.CompletedProcess(
                normalized,
                0,
                """
                {
                  "zenpower-pci-00c3": {
                    "RAPL_P_Package": {"power1_input": 65.25}
                  },
                  "zenpower-pci-00c4": {
                    "RAPL_P_Package": {"power1_input": 40.25}
                  }
                }
                """,
                "",
            )
        if normalized[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                normalized,
                0,
                "0, NVIDIA RTX A, 25, 200, 2400\n"
                "1, NVIDIA RTX B, 75, 275.25, 2800\n",
                "",
            )
        raise AssertionError(normalized)

    sampler = PsutilViewerTelemetrySampler(
        command_runner=run_command,
        allow_sudo_password_prompt=True,
    )

    first = sampler.sample()
    second = sampler.sample()

    assert first.ram_ddr_generation == "DDR5"
    assert first.ram_speed == "5,200 MT/s"
    assert first.cpu_power_w == 105.5
    assert [(gpu.index, gpu.name) for gpu in first.gpus] == [
        (0, "NVIDIA RTX A"),
        (1, "NVIDIA RTX B"),
    ]
    assert first.gpus[1].utilization_percent == 75.0
    assert first.gpus[1].power_draw_w == 275.25
    assert first.gpus[1].core_clock_mhz == 2800.0
    assert second.gpus == first.gpus
    assert commands.count(("sudo", "-n", "dmidecode", "--type", "memory")) == 1
    assert commands.count(("sudo", "dmidecode", "--type", "memory")) == 1
    assert commands.count(("sensors", "-j", "zenpower-*")) == 2
    assert sum(command[0] == "nvidia-smi" for command in commands) == 2


def test_textual_monitor_navigates_execution_history_and_resizes(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    create_context(layout, "first", "2026-01-01T00:00:00Z")
    create_context(layout, "second", "2026-01-02T00:00:00Z")
    monitor = RunMonitor(layout)
    app = MonitorApp(
        monitor,
        monitor.poll(),
        watch=False,
        telemetry=False,
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.snapshot.selected_execution_id == "second"
            await pilot.press("k")
            assert app.snapshot.selected_execution_id == "first"
            await pilot.press("enter")
            assert app.detail is True
            await pilot.resize_terminal(70, 30)
            assert app.query_one("#body").size.width < 80

    asyncio.run(exercise())


def test_exact_textual_monitor_opens_and_remains_in_locked_detail_mode(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    create_context(layout, "first", "2026-01-01T00:00:00Z")
    create_context(layout, "second", "2026-01-02T00:00:00Z")
    monitor = RunMonitor(layout, "first")
    app = MonitorApp(
        monitor,
        monitor.poll(),
        watch=False,
        telemetry=False,
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.detail is True
            assert app.snapshot.selected_execution_id == "first"
            await pilot.press("enter", "j")
            assert app.detail is True
            assert app.snapshot.selected_execution_id == "first"

    asyncio.run(exercise())
