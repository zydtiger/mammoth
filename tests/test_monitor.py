from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
from mammoth.monitor.dashboard import dashboard_layout, sparkline
from mammoth.monitor.psutil_telemetry import (
    PsutilViewerTelemetry,
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
):
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("opaque",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
        execution_id=execution_id,
        created_at=created_at,
        previous_execution_id=previous_execution_id,
        parent_execution_id=parent_execution_id,
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
        previous_execution_id="first",
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
    assert "epoch 2" in rendered
    assert "optimizer_step 12" in rendered
    assert "loss" in rendered
    assert "learning_rate" in rendered
    assert "VIEWER HOST · viewer-host" in rendered
    assert "CPU 25.0%" in rendered
    assert "Sampled" in rendered
    assert "2026-01-01T00:00:00Z" in rendered


def test_metric_sparkline_handles_sparse_constant_and_narrow_histories() -> None:
    assert sparkline((), 8) == "--"
    assert sparkline((2.0,), 8) == "▄"
    assert sparkline((2.0, 2.0, 2.0), 2) == "▄▄"
    assert len(sparkline(tuple(float(value) for value in range(100)), 12)) == 12


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
