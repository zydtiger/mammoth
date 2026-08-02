from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import RunLayout, create_execution_context
from mammoth.core.events import ExecutionEventWriter
from mammoth.monitor import (
    ExecutionMonitor,
    discover_executions,
    execution_lineage,
    render_snapshot,
    sample_viewer_telemetry,
    select_execution,
)


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
