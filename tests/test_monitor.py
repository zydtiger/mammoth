from __future__ import annotations

import asyncio
import json
import random
import subprocess
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console
from textual.pilot import Pilot
from textual.widgets import Static
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import (
    GroupEventWriter,
    GroupLayout,
    GroupMember,
    RunLayout,
    create_execution_context,
    publish_group_manifest,
)
from mammoth.core.events import (
    ExecutionEventReadError,
    ExecutionEventTailReader,
    ExecutionEventWriter,
)
from mammoth.monitor import (
    ExecutionMonitor,
    RunMonitor,
    discover_executions,
    execution_lineage,
    fold_events,
    render_snapshot,
    sample_viewer_telemetry,
    select_execution,
)
from mammoth.monitor.dashboard import (
    _rendered_height,
    _row_window,
    braille_line_chart,
    dashboard_layout,
    fleet_dashboard_layout,
    fleet_rows,
    group_dashboard_layout,
)
from mammoth.monitor.fleet import FleetMonitor
from mammoth.monitor.model import event_stream_paths
from mammoth.monitor.psutil_telemetry import (
    GpuTelemetry,
    PsutilViewerTelemetry,
    PsutilViewerTelemetrySampler,
    sample_psutil_viewer_telemetry,
)
from mammoth.monitor.textual_ui import (
    FleetApp,
    FleetScreen,
    GroupScreen,
    MonitorApp,
    RunScreen,
    run_fleet_textual,
)


def _composited_lines(body: Static) -> list[str]:
    """Return the actual visible rows a mounted Static widget shows.

    Reads ``Widget.render_line(y)`` for each row Textual's scrollable
    ``#body`` container actually composites, rather than printing the
    widget's full (possibly taller-than-viewport) renderable. A row that
    the layout function windowed in but the container then clips off screen
    would still show up in the unclipped renderable, so a visibility
    assertion must read this composited reality, not the pre-composite
    content, to be a genuine regression guard.
    """
    return [body.render_line(y).text for y in range(body.size.height)]


async def _wait_until(
    pilot: Pilot[object],
    predicate: Callable[[], bool],
    *,
    attempts: int = 50,
) -> None:
    """Pump the Textual message loop until ``predicate`` holds.

    Bounds waiting for a background-worker refresh (``@work(thread=True)``)
    to publish its result back onto the event loop: each iteration yields
    control via ``pilot.pause()`` so the worker's ``call_from_thread``
    callback gets a chance to run, without a fixed sleep that would be
    either flaky (too short) or slow (too long).
    """
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    assert predicate(), "condition did not become true before the attempt budget ran out"


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


def test_lineage_follows_only_explicit_links(tmp_path: Path) -> None:
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


def test_run_monitor_stops_on_unknown_and_cyclic_explicit_parents(tmp_path: Path) -> None:
    """Missing and cyclic explicit parents do not make monitor reconstruction loop."""
    unknown_layout = RunLayout(tmp_path / "unknown", "run").prepare()
    create_context(
        unknown_layout,
        "unknown-parent",
        "2026-01-01T00:00:00Z",
        parent_execution_id="missing",
    )
    unknown_snapshot = RunMonitor(unknown_layout).poll()
    assert [item.execution_id for item in unknown_snapshot.resume_lineage] == ["unknown-parent"]
    assert unknown_snapshot.metric_history == {}

    self_layout = RunLayout(tmp_path / "self", "run").prepare()
    create_context(
        self_layout,
        "self-parent",
        "2026-01-01T00:00:00Z",
        parent_execution_id="self-parent",
    )
    self_snapshot = RunMonitor(self_layout).poll()
    assert [item.execution_id for item in self_snapshot.resume_lineage] == ["self-parent"]
    assert self_snapshot.metric_history == {}

    cycle_layout = RunLayout(tmp_path / "cycle", "run").prepare()
    create_context(
        cycle_layout,
        "first",
        "2026-01-01T00:00:00Z",
        parent_execution_id="second",
    )
    create_context(
        cycle_layout,
        "second",
        "2026-01-02T00:00:00Z",
        parent_execution_id="first",
    )
    cycle_snapshot = RunMonitor(cycle_layout).poll()
    assert [item.execution_id for item in cycle_snapshot.resume_lineage] == ["first", "second"]
    assert cycle_snapshot.metric_history == {}


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
    assert "rank-0/opaque/unit-1: running 4/10 rate=2.0 b/s eta=3s" in rendered
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


def _oracle_snapshot(layout: RunLayout, context: object) -> object:
    """Fold every stream from scratch with the same per-path error isolation
    :class:`ExecutionMonitor` uses, as the full, non-incremental reference
    fold that incremental polling must stay equivalent to.
    """
    events: list[object] = []
    warnings: list[str] = []
    for path in event_stream_paths(context):
        reader = ExecutionEventTailReader(path)
        try:
            events.extend(reader.poll())
        except ExecutionEventReadError as error:
            events.extend(error.valid_events)
            warnings.append(str(error))
    return fold_events(
        context,
        events,
        lineage=execution_lineage(layout, context.metadata),
        warnings=warnings,
    )


def _assert_snapshot_equivalent(incremental: object, oracle: object) -> None:
    """Assert every observable field an incremental poll exposes matches the oracle.

    ``events`` is compared against only the oracle's terminal-class events,
    since incremental folding deliberately retains only that subset (see
    ``ExecutionMonitor``'s docstring) rather than the complete raw history.
    """
    assert incremental.status == oracle.status
    assert incremental.phases == oracle.phases
    assert incremental.producers == oracle.producers
    assert incremental.tasks == oracle.tasks
    assert incremental.metric_history == oracle.metric_history
    assert incremental.terminal_event_time == oracle.terminal_event_time
    assert incremental.duration_seconds == oracle.duration_seconds
    assert incremental.updated_at == oracle.updated_at
    assert incremental.current_task == oracle.current_task
    assert incremental.current_coordinates == oracle.current_coordinates
    assert incremental.execution_id == oracle.execution_id
    assert incremental.created_at == oracle.created_at
    assert incremental.lineage == oracle.lineage
    assert sorted(incremental.warnings) == sorted(oracle.warnings)
    assert incremental.events == tuple(event for event in oracle.events if event.is_terminal)


def _iso_clock(*values: str) -> Callable[[], str]:
    """Return a callable yielding ``values`` in order, one per call.

    Pins exact event timestamps through ``ExecutionEventWriter``'s
    ``utc_clock`` option, decoupled from real wall-clock write order — lets a
    test simulate a writer whose flush is delayed (or whose host clock is
    skewed) relative to another producer's writer.
    """
    iterator = iter(values)
    return lambda: next(iterator)


def test_incremental_fold_matches_full_fold_under_cross_producer_visibility_skew(
    tmp_path: Path,
) -> None:
    """P1-2 regression.

    A later poll can make a chronologically-earlier event (from a different
    producer) visible only after a chronologically-later event was already
    applied from an earlier poll: a stalled writer flush or cross-host clock
    skew, not merely poll timing. ``ExecutionMonitor.poll`` sorts only each
    poll's own newly read events among themselves and applies them after
    everything already folded, so a naive append/overwrite would record
    rank 1's chronologically-earlier point after rank 0's, diverging from
    the full re-fold. This must be equivalent regardless.
    """
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z", world_size=2)
    monitor = ExecutionMonitor(layout, "attempt")

    rank0 = ExecutionEventWriter.for_process(
        context,
        rank=0,
        world_size=2,
        utc_clock=_iso_clock("2026-01-01T00:00:00Z", "2026-01-01T00:03:20Z"),
    )
    rank1 = ExecutionEventWriter.for_process(
        context,
        rank=1,
        world_size=2,
        utc_clock=_iso_clock("2026-01-01T00:00:50Z"),
    )

    rank0.emit("process_started", phase="train")
    rank0.emit_progress(
        phase="train",
        task_id="epoch",
        completed=1,
        total=10,
        final=True,
        display_metrics={"loss": 9.0},
    )

    # First poll observes only rank 0 so far: its T+200 metric point is
    # applied now, well before rank 1's chronologically-earlier T+50 point
    # ever becomes visible.
    monitor.poll()

    # rank 1's event is only written now (visible in a later poll), but its
    # own timestamp (T+50) precedes rank 0's already-applied T+200 event.
    rank1.emit_progress(
        phase="train",
        task_id="epoch",
        completed=1,
        total=10,
        final=True,
        display_metrics={"loss": 1.0},
    )

    incremental = monitor.poll()
    oracle = _oracle_snapshot(layout, context)
    _assert_snapshot_equivalent(incremental, oracle)

    # Pin the concrete ordering this regression is about: true event-time
    # order, not application/visibility order.
    assert [point.time.isoformat() for point in oracle.metric_history["loss"]] == [
        "2026-01-01T00:00:50+00:00",
        "2026-01-01T00:03:20+00:00",
    ]
    assert [point.value for point in incremental.metric_history["loss"]] == [1.0, 9.0]

    rank0.close()
    rank1.close()


def _run_incremental_equivalence_case(
    tmp_path: Path,
    *,
    seed: int,
    world_size: int,
    checkpoint_probability: float = 0.3,
) -> None:
    """Drive a randomized multi-producer event sequence through incremental
    and full (oracle) folding, asserting equivalence at randomized
    checkpoints and at the end. This is the property/equivalence test that
    pins incremental folding as exactly matching a full re-fold — the spec
    issue #92 requires be written first and kept as the ongoing guarantee.
    """
    rng = random.Random(seed)
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z", world_size=world_size)
    monitor = ExecutionMonitor(layout, "attempt")

    runner_writer = ExecutionEventWriter.for_runner(context)
    rank_writers = [
        ExecutionEventWriter.for_process(context, rank=rank, world_size=world_size)
        for rank in range(world_size)
    ]
    writers: dict[str, ExecutionEventWriter] = {"runner": runner_writer}
    pending: dict[str, list[tuple[str, dict[str, object]]]] = {
        "runner": [("execution_started", {}), ("phase_started", {"phase": "train"})]
    }
    for rank in range(world_size):
        key = f"rank{rank}"
        writers[key] = rank_writers[rank]
        total = rng.randint(3, 6)
        steps: list[tuple[str, dict[str, object]]] = [
            ("process_started", {"phase": "train"}),
            ("task_started", {"phase": "train", "task_id": "epoch"}),
        ]
        for step in range(1, total + 1):
            steps.append(
                (
                    "progress",
                    {
                        "phase": "train",
                        "task_id": "epoch",
                        "completed": step,
                        "total": total,
                        "final": step == total,
                        "throughput": 2.0,
                        "display_metrics": {"loss": 1.0 / step},
                    },
                )
            )
        steps.append(("task_completed", {"phase": "train", "task_id": "epoch"}))
        steps.append(("process_completed", {"phase": "train", "exit_code": 0}))
        pending[key] = steps

    def _rank_keys() -> list[str]:
        return [f"rank{rank}" for rank in range(world_size)]

    def _checkpoint() -> None:
        oracle = _oracle_snapshot(layout, context)
        _assert_snapshot_equivalent(monitor.poll(), oracle)

    runner_closed = False
    while any(pending.values()):
        ranks_done = all(not pending[key] for key in _rank_keys())
        if ranks_done and not runner_closed:
            pending["runner"].extend(
                [("phase_completed", {"phase": "train"}), ("execution_completed", {})]
            )
            runner_closed = True
        active = [key for key, steps in pending.items() if steps]
        key = rng.choice(active)
        event_name, kwargs = pending[key].pop(0)
        writers[key].emit(event_name, **kwargs)
        if rng.random() < checkpoint_probability:
            _checkpoint()

    for writer in writers.values():
        writer.close()
    _checkpoint()


@pytest.mark.parametrize("seed", [1, 2, 3, 17, 42])
def test_incremental_fold_matches_full_fold_multi_rank(tmp_path: Path, seed: int) -> None:
    _run_incremental_equivalence_case(tmp_path, seed=seed, world_size=3)


@pytest.mark.parametrize("seed", [4, 9])
def test_incremental_fold_matches_full_fold_single_rank(tmp_path: Path, seed: int) -> None:
    _run_incremental_equivalence_case(tmp_path, seed=seed, world_size=1)


def test_incremental_fold_matches_full_fold_with_malformed_trailing_line(
    tmp_path: Path,
) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit("process_started", phase="train")
        writer.emit_progress(phase="train", task_id="epoch", completed=1, total=4, final=True)
    stream = context.execution_dir / "rank-0.jsonl"
    with stream.open("ab") as handle:
        handle.write(b"{not valid json\n")

    monitor = ExecutionMonitor(layout, "attempt")
    oracle = _oracle_snapshot(layout, context)
    _assert_snapshot_equivalent(monitor.poll(), oracle)
    assert len(oracle.warnings) == 1
    assert "line 4" in oracle.warnings[0]

    # A second poll of a permanently failed stream stays equivalent too: no
    # new bytes are (successfully) read from the now-isolated path.
    oracle_again = _oracle_snapshot(layout, context)
    _assert_snapshot_equivalent(monitor.poll(), oracle_again)


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
        ],
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


def test_dashboard_ignores_legacy_units_when_reconciling_progress(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z", world_size=2)
    for rank, unit in enumerate(("patches", "unrelated-items")):
        with ExecutionEventWriter.for_process(context, rank=rank) as writer:
            writer.emit_progress(
                phase="arbitrary-phase",
                task_id="unrelated-task",
                completed=rank + 1,
                total=3,
                throughput=2.0,
            )
        path = context.execution_dir / f"rank-{rank}.jsonl"
        payload = json.loads(path.read_text())
        payload["unit"] = unit
        path.write_text(json.dumps(payload) + "\n")
    snapshot = RunMonitor(layout).poll()
    task = snapshot.selected.current_task

    assert task is not None
    assert task.eta_seconds == 0.5
    assert "patches" not in render_snapshot(snapshot.selected)
    assert "unrelated-items" not in render_snapshot(snapshot.selected)

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
        assert "3/6" in rendered
        assert "patches" not in rendered
        assert "unrelated-items" not in rendered


def test_plain_snapshot_marks_missing_throughput_unavailable(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit_progress(phase="opaque", task_id="task", completed=1, total=2)

    rendered = render_snapshot(ExecutionMonitor(layout, "attempt").poll())

    assert "rank-0/opaque/task: running 1/2 rate=--" in rendered

    console = Console(width=80, record=True, color_system=None)
    console.print(
        dashboard_layout(
            RunMonitor(layout).poll(),
            host=None,
            detail=False,
            compact=True,
        )
    )

    assert "overall: 1/2 · --" in console.export_text()


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
    assert "overall: 16/2,000 · 6.5 b/s" in compact
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
                "0, NVIDIA RTX A, 25, 200, 2400\n1, NVIDIA RTX B, 75, 275.25, 2800\n",
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


def test_fleet_dashboard_layout_renders_group_roll_up_and_selection_marker(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
    )
    layout = GroupLayout(entry, manifest.group_id)
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.close()
    RunLayout(entry, "loose").prepare()

    snapshot = FleetMonitor(entry).poll()
    console = Console(width=120, record=True, color_system=None)
    console.print(
        fleet_dashboard_layout(snapshot, selected_index=0, compact=False, now=datetime.now(UTC))
    )
    rendered = console.export_text()

    assert manifest.group_id[:16] in rendered
    assert "run-major" in rendered
    assert "0/0/1" in rendered
    assert "alpha" in rendered
    assert "loose" in rendered
    assert ">" in rendered


def test_group_dashboard_layout_renders_member_steps_and_progress(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember("alpha", ("prepare", "train"))],
    )
    layout = GroupLayout(entry, manifest.group_id)
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("step_started", run_name="alpha", step_name="prepare")
    writer.emit("step_completed", run_name="alpha", step_name="prepare")
    writer.emit("step_started", run_name="alpha", step_name="train")
    writer.close()
    run_layout = RunLayout(entry, "alpha").prepare()
    context = create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as event_writer:
        event_writer.emit("execution_started")
        event_writer.emit_progress(phase="train", task_id="epoch", completed=3, total=10)

    snapshot = FleetMonitor(entry).poll()
    console = Console(width=120, record=True, color_system=None)
    console.print(
        group_dashboard_layout(
            snapshot.groups[0], selected_index=0, compact=False, now=datetime.now(UTC)
        )
    )
    rendered = console.export_text()

    assert "alpha" in rendered
    assert "prepare:completed" in rendered
    assert "train:running" in rendered
    assert "3/10" in rendered


def test_fleet_textual_navigates_fleet_group_run_and_back(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
    )
    group_layout = GroupLayout(entry, manifest.group_id)
    writer = GroupEventWriter(group_layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.close()
    run_layout = RunLayout(entry, "alpha").prepare()
    create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")

    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FleetScreen)
            assert len(app.screen_stack) == 2

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, GroupScreen)
            assert app.screen.group_id == manifest.group_id

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RunScreen)
            run_screen = app.screen
            await _wait_until(pilot, lambda: run_screen.snapshot is not None)
            assert run_screen.snapshot is not None
            assert run_screen.snapshot.layout.run_name == "alpha"

            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, GroupScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, FleetScreen)

    asyncio.run(exercise())


def test_fleet_textual_drills_directly_into_a_loose_run(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    run_layout = RunLayout(entry, "loose").prepare()
    create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")

    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert fleet_rows(app.screen.snapshot) == fleet_rows(fleet_monitor.poll())
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RunScreen)
            run_screen = app.screen
            await _wait_until(pilot, lambda: run_screen.snapshot is not None)
            assert run_screen.snapshot is not None
            assert run_screen.snapshot.layout.run_name == "loose"

    asyncio.run(exercise())


def test_fleet_textual_run_screen_shows_loading_state_before_first_poll(
    tmp_path: Path,
) -> None:
    """Drilling in must not block the UI thread on the run's first poll.

    Gates ``RunMonitor.poll`` on a controllable event so the test can assert
    a visible loading state deterministically, before releasing the poll and
    asserting the real dashboard replaces it once the background worker's
    first snapshot arrives.
    """
    entry = tmp_path / "runs"
    run_layout = RunLayout(entry, "loose").prepare()
    create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")

    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    release = threading.Event()
    original_poll = RunMonitor.poll

    def gated_poll(self: RunMonitor, selected_execution_id: str | None = None) -> object:
        release.wait(timeout=5)
        return original_poll(self, selected_execution_id)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch.object(RunMonitor, "poll", gated_poll):
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, RunScreen)
                run_screen = app.screen
                assert run_screen.snapshot is None
                body = run_screen.query_one("#body", Static)
                assert "loading" in " ".join(_composited_lines(body)).lower()

                release.set()
                await _wait_until(pilot, lambda: run_screen.snapshot is not None)

            assert run_screen.snapshot is not None
            assert run_screen.snapshot.layout.run_name == "loose"
            body = run_screen.query_one("#body", Static)
            assert "loading" not in " ".join(_composited_lines(body)).lower()

    asyncio.run(exercise())


def test_fleet_textual_run_screen_ignores_navigation_before_first_poll(
    tmp_path: Path,
) -> None:
    """P3-3(a) regression: navigation/detail keys while still loading must
    no-op, not crash, since ``RunScreen.snapshot`` is ``None`` until the
    background worker's first poll completes.
    """
    entry = tmp_path / "runs"
    run_layout = RunLayout(entry, "loose").prepare()
    create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")

    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    release = threading.Event()
    original_poll = RunMonitor.poll

    def gated_poll(self: RunMonitor, selected_execution_id: str | None = None) -> object:
        release.wait(timeout=5)
        return original_poll(self, selected_execution_id)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch.object(RunMonitor, "poll", gated_poll):
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, RunScreen)
                run_screen = app.screen
                assert run_screen.snapshot is None

                # Next/previous execution and the detail toggle must all
                # no-op while there is nothing to navigate or toggle yet.
                await pilot.press("j")
                await pilot.press("k")
                await pilot.press("down")
                await pilot.press("up")
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, RunScreen)
                assert run_screen.snapshot is None
                assert run_screen.detail is False

                release.set()
                await _wait_until(pilot, lambda: run_screen.snapshot is not None)

            assert run_screen.snapshot is not None
            assert run_screen.snapshot.layout.run_name == "loose"
            # Navigation works normally once loaded.
            await pilot.press("enter")
            assert run_screen.detail is True

    asyncio.run(exercise())


def test_fleet_textual_stale_worker_from_a_popped_run_screen_never_leaks_forward(
    tmp_path: Path,
) -> None:
    """P3-3(b) regression: a popped screen's slow worker must not leak into
    whatever screen is active when it finally completes.

    Pushes a run screen, pops it before its gated first poll ever finishes,
    and only then releases that stale poll. Textual routes a background
    worker's result back through the *screen instance* that started it
    (``self.app.call_from_thread`` from inside ``_refresh_state``); once
    that screen is popped, Textual itself refuses to deliver the result
    (``NoActiveAppError``) rather than silently applying it anywhere,
    including back onto the popped screen. This test pins that safety
    property — the stale worker's completion is a no-op, never a crash and
    never a leak — rather than merely asserting on
    ``RunScreen._refresh_generation`` in isolation. A fresh drill-down into a
    second run afterward confirms its own independent poll still produces
    the correct snapshot, unaffected by the earlier stale worker.
    """
    entry = tmp_path / "runs"
    for name in ("loose-a", "loose-b"):
        layout = RunLayout(entry, name).prepare()
        create_context(layout, "attempt", "2026-01-01T00:00:00Z")

    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    release_by_run: dict[str, threading.Event] = {
        "loose-a": threading.Event(),
        "loose-b": threading.Event(),
    }
    completed_by_run: dict[str, threading.Event] = {
        "loose-a": threading.Event(),
        "loose-b": threading.Event(),
    }
    original_poll = RunMonitor.poll

    def gated_poll(self: RunMonitor, selected_execution_id: str | None = None) -> object:
        release_by_run[self.layout.run_name].wait(timeout=5)
        result = original_poll(self, selected_execution_id)
        completed_by_run[self.layout.run_name].set()
        return result

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FleetScreen)
            rows = fleet_rows(app.screen.snapshot)
            names = [row.key for row in rows]

            with mock.patch.object(RunMonitor, "poll", gated_poll):
                # Drill into "loose-a" and pop back out before its gated
                # poll ever completes.
                app.screen.selected_index = names.index("loose-a")
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, RunScreen)
                stale_screen = app.screen
                assert stale_screen.snapshot is None

                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, FleetScreen)

                # Release the now-orphaned poll and wait for the underlying
                # RunMonitor.poll call itself (not any UI callback, which
                # this stale worker can no longer deliver) to finish, so the
                # background thread genuinely ran its course rather than the
                # assertions below merely racing it.
                release_by_run["loose-a"].set()
                assert completed_by_run["loose-a"].wait(timeout=5)
                await pilot.pause()

                # The stale result was never applied anywhere: not back onto
                # the popped screen, and not onto whatever is now active.
                assert stale_screen.snapshot is None
                assert isinstance(app.screen, FleetScreen)

                # A fresh drill-down into a different run gets its own
                # independent, correct snapshot, unaffected by the earlier
                # stale worker.
                app.screen.selected_index = names.index("loose-b")
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, RunScreen)
                new_screen = app.screen
                assert new_screen is not stale_screen
                assert new_screen.snapshot is None

                release_by_run["loose-b"].set()
                await _wait_until(pilot, lambda: new_screen.snapshot is not None)

            assert new_screen.snapshot is not None
            assert new_screen.snapshot.layout.run_name == "loose-b"
            # The popped screen's own snapshot is still untouched: its
            # worker's result was discarded, not queued or replayed later.
            assert stale_screen.snapshot is None

    asyncio.run(exercise())


def test_fleet_textual_opens_a_group_directly_via_open_group_id(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
    )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(
        entry,
        fleet_monitor,
        fleet_monitor.poll(),
        watch=False,
        telemetry=False,
        open_group_id=manifest.group_id,
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert len(app.screen_stack) == 3
            assert isinstance(app.screen, GroupScreen)
            assert app.screen.group_id == manifest.group_id

    asyncio.run(exercise())


def test_fleet_textual_selecting_a_run_with_no_executions_does_not_navigate(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    RunLayout(entry, "empty").prepare()
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, FleetScreen)
            assert len(app.screen_stack) == 2

    asyncio.run(exercise())


def test_run_fleet_textual_samples_telemetry_before_the_app_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for the fleet sudo-prompt-inside-the-TUI defect.

    ``run_fleet_textual`` must sample viewer-host telemetry (and take any
    sudo password prompt it needs) on the plain terminal before the Textual
    app is even constructed, mirroring ``run_textual``'s single-run
    ordering. A recording fake sampler and fake ``FleetApp`` pin the exact
    call order without spinning up a real Textual application.
    """
    entry = tmp_path / "runs"
    RunLayout(entry, "loose").prepare()
    fleet_monitor = FleetMonitor(entry)
    snapshot = fleet_monitor.poll()

    events: list[str] = []

    class RecordingSampler:
        def __init__(self, *, allow_sudo_password_prompt: bool = False) -> None:
            events.append("sampler-constructed")
            self.allow_sudo_password_prompt = allow_sudo_password_prompt

        def sample(self) -> str:
            events.append("sampled")
            return "host-sample"

    class RecordingFleetApp:
        def __init__(self, *_args, **kwargs) -> None:
            events.append("app-constructed")
            self.kwargs = kwargs

        def run(self) -> None:
            events.append("app-run")

    monkeypatch.setattr(
        "mammoth.monitor.textual_ui.PsutilViewerTelemetrySampler", RecordingSampler
    )
    monkeypatch.setattr("mammoth.monitor.textual_ui.FleetApp", RecordingFleetApp)

    run_fleet_textual(
        entry,
        fleet_monitor,
        snapshot,
        watch=False,
        telemetry=True,
        interval_seconds=2.0,
        stale_after_seconds=90.0,
    )

    assert events == ["sampler-constructed", "sampled", "app-constructed", "app-run"]


def test_run_fleet_textual_disabled_telemetry_builds_no_sampler_or_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = tmp_path / "runs"
    RunLayout(entry, "loose").prepare()
    fleet_monitor = FleetMonitor(entry)
    snapshot = fleet_monitor.poll()

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("no sampler should be constructed when telemetry is disabled")

    captured: dict = {}

    class RecordingFleetApp:
        def __init__(self, *_args, **kwargs) -> None:
            captured.update(kwargs)

        def run(self) -> None:
            pass

    monkeypatch.setattr(
        "mammoth.monitor.textual_ui.PsutilViewerTelemetrySampler", fail_if_constructed
    )
    monkeypatch.setattr("mammoth.monitor.textual_ui.FleetApp", RecordingFleetApp)

    run_fleet_textual(
        entry,
        fleet_monitor,
        snapshot,
        watch=False,
        telemetry=False,
        interval_seconds=2.0,
        stale_after_seconds=90.0,
    )

    assert captured["telemetry_sampler"] is None
    assert captured["initial_host"] is None


def test_fleet_app_next_run_screen_host_reuses_the_pre_ui_sample_once(tmp_path: Path) -> None:
    """The pre-UI telemetry sample is spent on the first drill-in, not wasted.

    ``push_run_screen`` never constructs its own sampler; it either consumes
    the initial pre-UI sample once or delegates to the already-shared
    sampler, which is exactly what ``FleetApp`` receives from
    ``run_fleet_textual``.
    """
    entry = tmp_path / "runs"
    fleet_monitor = FleetMonitor(entry)

    class CountingSampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> str:
            self.calls += 1
            return f"sampled-{self.calls}"

    sampler = CountingSampler()
    app = FleetApp(
        entry,
        fleet_monitor,
        fleet_monitor.poll(),
        watch=False,
        telemetry=True,
        telemetry_sampler=sampler,
        initial_host="pre-ui-sample",
    )

    assert app._next_run_screen_host() == "pre-ui-sample"
    assert sampler.calls == 0
    assert app._next_run_screen_host() == "sampled-1"
    assert sampler.calls == 1
    assert app._next_run_screen_host() == "sampled-2"
    assert sampler.calls == 2


def test_fleet_textual_run_screen_receives_the_fleet_apps_shared_sampler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Drilling into a run must reuse the injected sampler, not build a new one."""
    entry = tmp_path / "runs"
    run_layout = RunLayout(entry, "loose").prepare()
    create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("FleetApp/RunScreen must not construct their own sampler")

    monkeypatch.setattr(
        "mammoth.monitor.textual_ui.PsutilViewerTelemetrySampler", fail_if_constructed
    )

    class FakeSampler:
        def __init__(self) -> None:
            self.sample_calls = 0

        def sample(self) -> PsutilViewerTelemetry:
            self.sample_calls += 1
            return PsutilViewerTelemetry(
                host_role="viewer",
                hostname="refreshed",
                sampled_at="2026-01-01T00:00:05Z",
                cpu_percent=None,
                memory_percent=None,
                load_average_1m=None,
            )

    fake_sampler = FakeSampler()
    initial_host = PsutilViewerTelemetry(
        host_role="viewer",
        hostname="pre-ui",
        sampled_at="2026-01-01T00:00:00Z",
        cpu_percent=None,
        memory_percent=None,
        load_average_1m=None,
    )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(
        entry,
        fleet_monitor,
        fleet_monitor.poll(),
        watch=False,
        telemetry=True,
        telemetry_sampler=fake_sampler,
        initial_host=initial_host,
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RunScreen)
            assert app.screen.telemetry_sampler is fake_sampler

    asyncio.run(exercise())


def test_fleet_dashboard_layout_windows_loose_runs_around_the_selection(tmp_path: Path) -> None:
    """Regression for the fleet-view overflow defect: window, don't dump everything.

    With 20 loose runs and a viewport too small to show them all, only a
    band around the selected row renders, bounded by "N more above/below"
    markers instead of every row overflowing the terminal.
    """
    entry = tmp_path / "runs"
    for index in range(20):
        RunLayout(entry, f"run-{index:02d}").prepare()
    snapshot = FleetMonitor(entry).poll()

    console = Console(width=120, record=True, color_system=None)
    console.print(
        fleet_dashboard_layout(
            snapshot,
            selected_index=10,
            compact=False,
            now=datetime.now(UTC),
            viewport_rows=20,
        )
    )
    rendered = console.export_text()

    assert "run-10" in rendered
    assert "run-00" not in rendered
    assert "run-19" not in rendered
    assert "6 more above" in rendered
    assert "5 more below" in rendered


def test_fleet_dashboard_layout_reaches_first_and_last_rows_without_a_dangling_marker(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    for index in range(20):
        RunLayout(entry, f"run-{index:02d}").prepare()
    snapshot = FleetMonitor(entry).poll()

    def rendered_for(selected_index: int) -> str:
        console = Console(width=120, record=True, color_system=None)
        console.print(
            fleet_dashboard_layout(
                snapshot,
                selected_index=selected_index,
                compact=False,
                now=datetime.now(UTC),
                viewport_rows=20,
            )
        )
        return console.export_text()

    first = rendered_for(0)
    assert "run-00" in first
    assert "more above" not in first
    assert "10 more below" in first

    last = rendered_for(19)
    assert "run-19" in last
    assert "more below" not in last
    assert "10 more above" in last


def test_group_dashboard_layout_windows_members_around_the_selection(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember(f"member-{index:02d}", ("prepare",)) for index in range(20)],
    )
    snapshot = FleetMonitor(entry).poll()
    group = snapshot.groups[0]
    assert group.group_id == manifest.group_id

    console = Console(width=120, record=True, color_system=None)
    console.print(
        group_dashboard_layout(
            group,
            selected_index=10,
            compact=False,
            now=datetime.now(UTC),
            viewport_rows=20,
        )
    )
    rendered = console.export_text()

    assert "member-10" in rendered
    assert "member-00" not in rendered
    assert "member-19" not in rendered
    assert "5 more above" in rendered
    assert "4 more below" in rendered


def test_row_window_never_exceeds_budget_and_always_shows_the_selected_row() -> None:
    """P1-1 regression: the reviewed standalone repro, plus its general invariant.

    The old ``budget = max(1, max_visible - 2)`` reservation could still
    render one data row plus two marker lines against a budget of 1 (three
    rendered lines for a one-row budget). The fixed function must always
    keep the selected row inside the window and never let the window plus
    whichever markers it actually shows exceed ``max_visible``.
    """
    start, end, hidden_above, hidden_below = _row_window(
        total=5, selected_local_index=2, max_visible=1
    )
    assert start <= 2 < end
    footprint = (end - start) + (1 if hidden_above else 0) + (1 if hidden_below else 0)
    assert footprint <= 1

    for total in (1, 2, 5, 11, 40, 227):
        for max_visible in (1, 2, 3, 7, 9, 15):
            for index in (0, total // 2, total - 1):
                start, end, hidden_above, hidden_below = _row_window(total, index, max_visible)
                assert start <= index < end
                footprint = (
                    (end - start) + (1 if hidden_above else 0) + (1 if hidden_below else 0)
                )
                assert footprint <= max_visible


def test_fleet_textual_windows_many_loose_runs_and_keeps_selection_visible(
    tmp_path: Path,
) -> None:
    """End-to-end: a small terminal windows the fleet screen and j/k reach both ends."""
    entry = tmp_path / "runs"
    for index in range(40):
        RunLayout(entry, f"run-{index:03d}").prepare()
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    def visible_lines() -> list[str]:
        return _composited_lines(app.screen.query_one("#body", Static))

    async def exercise() -> None:
        async with app.run_test(size=(120, 16)) as pilot:
            await pilot.pause()
            rows = fleet_rows(app.screen.snapshot)
            last_index = len(rows) - 1

            lines = visible_lines()
            assert any("run-000" in line for line in lines)
            assert not any("more above" in line for line in lines)
            assert any("more below" in line for line in lines)

            for _ in range(last_index + 5):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == last_index
            lines = visible_lines()
            assert any("run-039" in line for line in lines)
            assert not any("more below" in line for line in lines)

            for _ in range(last_index + 5):
                await pilot.press("k")
            await pilot.pause()
            assert app.screen.selected_index == 0
            lines = visible_lines()
            assert any("run-000" in line for line in lines)
            assert not any("more above" in line for line in lines)

    asyncio.run(exercise())


def test_group_textual_windows_many_members_and_keeps_selection_visible(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember(f"member-{index:03d}", ("prepare",)) for index in range(40)],
    )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(
        entry,
        fleet_monitor,
        fleet_monitor.poll(),
        watch=False,
        telemetry=False,
        open_group_id=manifest.group_id,
    )

    def visible_lines() -> list[str]:
        return _composited_lines(app.screen.query_one("#body", Static))

    async def exercise() -> None:
        async with app.run_test(size=(120, 16)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, GroupScreen)

            lines = visible_lines()
            assert any("member-000" in line for line in lines)
            assert not any("more above" in line for line in lines)
            assert any("more below" in line for line in lines)

            for _ in range(45):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == 39
            lines = visible_lines()
            assert any("member-039" in line for line in lines)
            assert not any("more below" in line for line in lines)

            for _ in range(45):
                await pilot.press("k")
            await pilot.pause()
            assert app.screen.selected_index == 0
            lines = visible_lines()
            assert any("member-000" in line for line in lines)
            assert not any("more above" in line for line in lines)

    asyncio.run(exercise())


def test_group_textual_keeps_the_selected_row_visible_at_the_named_benchmark(
    tmp_path: Path,
) -> None:
    """P1-1 regression, named benchmark: GroupScreen at size (80, 7).

    Reproduces the reviewed failure directly: 20 members, selection moved to
    index 10, terminal size (80, 7) -- small enough that a fixed chrome
    estimate would have left no room for the table header, let alone the
    selected row, once the summary line wraps at this width. Asserts on the
    actual composited output (``render_line``), not the unclipped
    renderable, so a still-broken screen cannot pass by accident.
    """
    entry = tmp_path / "runs"
    manifest = publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember(f"member-{index:02d}", ("prepare",)) for index in range(20)],
    )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(
        entry,
        fleet_monitor,
        fleet_monitor.poll(),
        watch=False,
        telemetry=False,
        open_group_id=manifest.group_id,
    )

    async def exercise() -> None:
        async with app.run_test(size=(80, 7)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, GroupScreen)
            for _ in range(10):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == 10

            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any("member-10" in line for line in lines)

    asyncio.run(exercise())


def test_fleet_textual_keeps_the_selected_row_visible_at_the_named_benchmark(
    tmp_path: Path,
) -> None:
    """P1-1 regression, named benchmark: FleetScreen at size (80, 8)."""
    entry = tmp_path / "runs"
    for index in range(20):
        RunLayout(entry, f"run-{index:02d}").prepare()
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(80, 8)) as pilot:
            await pilot.pause()
            for _ in range(10):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == 10

            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any("run-10" in line for line in lines)

    asyncio.run(exercise())


@pytest.mark.parametrize("width,height", [(60, 4), (78, 6), (80, 8), (100, 12), (120, 16)])
def test_fleet_textual_keeps_a_groups_focused_selection_visible_at_small_sizes(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    """Coverage (a): groups-focused fleet windowing, no loose runs at all."""
    entry = tmp_path / "runs"
    for index in range(20):
        publish_group_manifest(
            entry,
            group_id=f"group-{index:02d}",
            order="run-major",
            members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
        )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            rows = fleet_rows(app.screen.snapshot)
            index = min(10, len(rows) - 1)
            for _ in range(index):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == index

            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any(rows[index].key in line for line in lines)

    asyncio.run(exercise())


@pytest.mark.parametrize("width,height", [(60, 4), (78, 6), (80, 8), (100, 15), (120, 19)])
@pytest.mark.parametrize("fixture", ["loose", "groups", "mixed"])
def test_fleet_textual_keeps_the_last_row_visible_at_small_sizes(
    tmp_path: Path,
    fixture: str,
    width: int,
    height: int,
) -> None:
    """Coverage (b): last-row selection, the header-wrap failure mode, across shapes."""
    entry = tmp_path / "runs"
    if fixture in ("loose", "mixed"):
        for index in range(20 if fixture == "loose" else 12):
            RunLayout(entry, f"run-{index:02d}").prepare()
    if fixture in ("groups", "mixed"):
        count = 20 if fixture == "groups" else 10
        for index in range(count):
            publish_group_manifest(
                entry,
                group_id=f"group-{index:02d}",
                order="run-major",
                members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
            )
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            rows = fleet_rows(app.screen.snapshot)
            last_index = len(rows) - 1
            for _ in range(last_index + 5):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == last_index

            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any(rows[last_index].key in line for line in lines)

    asyncio.run(exercise())


@pytest.mark.parametrize("width,height", [(60, 4), (78, 6), (80, 8), (100, 12)])
def test_fleet_textual_keeps_selection_visible_in_a_mixed_fleet_both_directions(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    """Coverage (c): mixed groups + loose runs, selection tested in each table."""
    entry = tmp_path / "runs"
    for index in range(8):
        publish_group_manifest(
            entry,
            group_id=f"group-{index:02d}",
            order="run-major",
            members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
        )
    for index in range(10):
        RunLayout(entry, f"run-{index:02d}").prepare()
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            rows = fleet_rows(app.screen.snapshot)
            group_count = sum(1 for row in rows if row.kind == "group")
            assert 0 < group_count < len(rows)

            # A selection inside the groups table.
            groups_index = group_count // 2
            for _ in range(groups_index):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == groups_index
            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any(rows[groups_index].key in line for line in lines)

            # A selection inside the loose-runs table.
            loose_index = group_count + (len(rows) - group_count) // 2
            for _ in range(loose_index - groups_index):
                await pilot.press("j")
            await pilot.pause()
            assert app.screen.selected_index == loose_index
            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any(rows[loose_index].key in line for line in lines)

    asyncio.run(exercise())


def test_fleet_textual_shows_the_footer_with_groups_focused_at_an_ordinary_size(
    tmp_path: Path,
) -> None:
    """Round-4 P1 regression: the loose section's own label was never budgeted.

    With groups focused, `_fleet_tables_within_viewport` measured
    `header + groups_table` for the room left over for the loose table but
    then unconditionally appended `loose_label` too, so the real render
    always ran about two rows taller than computed -- clipping the footer
    at a perfectly ordinary size (120x16 with 5 groups + 5 loose runs; the
    footer reappeared only at 120x18).
    """
    entry = tmp_path / "runs"
    for index in range(5):
        publish_group_manifest(
            entry,
            group_id=f"group-{index:02d}",
            order="run-major",
            members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
        )
    for index in range(5):
        RunLayout(entry, f"run-{index:02d}").prepare()
    fleet_monitor = FleetMonitor(entry)
    app = FleetApp(entry, fleet_monitor, fleet_monitor.poll(), watch=False, telemetry=False)

    async def exercise() -> None:
        async with app.run_test(size=(120, 16)) as pilot:
            await pilot.pause()
            assert app.screen.selected_index == 0  # a group: groups are focused

            lines = _composited_lines(app.screen.query_one("#body", Static))
            assert any("Enter drill in" in line for line in lines)

    asyncio.run(exercise())


@pytest.mark.parametrize("width", [60, 78, 80, 100, 120])
@pytest.mark.parametrize("focus", ["groups", "loose"])
def test_fleet_dashboard_layout_never_exceeds_its_viewport_rows_budget(
    tmp_path: Path,
    focus: str,
    width: int,
) -> None:
    """General invariant behind the round-4 fix: rendered height <= viewport_rows.

    Sweeps a range of heights for both the groups-focused and loose-focused
    branches of ``_fleet_tables_within_viewport``, which is the exact claim
    the fix's commit message makes -- this test is what enforces it instead
    of leaving it as an unverified assertion in prose.
    """
    entry = tmp_path / "runs"
    for index in range(5):
        publish_group_manifest(
            entry,
            group_id=f"group-{index:02d}",
            order="run-major",
            members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
        )
    for index in range(5):
        RunLayout(entry, f"run-{index:02d}").prepare()
    snapshot = FleetMonitor(entry).poll()
    rows = fleet_rows(snapshot)
    group_count = sum(1 for row in rows if row.kind == "group")
    selected_index = 0 if focus == "groups" else len(rows) - 1

    avail_width = max(1, width - 2)
    for height in range(4, 21):
        avail_height = max(1, height - 1)
        renderable = fleet_dashboard_layout(
            snapshot,
            selected_index=selected_index,
            compact=avail_width < 80,
            viewport_rows=avail_height,
            width=avail_width,
        )
        actual_height = _rendered_height(renderable, width=avail_width)
        assert actual_height <= avail_height, (
            f"width={width} height={height} focus={focus}: "
            f"rendered {actual_height} rows against a budget of {avail_height}"
        )
    assert 0 < group_count < len(rows)


def test_fleet_dashboard_layout_uses_full_room_at_a_large_viewport(tmp_path: Path) -> None:
    """Round-4 P2 regression: the unfocused table must not waste surplus room.

    A fixed cap of 3 rows for the unfocused table meant a 120x40 viewport
    with only 5 groups and 5 loose runs still showed an overflow marker and
    hid rows despite ~20 empty rows of surplus space. The cap should only
    bind when room is genuinely scarce, not unconditionally.
    """
    entry = tmp_path / "runs"
    for index in range(5):
        publish_group_manifest(
            entry,
            group_id=f"group-{index:02d}",
            order="run-major",
            members=[GroupMember(f"g{index:02d}-member", ("prepare",))],
        )
    for index in range(5):
        RunLayout(entry, f"run-{index:02d}").prepare()
    snapshot = FleetMonitor(entry).poll()
    rows = fleet_rows(snapshot)

    for selected_index in (0, len(rows) - 1):
        renderable = fleet_dashboard_layout(
            snapshot,
            selected_index=selected_index,
            compact=False,
            viewport_rows=39,
            width=118,
        )
        console = Console(width=118, record=True, color_system=None)
        console.print(renderable)
        text = console.export_text()
        assert "more above" not in text
        assert "more below" not in text
        for row in rows:
            assert row.key in text
