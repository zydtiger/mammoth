"""Exercise fleet and group folding above one logical run."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mammoth.core import (
    GroupEventWriter,
    GroupLayout,
    GroupMember,
    RunLayout,
    create_execution_context,
    publish_group_manifest,
)
from mammoth.core.events import ExecutionEventWriter
from mammoth.monitor import fleet as fleet_module
from mammoth.monitor.fleet import (
    FleetMonitor,
    discover_group_ids,
    discover_run_names,
)
from mammoth.monitor.render import render_fleet_snapshot, render_group_snapshot


def create_context(layout: RunLayout, execution_id: str, created_at: str):
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
    )


def _clock(*values: str) -> Any:
    """Return a callable yielding ``values`` in order, one per call.

    Lets a test pin exact event timestamps through ``ExecutionEventWriter``'s
    or ``GroupEventWriter``'s ``utc_clock`` option instead of relying on the
    real wall clock.
    """
    iterator = iter(values)
    return lambda: next(iterator)


def make_manifest_group(
    entry: Path,
    *,
    order: str = "run-major",
    members: tuple[GroupMember, ...],
):
    manifest = publish_group_manifest(entry, order=order, members=list(members))
    layout = GroupLayout(entry, manifest.group_id)
    return manifest, layout


def test_discover_group_ids_and_run_names_ignore_incomplete_and_reserved_entries(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    entry.mkdir()
    (entry / "alpha").mkdir()
    (entry / "beta").mkdir()
    (entry / ".mammoth" / "groups" / "complete-group").mkdir(parents=True)
    (entry / ".mammoth" / "groups" / "complete-group" / "manifest.json").write_text("{}")
    (entry / ".mammoth" / "groups" / "incomplete-group").mkdir(parents=True)

    assert discover_group_ids(entry) == ["complete-group"]
    assert discover_run_names(entry) == ["alpha", "beta"]
    assert discover_group_ids(tmp_path / "missing") == []
    assert discover_run_names(tmp_path / "missing") == []


def test_fleet_monitor_reports_empty_fleet_for_an_entry_with_nothing_published(
    tmp_path: Path,
) -> None:
    """An entry without .mammoth/ or any run is a useful empty fleet, never an error."""
    entry = tmp_path / "runs"

    snapshot = FleetMonitor(entry).poll()

    assert snapshot.groups == ()
    assert snapshot.loose_runs == ()
    assert snapshot.warnings == ()
    rendered = render_fleet_snapshot(snapshot)
    assert "(none observed)" in rendered
    assert "\x1b[" not in rendered


def test_fleet_monitor_folds_loose_runs_only(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    layout = RunLayout(entry, "solo").prepare()
    context = create_context(layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")

    snapshot = FleetMonitor(entry).poll()

    assert snapshot.groups == ()
    assert [run.run_name for run in snapshot.loose_runs] == ["solo"]
    assert snapshot.loose_runs[0].status == "completed"


def test_fleet_monitor_skips_attempt_rediscovery_once_the_newest_execution_is_known(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound the per-tick cost: no full directory rescan once nothing changed."""
    entry = tmp_path / "runs"
    layout = RunLayout(entry, "alpha").prepare()
    context = create_context(layout, "first", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")

    calls: list[Any] = []
    original_select_execution = fleet_module.select_execution

    def counting_select_execution(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original_select_execution(*args, **kwargs)

    monkeypatch.setattr(fleet_module, "select_execution", counting_select_execution)

    monitor = FleetMonitor(entry)
    monitor.poll()
    monitor.poll()
    monitor.poll()
    assert len(calls) == 1

    # A newer attempt appears: its directory creation bumps the executions
    # directory mtime, so the very next poll must rediscover exactly once.
    create_context(layout, "second", "2026-01-02T00:00:00Z")
    monitor.poll()
    assert len(calls) == 2
    monitor.poll()
    assert len(calls) == 2


def test_fleet_monitor_excludes_group_members_from_loose_runs(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    RunLayout(entry, "loose").prepare()
    make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare", "train")),),
    )

    snapshot = FleetMonitor(entry).poll()

    assert [run.run_name for run in snapshot.loose_runs] == ["loose"]
    assert len(snapshot.groups) == 1
    assert snapshot.groups[0].members[0].run_name == "alpha"


def test_fleet_monitor_folds_member_step_status_from_group_events_in_schedule_order(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        order="run-major",
        members=(
            GroupMember("alpha", ("prepare", "train")),
            GroupMember("beta", ("prepare", "train")),
        ),
    )
    RunLayout(entry, "alpha").prepare()
    RunLayout(entry, "beta").prepare()
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("step_started", run_name="alpha", step_name="prepare")
    writer.emit("step_completed", run_name="alpha", step_name="prepare")
    writer.emit("step_started", run_name="alpha", step_name="train")
    writer.close()

    snapshot = FleetMonitor(entry).poll()

    group = snapshot.groups[0]
    assert [member.run_name for member in group.members] == ["alpha", "beta"]
    alpha = group.members[0]
    assert [(step.name, step.status) for step in alpha.steps] == [
        ("prepare", "completed"),
        ("train", "running"),
    ]
    assert alpha.active_step == "train"
    beta = group.members[1]
    assert [(step.name, step.status) for step in beta.steps] == [
        ("prepare", "pending"),
        ("train", "pending"),
    ]
    assert beta.status == "pending"
    # alpha has no run artifacts yet, but a run_started event alone still
    # reports it as running rather than pending.
    assert alpha.status == "running"


def test_fleet_monitor_prefers_terminal_group_event_over_run_tail(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare",)),),
    )
    run_layout = RunLayout(entry, "alpha").prepare()
    context = create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")
    writer2 = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer2.emit("group_started")
    writer2.emit("run_started", run_name="alpha")
    writer2.emit("step_started", run_name="alpha", step_name="prepare")
    writer2.emit("step_completed", run_name="alpha", step_name="prepare")
    writer2.emit("run_completed", run_name="alpha")
    writer2.emit("group_completed")
    writer2.close()

    snapshot = FleetMonitor(entry).poll()

    group = snapshot.groups[0]
    # The run's own execution stream never recorded a terminal event (as if
    # the process crashed after its runner started), but the group event
    # stream's authoritative run_completed record still wins.
    assert group.members[0].status == "completed"
    assert group.terminal_status == "group_completed"
    assert group.completed_count == 1
    assert group.failed_count == 0
    assert group.total_count == 1


def test_fleet_monitor_marks_blocked_member_a_failure_for_roll_up(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        order="run-major",
        members=(
            GroupMember("alpha", ("prepare",)),
            GroupMember("beta", ("prepare",)),
        ),
    )
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("run_failed", run_name="alpha", message="boom")
    writer.emit("run_blocked", run_name="beta")
    writer.emit("group_failed", message="boom")
    writer.close()

    snapshot = FleetMonitor(entry).poll()

    group = snapshot.groups[0]
    assert group.members[0].status == "failed"
    assert group.members[1].status == "blocked"
    assert group.completed_count == 0
    assert group.failed_count == 2
    assert group.terminal_status == "group_failed"
    assert group.active_member is None


def test_fleet_monitor_detects_stale_running_member_without_a_terminal_group_event(
    tmp_path: Path,
) -> None:
    """A crashed workflow leaves no terminal group event; staleness comes from the run tail."""
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("train",)),),
    )
    run_layout = RunLayout(entry, "alpha").prepare()
    context = create_context(run_layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit("process_started", phase="train")
    group_writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    group_writer.emit("group_started")
    group_writer.emit("run_started", run_name="alpha")
    group_writer.close()

    monitor = FleetMonitor(entry)
    fresh = monitor.poll(now=datetime.now(UTC))
    assert fresh.groups[0].members[0].status == "running"
    assert fresh.groups[0].terminal_status is None

    much_later = datetime.now(UTC) + timedelta(seconds=1000)
    stale = monitor.poll(now=much_later, stale_after_seconds=90.0)
    assert stale.groups[0].members[0].status == "stale"
    assert stale.groups[0].active_member == "alpha"


def test_fleet_monitor_derives_aggregate_eta_only_when_every_pending_member_has_a_rate(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("train",)), GroupMember("beta", ("train",))),
    )
    alpha_layout = RunLayout(entry, "alpha").prepare()
    alpha_context = create_context(alpha_layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(alpha_context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit_progress(phase="train", task_id="epoch", completed=4, total=10, throughput=2.0)
    beta_layout = RunLayout(entry, "beta").prepare()
    beta_context = create_context(beta_layout, "attempt", "2026-01-01T00:00:00Z")
    with ExecutionEventWriter.for_process(beta_context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit_progress(phase="train", task_id="epoch", completed=1, total=5)

    monitor = FleetMonitor(entry)
    snapshot = monitor.poll()
    group = snapshot.groups[0]
    # beta has no throughput yet, so a synthetic combined ETA would be dishonest.
    assert group.aggregate_eta_seconds is None

    with ExecutionEventWriter.for_process(beta_context, rank=0) as writer:
        writer.emit_progress(
            phase="train", task_id="epoch", completed=2, total=5, throughput=1.0, final=True
        )
    snapshot = monitor.poll()
    group = snapshot.groups[0]
    # remaining: alpha 6 @ 2.0/s, beta 3 @ 1.0/s -> 9 units / 3.0 rate = 3.0s
    assert group.aggregate_eta_seconds == 3.0


def test_fleet_monitor_ignores_a_manifest_that_fails_to_load(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    groups_root = entry / ".mammoth" / "groups" / "broken"
    groups_root.mkdir(parents=True)
    (groups_root / "manifest.json").write_text("not json")

    snapshot = FleetMonitor(entry).poll()

    assert snapshot.groups == ()
    assert any("broken" in warning for warning in snapshot.warnings)


def test_fleet_monitor_isolates_a_malformed_group_event_stream(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare",)),),
    )
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.close()
    with layout.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":1,"sequence":9,"group_id":"' + manifest.group_id)
        handle.write('","event":"run_completed","run_name":"alpha"}\n')

    monitor = FleetMonitor(entry)
    snapshot = monitor.poll()

    # The valid prefix folds normally; the out-of-sequence record is reported
    # as one warning instead of crashing the poll.
    assert snapshot.groups[0].members[0].status == "running"
    assert len(snapshot.warnings) == 1

    # A further poll does not repeat the warning.
    again = monitor.poll()
    assert len(again.warnings) == 1


def test_fleet_monitor_retains_a_partially_written_group_event_line_across_polls(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare",)),),
    )
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.close()
    with layout.events_path.open("ab") as handle:
        handle.write(b'{"schema_version":1,"sequence":2,"group_id":"' + manifest.group_id.encode())

    monitor = FleetMonitor(entry)
    snapshot = monitor.poll()

    assert snapshot.groups[0].terminal_status is None
    assert snapshot.warnings == ()


def test_fleet_monitor_match_groups_loose_runs_ad_hoc(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    RunLayout(entry, "sweep-a").prepare()
    RunLayout(entry, "sweep-b").prepare()
    RunLayout(entry, "other").prepare()

    snapshot = FleetMonitor(entry, match="sweep-*").poll()

    assert snapshot.loose_runs == ()
    assert len(snapshot.groups) == 1
    cohort = snapshot.groups[0]
    assert cohort.manifest is None
    assert cohort.pattern == "sweep-*"
    assert {member.run_name for member in cohort.members} == {"sweep-a", "sweep-b"}
    assert all(member.steps == () for member in cohort.members)


def test_render_group_snapshot_highlights_failed_members_and_is_ansi_free(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare",)), GroupMember("beta", ("prepare",))),
    )
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("run_failed", run_name="alpha", message="boom")
    writer.close()

    snapshot = FleetMonitor(entry).poll()
    rendered = render_group_snapshot(snapshot.groups[0])

    assert "\x1b[" not in rendered
    assert "! alpha: failed" in rendered
    assert "  beta: pending" in rendered


def test_render_fleet_snapshot_lists_group_roll_up_and_loose_runs(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    manifest, layout = make_manifest_group(
        entry,
        members=(GroupMember("alpha", ("prepare",)), GroupMember("beta", ("prepare",))),
    )
    writer = GroupEventWriter(layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("run_completed", run_name="alpha")
    writer.emit("run_blocked", run_name="beta")
    writer.emit("group_failed")
    writer.close()
    RunLayout(entry, "loose").prepare()

    snapshot = FleetMonitor(entry).poll()
    rendered = render_fleet_snapshot(snapshot)

    assert "\x1b[" not in rendered
    assert f"Entry: {entry}" in rendered
    assert "members=1/1/2 (completed/failed/total)" in rendered
    assert "terminal=group_failed" in rendered
    assert "loose: pending" in rendered


def test_render_fleet_snapshot_with_match_omits_the_loose_runs_section(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    RunLayout(entry, "sweep-a").prepare()

    snapshot = FleetMonitor(entry, match="sweep-*").poll()
    rendered = render_fleet_snapshot(snapshot)

    assert "Match: sweep-*" in rendered
    assert "Loose runs:" not in rendered


def test_fleet_monitor_orders_loose_runs_by_descending_recency_then_name(tmp_path: Path) -> None:
    entry = tmp_path / "runs"

    # Finished the longest ago of the two terminal runs.
    old_finished = RunLayout(entry, "loose-old-finished").prepare()
    with ExecutionEventWriter.for_runner(
        create_context(old_finished, "attempt", "2026-01-01T00:00:00Z"),
        utc_clock=_clock("2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"),
    ) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")

    # Finished most recently of everything in this fleet.
    new_finished = RunLayout(entry, "loose-new-finished").prepare()
    with ExecutionEventWriter.for_runner(
        create_context(new_finished, "attempt", "2026-01-02T00:00:00Z"),
        utc_clock=_clock("2026-01-02T00:05:00Z", "2026-01-03T00:00:00Z"),
    ) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")

    # Still running: no terminal event, so its heartbeat is the sort key.
    active = RunLayout(entry, "loose-active").prepare()
    with ExecutionEventWriter.for_process(
        create_context(active, "attempt", "2026-01-01T10:00:00Z"),
        rank=0,
        utc_clock=_clock("2026-01-02T12:00:00Z"),
    ) as writer:
        writer.emit("process_started", phase="train")

    # No execution at all: no usable timestamp, must sort last.
    RunLayout(entry, "loose-no-timestamp").prepare()

    # Two runs tied on the exact same heartbeat: broken deterministically by
    # name rather than reshuffling across polls.
    tie_b = RunLayout(entry, "loose-tie-b").prepare()
    with ExecutionEventWriter.for_process(
        create_context(tie_b, "attempt", "2026-01-01T09:00:00Z"),
        rank=0,
        utc_clock=_clock("2026-01-01T11:00:00Z"),
    ) as writer:
        writer.emit("process_started", phase="train")
    tie_a = RunLayout(entry, "loose-tie-a").prepare()
    with ExecutionEventWriter.for_process(
        create_context(tie_a, "attempt", "2026-01-01T09:00:00Z"),
        rank=0,
        utc_clock=_clock("2026-01-01T11:00:00Z"),
    ) as writer:
        writer.emit("process_started", phase="train")

    expected_order = [
        "loose-new-finished",
        "loose-active",
        "loose-tie-a",
        "loose-tie-b",
        "loose-old-finished",
        "loose-no-timestamp",
    ]

    snapshot = FleetMonitor(entry).poll()
    assert [run.run_name for run in snapshot.loose_runs] == expected_order

    rendered = render_fleet_snapshot(snapshot)
    rendered_order = [
        line.strip().split(":", 1)[0]
        for line in rendered.splitlines()
        if line.startswith("  loose-")
    ]
    assert rendered_order == expected_order


def test_fleet_monitor_orders_groups_by_descending_recency_then_name(tmp_path: Path) -> None:
    entry = tmp_path / "runs"

    old_finished = publish_group_manifest(
        entry,
        group_id="group-old-finished",
        order="run-major",
        members=[GroupMember("old-finished-member", ("prepare",))],
    )
    old_layout = GroupLayout(entry, old_finished.group_id)
    old_writer = GroupEventWriter(
        old_layout.events_path,
        group_id=old_finished.group_id,
        utc_clock=_clock("2026-01-01T00:05:00Z", "2026-01-01T00:20:00Z"),
    )
    old_writer.emit("group_started")
    old_writer.emit("group_completed")
    old_writer.close()

    new_finished = publish_group_manifest(
        entry,
        group_id="group-new-finished",
        order="run-major",
        members=[GroupMember("new-finished-member", ("prepare",))],
    )
    new_layout = GroupLayout(entry, new_finished.group_id)
    new_writer = GroupEventWriter(
        new_layout.events_path,
        group_id=new_finished.group_id,
        utc_clock=_clock("2026-01-02T00:05:00Z", "2026-01-03T01:00:00Z"),
    )
    new_writer.emit("group_started")
    new_writer.emit("group_completed")
    new_writer.close()

    # No terminal group event: its sole member's heartbeat is the sort key.
    active = publish_group_manifest(
        entry,
        group_id="group-active",
        order="run-major",
        members=[GroupMember("active-member", ("prepare",))],
    )
    active_layout = GroupLayout(entry, active.group_id)
    active_writer = GroupEventWriter(active_layout.events_path, group_id=active.group_id)
    active_writer.emit("group_started")
    active_writer.emit("run_started", run_name="active-member")
    active_writer.close()
    member_layout = RunLayout(entry, "active-member").prepare()
    with ExecutionEventWriter.for_process(
        create_context(member_layout, "attempt", "2026-01-02T13:00:00Z"),
        rank=0,
        utc_clock=_clock("2026-01-02T13:00:00Z"),
    ) as writer:
        writer.emit("process_started", phase="train")

    snapshot = FleetMonitor(entry).poll()
    expected_order = ["group-new-finished", "group-active", "group-old-finished"]
    assert [group.group_id for group in snapshot.groups] == expected_order

    rendered = render_fleet_snapshot(snapshot)
    rendered_order = [
        line.strip().split(":", 1)[0]
        for line in rendered.splitlines()
        if line.startswith("  group-")
    ]
    assert rendered_order == expected_order


def test_fleet_monitor_sorts_an_untimestamped_ad_hoc_cohort_last_among_groups(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    timestamped = publish_group_manifest(
        entry,
        group_id="group-timestamped",
        order="run-major",
        members=[GroupMember("timestamped-member", ("prepare",))],
    )
    layout = GroupLayout(entry, timestamped.group_id)
    writer = GroupEventWriter(
        layout.events_path,
        group_id=timestamped.group_id,
        utc_clock=_clock("2026-01-01T00:05:00Z", "2026-01-01T00:20:00Z"),
    )
    writer.emit("group_started")
    writer.emit("group_completed")
    writer.close()

    # An ad-hoc `--match` cohort has no manifest and no group event stream;
    # with no member execution tail yet, it has no usable timestamp at all.
    RunLayout(entry, "sweep-untimestamped").prepare()

    snapshot = FleetMonitor(entry, match="sweep-*").poll()

    assert [group.group_id for group in snapshot.groups] == [
        "group-timestamped",
        "match:sweep-*",
    ]
    assert snapshot.groups[1].finished_at is None
    assert snapshot.groups[1].updated_at is None
    rendered = render_fleet_snapshot(snapshot)
    assert "match:sweep-*" in rendered
