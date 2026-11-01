"""JSON-shaped documents of monitor snapshots for ``show --json`` and the web server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Union

from mammoth.monitor.fleet import (
    FleetSnapshot,
    GroupMemberSnapshot,
    GroupSnapshot,
    LooseRunSnapshot,
    MemberProgress,
)
from mammoth.monitor.model import MonitorSnapshot, RunSnapshot, TaskState


def _timestamp(value: Union[datetime, None]) -> Union[float, None]:
    return value.timestamp() if value is not None else None


def _progress(progress: Union[MemberProgress, None]) -> Union[dict[str, Any], None]:
    if progress is None:
        return None
    return {
        "completed": progress.completed,
        "total": progress.total,
        "throughput": progress.throughput,
        "eta_seconds": progress.eta_seconds,
    }


def _member(member: GroupMemberSnapshot) -> dict[str, Any]:
    return {
        "run_name": member.run_name,
        "status": member.status,
        "active_step": member.active_step,
        "steps": [{"name": step.name, "status": step.status} for step in member.steps],
        "progress": _progress(member.progress),
        "updated_at": _timestamp(member.updated_at),
        "warnings": list(member.warnings),
    }


def group_document(group: GroupSnapshot) -> dict[str, Any]:
    """Return one group's roll-up and members."""
    return {
        "group_id": group.group_id,
        "pattern": group.pattern,
        "terminal_status": group.terminal_status,
        "total_count": group.total_count,
        "completed_count": group.completed_count,
        "failed_count": group.failed_count,
        "active_member": group.active_member,
        "eta_seconds": group.aggregate_eta_seconds,
        "created_at": _timestamp(group.created_at),
        "updated_at": _timestamp(group.updated_at),
        "finished_at": _timestamp(group.finished_at),
        "members": [_member(member) for member in group.members],
        "warnings": list(group.warnings),
    }


def loose_run_document(run: LooseRunSnapshot) -> dict[str, Any]:
    """Return one loose run's fleet row."""
    return {
        "run_name": run.run_name,
        "status": run.status,
        "progress": _progress(run.progress),
        "updated_at": _timestamp(run.updated_at),
        "finished_at": _timestamp(run.finished_at),
        "warnings": list(run.warnings),
    }


def fleet_document(fleet: FleetSnapshot) -> dict[str, Any]:
    """Return one entry's groups and loose runs."""
    return {
        "entry": str(fleet.entry),
        "groups": [group_document(group) for group in fleet.groups],
        "loose_runs": [loose_run_document(run) for run in fleet.loose_runs],
        "warnings": list(fleet.warnings),
    }


def _task(task: TaskState) -> dict[str, Any]:
    return {
        "producer": task.producer.label,
        "phase": task.phase,
        "task_id": task.task_id,
        "parent_task_id": task.parent_task_id,
        "status": task.status,
        "completed": task.completed,
        "total": task.total,
        "fraction": task.fraction,
        "throughput": task.throughput,
        "eta_seconds": task.eta_seconds,
        "message": task.message,
        "updated_at": _timestamp(task.updated_at),
        "coordinates": dict(task.coordinates),
    }


def _attempt(snapshot: MonitorSnapshot) -> dict[str, Any]:
    metadata = snapshot.context.metadata
    return {
        "execution_id": snapshot.execution_id,
        "status": snapshot.status,
        "created_at": _timestamp(snapshot.created_at),
        "updated_at": _timestamp(snapshot.updated_at),
        "duration_seconds": snapshot.duration_seconds,
        "parent_execution_id": metadata.parent_execution_id,
    }


def run_document(
    run: RunSnapshot,
    *,
    now: Union[datetime, None] = None,
    stale_after_seconds: float = 90.0,
) -> dict[str, Any]:
    """Return one run's selected attempt, lineage histories, and display items."""
    observed_at = now or datetime.now(timezone.utc)
    selected = run.selected
    metrics = run.metric_history
    stage_names = sorted({(point.lane, point.stage) for point in run.stage_history})
    return {
        **_attempt(selected),
        "run_name": run.layout.run_name,
        "lineage": list(selected.lineage),
        "phases": dict(selected.phases),
        "producers": [
            {
                "label": key.label,
                "status": producer.effective_status(observed_at, stale_after_seconds),
                "phase": producer.phase,
                "updated_at": _timestamp(producer.updated_at),
                "exit_code": producer.exit_code,
                "signal": producer.signal,
            }
            for key, producer in sorted(selected.producers.items(), key=lambda item: item[0].label)
        ],
        "tasks": [_task(task) for task in selected.listed_tasks],
        "coordinates": run.logical_coordinates,
        "metrics": [
            {"name": name, "latest": points[-1].value, "count": len(points)}
            for name, points in sorted(metrics.items())
            if points
        ],
        "stages": [{"lane": lane, "stage": stage} for lane, stage in stage_names],
        "display_items": [
            {"kind": item.kind, "name": item.name, "time": _timestamp(item.time), "data": item.data}
            for _key, item in sorted(run.display_items.items())
        ],
        "attempts": [_attempt(snapshot) for snapshot in run.executions],
        "warnings": [*run.warnings, *selected.warnings],
    }
