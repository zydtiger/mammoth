"""Stable ANSI-free rendering for project-neutral monitor snapshots.

The CLI, tests, logs, and optional Rich UI share this canonical textual view.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mammoth.monitor.model import MonitorSnapshot


def render_snapshot(
    snapshot: MonitorSnapshot,
    *,
    now: datetime | None = None,
    stale_after_seconds: float = 90.0,
) -> str:
    """Render one deterministic plain-text snapshot without terminal escapes."""
    current_time = now or datetime.now(UTC)
    metadata = snapshot.context.metadata
    lines = [
        f"Run: {metadata.run_name}",
        f"Execution: {metadata.execution_id}",
        f"Status: {snapshot.status}",
        f"Created: {metadata.created_at}",
    ]
    if snapshot.lineage:
        lines.append(f"Lineage: {' -> '.join(snapshot.lineage)}")

    lines.append("Phases:")
    if snapshot.phases:
        lines.extend(f"  {name}: {status}" for name, status in sorted(snapshot.phases.items()))
    else:
        lines.append("  (none observed)")

    lines.append("Producers:")
    if snapshot.producers:
        for key, producer in sorted(
            snapshot.producers.items(), key=lambda item: item[0].label
        ):
            status = producer.effective_status(current_time, stale_after_seconds)
            detail = f" phase={producer.phase}" if producer.phase is not None else ""
            lines.append(f"  {key.label}: {status}{detail} sequence={producer.sequence}")
    else:
        lines.append("  (none observed)")

    lines.append("Tasks:")
    if snapshot.tasks:
        for (_producer, _task_id), task in sorted(
            snapshot.tasks.items(), key=lambda item: (item[0][0].label, item[0][1])
        ):
            progress = str(task.completed)
            if task.total is not None:
                progress = f"{progress}/{task.total}"
            rate = (
                f" rate={task.throughput:.1f} b/s"
                if task.throughput is not None
                else " rate=--"
            )
            eta = format_duration(task.eta_seconds)
            eta_text = f" eta={eta}" if eta is not None else ""
            lines.append(
                f"  {task.producer.label}/{task.phase}/{task.task_id}: "
                f"{task.status} {progress}{rate}{eta_text}"
            )
    else:
        lines.append("  (none observed)")

    lines.append("Metrics:")
    if snapshot.metric_history:
        for name, points in sorted(snapshot.metric_history.items()):
            lines.append(f"  {name}: {points[-1].value:g} ({len(points)} points)")
    else:
        lines.append("  (none observed)")

    if snapshot.warnings:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in snapshot.warnings)
    return "\n".join(lines) + "\n"


def format_duration(seconds: float | None) -> str | None:
    """Return a compact duration for generic ETA display."""
    if seconds is None:
        return None
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"
