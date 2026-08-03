"""Responsive Rich renderables for Mammoth's optional Textual monitor.

The Textual application calls this module with project-neutral run snapshots.
The canonical ANSI-free renderer remains in :mod:`mammoth.monitor.render`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from rich.console import Group, RenderableType
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from mammoth.monitor.model import MonitorSnapshot, RunSnapshot, TaskState
from mammoth.monitor.psutil_telemetry import PsutilViewerTelemetry
from mammoth.monitor.render import format_duration

_STATE_STYLE = {
    "pending": "dim",
    "running": "bold green",
    "stale": "bold yellow",
    "failed": "bold red",
    "completed": "bold cyan",
    "interrupted": "bold magenta",
    "skipped": "dim cyan",
}
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
_COORDINATE_ORDER = ("epoch", "global_step", "optimizer_step", "batch", "slide_id")


def dashboard_layout(
    snapshot: RunSnapshot,
    *,
    host: PsutilViewerTelemetry | None,
    detail: bool,
    compact: bool,
    now: datetime | None = None,
) -> RenderableType:
    """Build the wide or compact live dashboard for one logical run."""
    observed_at = now or datetime.now(UTC)
    selected = snapshot.selected
    pieces: list[RenderableType] = [
        _header(snapshot, observed_at),
        _section("RUN PROGRESS"),
        _progress_panel(selected, compact=compact, now=observed_at),
    ]
    if snapshot.metric_history:
        pieces.extend(
            (
                _section("METRIC TRENDS"),
                _metric_panel(snapshot, compact=compact),
            )
        )
    if host is not None:
        pieces.extend(
            (
                _section(f"VIEWER HOST · {host.hostname}"),
                _telemetry_panel(host, compact=compact),
            )
        )
    pieces.extend(
        (
            _section(f"SELECTED EXECUTION · {selected.execution_id}"),
            _execution_panel(selected, detail=detail, compact=compact),
        )
    )
    if detail:
        pieces.extend(
            (
                _section("PRODUCERS AND TASKS"),
                _producer_task_panel(selected, compact=compact, now=observed_at),
            )
        )
    pieces.extend(
        (
            _section("EXECUTION HISTORY"),
            _execution_history(snapshot, compact=compact),
        )
    )
    warnings = [*snapshot.warnings, *selected.warnings]
    if warnings:
        pieces.extend((_section("WARNINGS", style="yellow"), _warnings_panel(warnings)))
    pieces.append(
        Text(
            "j/k or ↑/↓ select · Enter overview/detail · r refresh · q quit",
            style="dim",
            justify="center",
        )
    )
    return Group(*pieces)


def sparkline(values: tuple[float, ...], width: int) -> str:
    """Return a terminal-width-aware sparkline for finite numeric values."""
    finite = tuple(value for value in values if math.isfinite(value))
    if not finite or width <= 0:
        return "--"
    sampled = _downsample(finite, width)
    low = min(sampled)
    high = max(sampled)
    if math.isclose(low, high):
        return _SPARK_BLOCKS[(len(_SPARK_BLOCKS) - 1) // 2] * len(sampled)
    scale = len(_SPARK_BLOCKS) - 1
    return "".join(
        _SPARK_BLOCKS[round((value - low) * scale / (high - low))] for value in sampled
    )


def _downsample(values: tuple[float, ...], width: int) -> tuple[float, ...]:
    """Downsample by deterministic bucket means without changing source history."""
    if len(values) <= width:
        return values
    sampled: list[float] = []
    for index in range(width):
        start = index * len(values) // width
        stop = (index + 1) * len(values) // width
        bucket = values[start:stop]
        sampled.append(sum(bucket) / len(bucket))
    return tuple(sampled)


def _header(snapshot: RunSnapshot, now: datetime) -> Table:
    """Render run identity, selected state, and refresh time."""
    selected = snapshot.selected
    table = Table.grid(expand=True, padding=(0, 1), pad_edge=False)
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="center", ratio=1, overflow="ellipsis")
    table.add_column(justify="right", no_wrap=True)
    status = Text(selected.status.upper(), style=_STATE_STYLE[selected.status])
    if any(
        producer.effective_status(now, 90.0) == "stale"
        for producer in selected.producers.values()
    ):
        status.append(" · STALE PRODUCER", style=_STATE_STYLE["stale"])
    table.add_row(
        Text("Mammoth Monitor", style="bold"),
        Text(snapshot.layout.run_name),
        Text.assemble(status, f" · {now.strftime('%H:%M:%S')}"),
    )
    return table


def _section(title: str, *, style: str = "dim") -> RenderableType:
    """Return a consistent section separator."""
    return Group(Text(), Rule(title, align="left", style=style))


def _progress_panel(
    selected: MonitorSnapshot,
    *,
    compact: bool,
    now: datetime,
) -> RenderableType:
    """Render current coordinates, progress, throughput, and ETA."""
    task = selected.current_task
    coordinates = _ordered_coordinates(selected.current_coordinates)
    summary = Table.grid(expand=True, padding=(0, 1))
    summary.add_column(ratio=1)
    summary.add_column(ratio=1)
    summary.add_column(ratio=1)
    coordinate_text = (
        " · ".join(f"{name} {value}" for name, value in coordinates) or "No coordinates"
    )
    phase = task.phase if task is not None else _current_phase(selected)
    task_name = task.task_id if task is not None else "--"
    summary.add_row(f"Phase {phase}", f"Task {task_name}", coordinate_text)
    if task is None:
        return summary

    progress = Table.grid(expand=True, padding=(0, 1))
    progress.add_column(ratio=3)
    progress.add_column(justify="right", ratio=1)
    if task.total is not None and task.total > 0:
        progress.add_row(
            ProgressBar(total=task.total, completed=min(task.completed, task.total)),
            f"{task.completed}/{task.total} {task.unit or ''}".rstrip(),
        )
    else:
        progress.add_row(Text("Progress total unavailable", style="dim"), str(task.completed))
    eta = format_duration(task.eta_seconds) or "--"
    completion = (
        (now + timedelta(seconds=task.eta_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        if task.eta_seconds is not None
        else "--"
    )
    throughput = f"{task.throughput:.3g} {task.unit or 'unit'}/s" if task.throughput else "--"
    detail = Text(f"Throughput {throughput} · ETA {eta}")
    if not compact:
        detail.append(f" · Estimated completion {completion}")
    return Group(summary, progress, detail)


def _metric_panel(snapshot: RunSnapshot, *, compact: bool) -> Table:
    """Render arbitrary metric histories with conventional metrics first."""
    table = Table(expand=True, box=None, padding=(0, 1), show_header=True)
    table.add_column("Metric", ratio=2, overflow="fold")
    table.add_column("Latest", justify="right", ratio=1)
    if not compact:
        table.add_column("Range", justify="right", ratio=2)
    table.add_column("Trend", ratio=4)
    width = 20 if compact else 48
    for name, points in sorted(snapshot.metric_history.items(), key=_metric_sort_key):
        values = tuple(point.value for point in points)
        latest = values[-1]
        row = [name, f"{latest:.6g}"]
        if not compact:
            row.append(f"{min(values):.6g} … {max(values):.6g}")
        row.append(sparkline(values, width))
        table.add_row(*row)
    return table


def _metric_sort_key(item: tuple[str, tuple[object, ...]]) -> tuple[int, str]:
    """Prioritize loss and learning-rate plots while preserving arbitrary names."""
    name = item[0]
    leaf = name.rsplit("/", 1)[-1].lower()
    if leaf == "loss" or leaf.endswith("_loss"):
        return (0, name)
    if leaf in {"learning_rate", "lr"}:
        return (1, name)
    return (2, name)


def _telemetry_panel(host: PsutilViewerTelemetry, *, compact: bool) -> Table:
    """Render explicitly viewer-host resource telemetry."""
    table = Table.grid(expand=True, padding=(0, 1))
    for _ in range(3 if compact else 5):
        table.add_column(ratio=1)
    load = f"{host.load_average_1m:.2f}" if host.load_average_1m is not None else "--"
    values = [
        "Provenance viewer-host",
        f"CPU {_percentage(host.cpu_percent)}",
        f"Memory {_percentage(host.memory_percent)}",
    ]
    if compact:
        table.add_row(*values)
        table.add_row(f"Sampled {host.sampled_at}", f"Load 1m {load}", "")
    else:
        table.add_row(*values, f"Load 1m {load}", f"Sampled {host.sampled_at}")
    return table


def _execution_panel(
    selected: MonitorSnapshot,
    *,
    detail: bool,
    compact: bool,
) -> RenderableType:
    """Render selected execution identity and optional provenance detail."""
    metadata = selected.context.metadata
    lines = Text()
    lines.append(f"ID {selected.execution_id}\n", style="bold")
    lines.append(
        f"State {selected.status} · Created {metadata.created_at} · "
        f"Duration {format_duration(selected.duration_seconds) or '0s'}\n"
    )
    if detail:
        lines.append(f"Mode {metadata.execution_mode} · World size {metadata.world_size}\n")
        lines.append(f"Previous {metadata.previous_execution_id or '--'}\n")
        lines.append(f"Parent {metadata.parent_execution_id or '--'}\n")
        lines.append(f"Resume checkpoint {metadata.resume_checkpoint or '--'}\n")
        starting_epoch = metadata.starting_epoch if metadata.starting_epoch is not None else "--"
        starting_step = (
            metadata.starting_global_step
            if metadata.starting_global_step is not None
            else "--"
        )
        lines.append(f"Starting epoch {starting_epoch} · Starting global step {starting_step}\n")
        if selected.lineage:
            lineage = " → ".join(selected.lineage)
            lines.append(f"Lineage {lineage}\n")
        if not compact:
            lines.append(f"Intended phases {', '.join(metadata.intended_phases)}")
    return lines


def _producer_task_panel(
    selected: MonitorSnapshot,
    *,
    compact: bool,
    now: datetime,
) -> Table:
    """Render generic producer and task state without project assumptions."""
    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Source", ratio=1)
    table.add_column("Scope", ratio=2, overflow="fold")
    table.add_column("State", ratio=1)
    table.add_column("Progress", ratio=2)
    if not compact:
        table.add_column("Coordinates / metrics", ratio=3, overflow="fold")
    for key, producer in sorted(selected.producers.items(), key=lambda item: item[0].label):
        status = producer.effective_status(now, 90.0)
        row = [key.label, producer.phase or "--", status, f"sequence {producer.sequence}"]
        if not compact:
            row.append("--")
        table.add_row(*row, style=_STATE_STYLE[status])
    for (_producer, _task_id), task in sorted(
        selected.tasks.items(), key=lambda item: (item[0][0].label, item[0][1])
    ):
        row = [
            task.producer.label,
            f"{task.phase}/{task.task_id}",
            task.status,
            _task_progress(task),
        ]
        if not compact:
            details = [f"{key}={value}" for key, value in _ordered_coordinates(task.coordinates)]
            details.extend(
                f"{key}={value:.6g}" for key, value in sorted(task.display_metrics.items())
            )
            row.append(" · ".join(details) or "--")
        table.add_row(*row, style=_STATE_STYLE[task.status])
    return table


def _execution_history(snapshot: RunSnapshot, *, compact: bool) -> Table:
    """Render every available immutable execution with a selected marker."""
    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("", width=1)
    table.add_column("Execution ID", ratio=4, overflow="fold")
    table.add_column("State", ratio=1)
    if not compact:
        table.add_column("Created", ratio=2)
        table.add_column("Duration", justify="right", ratio=1)
    for execution in snapshot.executions:
        row = [
            ">" if execution.execution_id == snapshot.selected_execution_id else "",
            execution.execution_id,
            execution.status,
        ]
        if not compact:
            row.extend(
                (
                    execution.context.metadata.created_at,
                    format_duration(execution.duration_seconds) or "0s",
                )
            )
        table.add_row(*row, style=_STATE_STYLE[execution.status])
    return table


def _warnings_panel(warnings: list[str]) -> Text:
    """Render isolated warnings without obscuring retained valid state."""
    text = Text()
    for warning in warnings:
        text.append(f"! {warning}\n", style="yellow")
    text.append("Valid state from unaffected streams remains available.", style="dim")
    return text


def _ordered_coordinates(
    coordinates: dict[str, int | float | str],
) -> list[tuple[str, int | float | str]]:
    """Order conventional coordinates first and retain every opaque coordinate."""
    priority = {name: index for index, name in enumerate(_COORDINATE_ORDER)}
    return sorted(coordinates.items(), key=lambda item: (priority.get(item[0], 99), item[0]))


def _current_phase(snapshot: MonitorSnapshot) -> str:
    """Return the newest running phase or a stable fallback."""
    running = [name for name, state in snapshot.phases.items() if state == "running"]
    if running:
        return running[-1]
    return next(reversed(snapshot.phases), "--")


def _task_progress(task: TaskState) -> str:
    """Return compact task progress text."""
    total = f"/{task.total}" if task.total is not None else ""
    eta = format_duration(task.eta_seconds)
    suffix = f" · ETA {eta}" if eta is not None else ""
    return f"{task.completed}{total} {task.unit or ''}{suffix}".strip()


def _percentage(value: float | None) -> str:
    """Format optional viewer telemetry without inventing a measurement."""
    return f"{value:.1f}%" if value is not None else "--"
