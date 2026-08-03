"""Responsive Rich renderables for Mammoth's optional Textual monitor.

The interactive dashboard preserves the dense wide and readable compact
layouts established by the originating monitor while consuming only Mammoth's
project-neutral run state. The canonical ANSI-free renderer remains in
:mod:`mammoth.monitor.render`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import median

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
from rich.progress import Task as RichTask
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from mammoth.core.events import ExecutionEvent
from mammoth.monitor.model import (
    MetricPoint,
    MonitorSnapshot,
    ProducerKey,
    RunSnapshot,
    TaskState,
)
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
_SHORT_EXECUTION_ID_LENGTH = 8
_COORDINATE_ORDER = (
    "epoch",
    "epoch_total",
    "global_step",
    "optimizer_step",
    "optimizer_step_total",
    "optimizer_total",
    "batch",
    "slide_id",
)
_LIGHT_LOAD_PERCENT = 40.0
_HEAVY_LOAD_PERCENT = 80.0


@dataclass(frozen=True, slots=True)
class _ProgressView:
    """One task progress summary, optionally reconciled across process ranks."""

    task: TaskState
    completed: int
    total: int | None
    unit: str | None
    throughput: float | None
    eta_seconds: float | None
    aggregation_pending: bool = False


class _MonitorSpeedColumn(ProgressColumn):
    """Render producer-derived throughput without using renderer timing."""

    def render(self, task: RichTask) -> Text:
        """Return the immutable throughput stored in the passive Rich task."""
        throughput = task.fields.get("monitor_throughput")
        if not isinstance(throughput, int | float):
            return Text("--")
        unit = task.fields.get("monitor_unit")
        unit_text = _rate_unit(str(unit) if unit else None)
        return Text(f"{float(throughput):.1f} {unit_text}/s", style="progress.data.speed")


class _MetricTrend:
    """Render one width-aware multi-row Braille metric chart."""

    def __init__(
        self,
        label: str,
        values: tuple[float, ...],
        *,
        value_format: str,
        aggregation: str,
        height: int,
    ) -> None:
        self.label = label
        self.values = values
        self.value_format = value_format
        self.aggregation = aggregation
        self.height = height

    def __rich_console__(
        self,
        _console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Fit the chart to the width assigned by its surrounding layout."""
        finite = tuple(value for value in self.values if math.isfinite(value))
        if not finite:
            yield Text(f"{self.label} · unavailable", style="dim")
            return
        yield Text(
            f"{self.label} · latest {format(finite[-1], self.value_format)}",
            style="bold",
        )
        yield Text(
            braille_line_chart(
                finite,
                max(1, options.max_width),
                height=self.height,
                aggregation=self.aggregation,
            )
        )
        yield Text(
            "range "
            f"{format(min(finite), self.value_format)}–"
            f"{format(max(finite), self.value_format)} · {len(finite)} samples",
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )


def dashboard_layout(
    snapshot: RunSnapshot,
    *,
    host: PsutilViewerTelemetry | None,
    detail: bool,
    compact: bool,
    pinned: bool = False,
    stale_after_seconds: float = 90.0,
    refresh_seconds: float = 2.0,
    now: datetime | None = None,
) -> RenderableType:
    """Build the legacy-style wide or compact dashboard for one logical run."""
    observed_at = now or datetime.now(UTC)
    if compact:
        return _compact_layout(
            snapshot,
            host=host,
            detail=detail,
            pinned=pinned,
            stale_after_seconds=stale_after_seconds,
            refresh_seconds=refresh_seconds,
            now=observed_at,
        )
    return _wide_layout(
        snapshot,
        host=host,
        detail=detail,
        pinned=pinned,
        stale_after_seconds=stale_after_seconds,
        refresh_seconds=refresh_seconds,
        now=observed_at,
    )


def braille_line_chart(
    values: tuple[float, ...],
    width: int,
    *,
    height: int,
    aggregation: str = "mean",
) -> str:
    """Render an axis-free line chart at two-by-four Braille sub-cell resolution."""
    finite = tuple(value for value in values if math.isfinite(value))
    if not finite or width <= 0 or height <= 0:
        return ""
    pixel_width = width * 2
    pixel_height = height * 4
    sampled = _downsample(finite, pixel_width, aggregation=aggregation)
    minimum = min(sampled)
    maximum = max(sampled)
    if math.isclose(minimum, maximum):
        y_coordinates = [pixel_height // 2] * len(sampled)
    else:
        y_coordinates = [
            int(round((maximum - value) / (maximum - minimum) * (pixel_height - 1)))
            for value in sampled
        ]
    if len(sampled) == 1:
        coordinates = [(0, y_coordinates[0])]
    else:
        coordinates = [
            (
                int(round(index * (pixel_width - 1) / (len(sampled) - 1))),
                y_coordinate,
            )
            for index, y_coordinate in enumerate(y_coordinates)
        ]

    pixels: set[tuple[int, int]] = set()
    for start, end in zip(coordinates, coordinates[1:], strict=False):
        start_x, start_y = start
        end_x, end_y = end
        steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
        for step in range(steps + 1):
            fraction = step / steps
            pixels.add(
                (
                    int(round(start_x + (end_x - start_x) * fraction)),
                    int(round(start_y + (end_y - start_y) * fraction)),
                )
            )
    if len(coordinates) == 1:
        pixels.add(coordinates[0])

    dot_bits = ((0, 1, 2, 6), (3, 4, 5, 7))
    cells = [[0 for _ in range(width)] for _ in range(height)]
    for x_coordinate, y_coordinate in pixels:
        cell_x, dot_x = divmod(x_coordinate, 2)
        cell_y, dot_y = divmod(y_coordinate, 4)
        cells[cell_y][cell_x] |= 1 << dot_bits[dot_x][dot_y]
    return "\n".join(
        "".join(chr(0x2800 + dots) if dots else " " for dots in row).rstrip()
        for row in cells
    )


def _wide_layout(
    snapshot: RunSnapshot,
    *,
    host: PsutilViewerTelemetry | None,
    detail: bool,
    pinned: bool,
    stale_after_seconds: float,
    refresh_seconds: float,
    now: datetime,
) -> RenderableType:
    """Build the dense wide layout in the originating monitor's hierarchy."""
    selected = snapshot.selected
    heading = f"SELECTED ATTEMPT · {_short_execution_id(selected.execution_id)}"
    if detail:
        heading += " · EXACT"
    pieces: list[RenderableType] = [
        _header(snapshot, now, stale_after_seconds, refresh_seconds),
        _section("RUN PROGRESS"),
        _logical_panel(snapshot, compact=False),
    ]
    if _featured_metric_names(list(snapshot.metric_history.items())):
        pieces.extend((_section("TRAINING TRENDS"), _metric_panel(snapshot, compact=False)))
    if host is not None:
        pieces.extend(
            (
                _section(f"HOST RESOURCES · {host.hostname}"),
                _telemetry_panel(host, compact=False),
            )
        )
    pieces.extend(
        (
            _section(heading),
            _selected_attempt_panel(selected, detail=detail, compact=False, now=now),
            _section("RANKS"),
            _rank_table(selected, now=now, stale_after_seconds=stale_after_seconds),
            _section("ATTEMPT HISTORY"),
            _attempt_history(
                snapshot,
                compact=False,
                pinned=pinned,
                now=now,
                stale_after_seconds=stale_after_seconds,
            ),
        )
    )
    warnings = [*snapshot.warnings, *selected.warnings]
    if warnings:
        pieces.extend((_section("WARNINGS", style="yellow"), _warnings_panel(warnings)))
    pieces.extend((Text(), _footer(pinned)))
    return Group(*pieces)


def _compact_layout(
    snapshot: RunSnapshot,
    *,
    host: PsutilViewerTelemetry | None,
    detail: bool,
    pinned: bool,
    stale_after_seconds: float,
    refresh_seconds: float,
    now: datetime,
) -> RenderableType:
    """Build a stacked narrow layout without retaining wide-only grids or bars."""
    selected = snapshot.selected
    pieces: list[RenderableType] = [
        _header(snapshot, now, stale_after_seconds, refresh_seconds),
        Text(),
        Text("RUN PROGRESS", style="bold"),
        _logical_panel(snapshot, compact=True),
    ]
    if _featured_metric_names(list(snapshot.metric_history.items())):
        pieces.extend(
            (
                Text("\nTRAINING TRENDS", style="bold"),
                _metric_panel(snapshot, compact=True),
            )
        )
    if host is not None:
        pieces.extend(
            (
                Text(f"\nHOST RESOURCES · {host.hostname}", style="bold"),
                _telemetry_panel(host, compact=True),
            )
        )
    heading = f"\nSELECTED ATTEMPT · {_short_execution_id(selected.execution_id)}"
    if detail:
        heading += " · EXACT"
    pieces.extend(
        (
            Text(heading, style="bold"),
            _selected_attempt_panel(selected, detail=detail, compact=True, now=now),
            Text("\nRANKS", style="bold"),
            _compact_ranks(selected, now=now, stale_after_seconds=stale_after_seconds),
            Text("\nATTEMPT HISTORY", style="bold"),
            _attempt_history(
                snapshot,
                compact=True,
                pinned=pinned,
                now=now,
                stale_after_seconds=stale_after_seconds,
            ),
        )
    )
    warnings = [*snapshot.warnings, *selected.warnings]
    if warnings:
        pieces.extend((Text("\nWARNINGS", style="bold yellow"), _warnings_panel(warnings)))
    pieces.extend((Text(), _footer(pinned)))
    return Group(*pieces)


def _header(
    snapshot: RunSnapshot,
    now: datetime,
    stale_after_seconds: float,
    refresh_seconds: float,
) -> Table:
    """Place monitor identity, run name, and refresh state at available edges."""
    state = _execution_state(snapshot.selected, now, stale_after_seconds)
    table = Table.grid(expand=True, padding=(0, 1), pad_edge=False)
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="center", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column(justify="right", no_wrap=True)
    status = Text.assemble(
        (state.upper(), _STATE_STYLE[state]),
        f"  refresh {refresh_seconds:g}s",
    )
    table.add_row(
        Text("Mammoth Monitor", style="bold"),
        Text(snapshot.layout.run_name),
        status,
    )
    return table


def _section(title: str, *, style: str = "dim") -> RenderableType:
    """Separate wide dashboard sections with one consistent blank row and rule."""
    return Group(Text(), Rule(title, align="left", style=style))


def _logical_panel(snapshot: RunSnapshot, *, compact: bool) -> RenderableType:
    """Render resume-aware coordinates and compact execution provenance."""
    selected = snapshot.selected
    coordinates = snapshot.logical_coordinates
    fields: list[str] = []
    epoch = coordinates.get("epoch")
    epoch_total = coordinates.get("epoch_total")
    if epoch is not None or epoch_total is not None:
        fields.append(f"Epoch {_coordinate(epoch, epoch_total)}")
    optimizer = coordinates.get("optimizer_step")
    optimizer_total = coordinates.get(
        "optimizer_step_total",
        coordinates.get("optimizer_total"),
    )
    if optimizer is not None or optimizer_total is not None:
        fields.append(f"Optimizer {_coordinate(optimizer, optimizer_total)}")
    progress = _progress_view(selected)
    eta_text = _logical_eta_text(snapshot, progress)
    if eta_text is not None:
        fields.append(f"ETA {eta_text}")
    source = snapshot.logical_source_execution_id or selected.execution_id
    fields.append(f"Source {_short_execution_id(source)}")
    if len(fields) == 1:
        fields.insert(0, f"Phase {_current_phase(selected)}")

    if compact:
        primary = [
            field
            for field in fields
            if not field.startswith("ETA ")
        ]
        lines = [" | ".join(field.lower() for field in primary)]
        if eta_text is not None:
            lines.append(f"eta {eta_text}")
        return Text("\n".join(lines))
    grid = Table.grid(expand=True, padding=0)
    for index in range(len(fields)):
        if index == 0:
            grid.add_column(justify="left", ratio=1)
        elif index == len(fields) - 1:
            grid.add_column(justify="right", ratio=1)
        else:
            grid.add_column(justify="center", ratio=1)
    grid.add_row(*fields)
    return grid


def _metric_panel(snapshot: RunSnapshot, *, compact: bool) -> RenderableType:
    """Render only conventional training-loss and learning-rate Braille charts."""
    ordered = sorted(snapshot.metric_history.items(), key=_metric_sort_key)
    featured_names = _featured_metric_names(ordered)
    pieces: list[RenderableType] = []
    if featured_names:
        trends = [
            _MetricTrend(
                _metric_label(name),
                tuple(point.value for point in snapshot.metric_history[name]),
                value_format=_metric_value_format(name),
                aggregation="last" if _is_learning_rate(name) else "mean",
                height=3 if compact else 4,
            )
            for name in featured_names
        ]
        if compact:
            for index, trend in enumerate(trends):
                if index:
                    pieces.append(Text())
                pieces.append(trend)
        else:
            grid = Table.grid(expand=True, padding=(0, 2), pad_edge=False)
            for _ in trends:
                grid.add_column(ratio=1)
            grid.add_row(*trends)
            pieces.append(grid)
    return Group(*pieces)


def _featured_metric_names(
    histories: list[tuple[str, tuple[MetricPoint, ...]]],
) -> tuple[str, ...]:
    """Choose at most one conventional loss and learning-rate chart."""
    loss = next((name for name, _points in histories if _is_loss(name)), None)
    learning_rate = next(
        (name for name, _points in histories if _is_learning_rate(name)),
        None,
    )
    return tuple(name for name in (loss, learning_rate) if name is not None)


def _metric_sort_key(
    item: tuple[str, tuple[MetricPoint, ...]],
) -> tuple[int, str]:
    """Prioritize conventional loss and learning-rate names without reinterpreting them."""
    name = item[0]
    if _is_loss(name):
        return (0, name)
    if _is_learning_rate(name):
        return (1, name)
    return (2, name)


def _metric_leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


def _is_loss(name: str) -> bool:
    leaf = _metric_leaf(name)
    return leaf == "loss" or leaf.endswith("_loss")


def _is_learning_rate(name: str) -> bool:
    return _metric_leaf(name) in {"learning_rate", "lr"}


def _metric_label(name: str) -> str:
    if _metric_leaf(name) == "loss":
        return "Training losses · batch"
    if _is_learning_rate(name):
        return "Learning rate"
    return name


def _metric_value_format(name: str) -> str:
    return ".3e" if _is_learning_rate(name) else ".4f"


def _telemetry_panel(host: PsutilViewerTelemetry, *, compact: bool) -> RenderableType:
    """Render every host resource as an identity row plus one metric row."""
    blocks: list[RenderableType] = [
        _resource_block(
            _cpu_label(host),
            _load_metric(f"Util {_percentage(host.cpu_percent)}", host.cpu_percent),
            f"Core {_frequency(host.cpu_frequency_mhz)}",
            compact=compact,
        ),
        _resource_block(
            "RAM",
            _load_metric(_memory_usage(host), host.memory_percent),
            _ram_hardware(host),
            compact=compact,
        ),
    ]
    blocks.extend(
        _resource_block(
            f"GPU {gpu.index} · {gpu.name}",
            _load_metric(
                f"Util {_percentage(gpu.utilization_percent)}",
                gpu.utilization_percent,
            ),
            f"Core {_frequency(gpu.core_clock_mhz)}",
            compact=compact,
        )
        for gpu in host.gpus
    )
    if not host.gpus:
        blocks.append(_resource_block("GPU", "none reported", "--", compact=compact))
    return Group(*blocks)


def _resource_block(
    identity: str,
    left: str | Text,
    right: str | Text,
    *,
    compact: bool,
) -> RenderableType:
    """Keep one full-width identity above edge-aligned resource metrics."""
    prefix = "  " if compact else ""
    metrics = Table.grid(expand=True, padding=0, pad_edge=False)
    metrics.add_column(justify="left", no_wrap=True, overflow="ellipsis")
    metrics.add_column(ratio=1, min_width=2)
    metrics.add_column(justify="right", no_wrap=True, overflow="ellipsis")
    metrics.add_row(Text.assemble(prefix, left), "  ", right)
    return Group(
        Text(prefix + identity, style="italic", no_wrap=True, overflow="ellipsis"),
        metrics,
    )


def _cpu_label(host: PsutilViewerTelemetry) -> str:
    return "CPU" if host.cpu_model_name is None else f"CPU · {host.cpu_model_name}"


def _frequency(value: float | None) -> str:
    return "--" if value is None else f"{value:,.0f} MHz"


def _ram_hardware(host: PsutilViewerTelemetry) -> str:
    values = [value for value in (host.ram_ddr_generation, host.ram_speed) if value]
    return " · ".join(values) if values else "--"


def _selected_attempt_panel(
    selected: MonitorSnapshot,
    *,
    detail: bool,
    compact: bool,
    now: datetime,
) -> RenderableType:
    """Render the selected task and progress, adding provenance only in detail mode."""
    progress = _progress_view(selected)
    pieces: list[RenderableType] = []
    if progress is not None:
        pieces.append(Text(_task_scope(selected, progress.task), style="bold"))
        if progress.aggregation_pending:
            pieces.append(Text("Overall progress pending: waiting for all ranks", style="yellow"))
        elif compact:
            pieces.append(Text(f"  overall: {_progress_text(progress)}"))
        else:
            pieces.append(_progress_renderable(progress, label="Overall"))
    else:
        pieces.append(Text(f"{_current_phase(selected)} / --", style="bold"))

    if detail:
        pieces.append(_execution_details(selected, compact=compact, now=now))
        phase_history = _phase_history(selected)
        if phase_history:
            pieces.append(Text("Phase history:\n" + "\n".join(phase_history)))
        logs = _rank_log_paths(selected)
        if logs:
            pieces.append(Text(f"Rank logs (paths only): {', '.join(logs)}", style="dim"))
    return Group(*pieces)


def _execution_details(
    selected: MonitorSnapshot,
    *,
    compact: bool,
    now: datetime,
) -> RenderableType:
    """Render exact execution identity, lineage, terminal state, and duration."""
    metadata = selected.context.metadata
    terminal = _terminal_event(selected)
    rows = [
        ("ID", selected.execution_id),
        ("State", selected.status),
        ("Mode", f"{metadata.execution_mode} · world size {metadata.world_size}"),
        ("Created", metadata.created_at),
        ("Previous", metadata.previous_execution_id or "--"),
        ("Parent", metadata.parent_execution_id or "--"),
        ("Resume checkpoint", str(metadata.resume_checkpoint or "--")),
        ("Starting epoch", _optional(metadata.starting_epoch)),
        ("Starting global step", _optional(metadata.starting_global_step)),
        ("Duration", _attempt_duration(selected, now)),
        ("Terminal", _terminal_text(terminal) if terminal is not None else "--"),
    ]
    if compact:
        return Text("\n".join(f"  {label.lower()}: {value}" for label, value in rows))
    table = Table.grid(expand=True, padding=0)
    table.add_column(style="dim", justify="left")
    table.add_column(justify="right", overflow="fold")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _progress_view(selected: MonitorSnapshot) -> _ProgressView | None:
    """Return compatible attempt-wide progress without summing replicated counters."""
    task = _overview_task(selected)
    if task is None:
        return None
    expected = selected.context.metadata.world_size
    if task.producer.source != "process" or expected <= 1:
        return _task_progress_view(task)
    matching = [
        item
        for item in selected.tasks.values()
        if item.producer.source == "process"
        and item.phase == task.phase
        and item.task_id == task.task_id
    ]
    ranks = {item.producer.rank for item in matching}
    if len(ranks) < expected:
        view = _task_progress_view(task)
        return _ProgressView(
            task=view.task,
            completed=view.completed,
            total=view.total,
            unit=view.unit,
            throughput=view.throughput,
            eta_seconds=view.eta_seconds,
            aggregation_pending=True,
        )
    units = {item.unit for item in matching}
    if len(units) != 1 or any(item.total is None for item in matching):
        return _task_progress_view(task)
    counts = {(item.completed, item.total) for item in matching}
    if len(counts) == 1:
        primary = min(matching, key=lambda item: item.producer.rank or 0)
        return _task_progress_view(primary)
    completed = sum(item.completed for item in matching)
    total = sum(item.total or 0 for item in matching)
    throughputs = [item.throughput for item in matching]
    throughput = (
        sum(value for value in throughputs if value is not None)
        if all(value is not None for value in throughputs)
        else None
    )
    eta = (
        (total - completed) / throughput
        if throughput is not None and throughput > 0 and completed < total
        else None
    )
    return _ProgressView(
        task=task,
        completed=completed,
        total=total,
        unit=task.unit,
        throughput=throughput,
        eta_seconds=eta,
    )


def _overview_task(selected: MonitorSnapshot) -> TaskState | None:
    """Prefer the highest active ancestor with progress for the run overview."""
    task = selected.current_task
    if task is None:
        return None
    by_id = {
        item.task_id: item
        for item in selected.tasks.values()
        if item.producer == task.producer
    }
    current = task
    seen = {task.task_id}
    while current.parent_task_id is not None and current.parent_task_id not in seen:
        parent = by_id.get(current.parent_task_id)
        if parent is None or parent.status != "running":
            break
        seen.add(parent.task_id)
        if parent.total is not None or parent.completed > 0 or parent.throughput is not None:
            task = parent
        current = parent
    return task


def _task_progress_view(task: TaskState) -> _ProgressView:
    return _ProgressView(
        task=task,
        completed=task.completed,
        total=task.total,
        unit=task.unit,
        throughput=task.throughput,
        eta_seconds=task.eta_seconds,
    )


def _progress_renderable(progress_view: _ProgressView, *, label: str) -> Progress:
    """Build a passive Rich progress row with label, bar, count, and speed."""
    progress = Progress(
        "{task.description}",
        BarColumn(bar_width=None),
        TextColumn("{task.fields[monitor_count]}"),
        _MonitorSpeedColumn(),
        auto_refresh=False,
        expand=True,
    )
    total = progress_view.total or max(progress_view.completed, 1)
    progress.add_task(
        label,
        total=total,
        completed=min(progress_view.completed, total),
        monitor_count=(
            f"{progress_view.completed:,}/{progress_view.total:,}"
            if progress_view.total is not None
            else f"{progress_view.completed:,}/?"
        ),
        monitor_throughput=progress_view.throughput,
        monitor_unit=progress_view.unit,
    )
    return progress


def _rank_table(
    selected: MonitorSnapshot,
    *,
    now: datetime,
    stale_after_seconds: float,
) -> Table:
    """Render process-local liveness, progress, throughput, and current task."""
    table = Table(
        box=None,
        expand=True,
        padding=(0, 1),
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("Task", width=22, no_wrap=True, overflow="ellipsis")
    table.add_column("Rank", ratio=1)
    table.add_column("State", ratio=1)
    table.add_column("Last", ratio=1)
    table.add_column("Progress", ratio=1)
    table.add_column("Rate", ratio=1)
    table.add_column("Task ETA", justify="right", ratio=1)
    for rank in range(selected.context.metadata.world_size):
        key = ProducerKey("process", rank)
        producer = selected.producers.get(key)
        task = _producer_current_task(selected, key)
        if producer is None:
            table.add_row("--", str(rank), Text("PENDING", style="dim"), "--", "--", "--", "--")
            continue
        state = producer.effective_status(now, stale_after_seconds)
        table.add_row(
            task.task_id if task is not None else "--",
            str(rank),
            Text(state.upper(), style=_STATE_STYLE[state]),
            _age(producer.updated_at, now),
            _task_counts(task),
            _task_rate(task),
            format_duration(task.eta_seconds) if task is not None else "--",
        )
    return table


def _compact_ranks(
    selected: MonitorSnapshot,
    *,
    now: datetime,
    stale_after_seconds: float,
) -> Text:
    """Render one stable line per expected process rank in compact mode."""
    lines = Text()
    for rank in range(selected.context.metadata.world_size):
        key = ProducerKey("process", rank)
        producer = selected.producers.get(key)
        task = _producer_current_task(selected, key)
        if producer is None:
            lines.append(f"  {rank}  PENDING  task=--\n", style="dim")
            continue
        state = producer.effective_status(now, stale_after_seconds)
        last = producer.last_event or "--"
        lines.append(f"  {rank}  ")
        lines.append(f"{state.upper():<11}", style=_STATE_STYLE[state])
        lines.append(
            f" {task.task_id if task is not None else '--'}  "
            f"{_task_counts(task)}  {_compact_task_rate(task)}  "
            f"{last} {_age(producer.updated_at, now)}\n"
        )
    return lines


def _attempt_history(
    snapshot: RunSnapshot,
    *,
    compact: bool,
    pinned: bool,
    now: datetime,
    stale_after_seconds: float,
) -> RenderableType:
    """Render compact attempt identity with lineage, phase, start, and gap."""
    executions = (snapshot.selected,) if pinned else snapshot.executions
    if compact:
        lines = Text()
        for execution in executions:
            marker = "> " if execution.execution_id == snapshot.selected_execution_id else "  "
            state = _execution_state(execution, now, stale_after_seconds)
            lines.append(marker)
            lines.append(f"{_short_execution_id(execution.execution_id):<10}")
            lines.append(f"{state.upper():<12}", style=_STATE_STYLE[state])
            lines.append(
                f"{_lineage_text(execution):<22} phase={_current_phase(execution)}\n"
            )
        return lines

    table = Table(
        box=None,
        expand=True,
        padding=(0, 1),
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("", width=1)
    table.add_column("ID")
    table.add_column("State")
    table.add_column("Lineage")
    table.add_column("Phase")
    table.add_column("Started")
    table.add_column("Gap", justify="right")
    for execution in executions:
        state = _execution_state(execution, now, stale_after_seconds)
        index = snapshot.executions.index(execution)
        table.add_row(
            ">" if execution.execution_id == snapshot.selected_execution_id else "",
            _short_execution_id(execution.execution_id),
            Text(state.upper(), style=_STATE_STYLE[state]),
            _lineage_text(execution),
            _current_phase(execution),
            _display_time(execution.context.metadata.created_at),
            _downtime_before(snapshot.executions, index),
        )
    return table


def _producer_current_task(
    selected: MonitorSnapshot,
    producer: ProducerKey,
) -> TaskState | None:
    tasks = [task for task in selected.tasks.values() if task.producer == producer]
    tasks.sort(key=lambda task: task.updated_at or selected.created_at)
    running = [task for task in tasks if task.status == "running"]
    active_parent_ids = {
        task.parent_task_id for task in running if task.parent_task_id is not None
    }
    active_leaves = [task for task in running if task.task_id not in active_parent_ids]
    return (active_leaves or running or tasks)[-1] if tasks else None


def _task_scope(selected: MonitorSnapshot, task: TaskState) -> str:
    """Render a bounded same-producer parent chain for one generic task."""
    tasks = {
        item.task_id: item
        for item in selected.tasks.values()
        if item.producer == task.producer
    }
    names = [task.task_id]
    parent = task.parent_task_id
    seen = {task.task_id}
    while parent is not None and parent not in seen:
        names.append(parent)
        seen.add(parent)
        parent_task = tasks.get(parent)
        parent = parent_task.parent_task_id if parent_task is not None else None
    names.reverse()
    return f"{task.phase} / " + " / ".join(names)


def _phase_history(selected: MonitorSnapshot) -> tuple[str, ...]:
    events = [
        event
        for event in selected.events
        if event.event in {"phase_completed", "phase_failed", "phase_skipped"}
    ]
    return tuple(
        f"{event.phase}{f' rank={event.rank}' if event.rank is not None else ''}: "
        f"{_terminal_text(event)}"
        for event in events
    )


def _terminal_event(selected: MonitorSnapshot) -> ExecutionEvent | None:
    runner_terminal = next(
        (
            event
            for event in reversed(selected.events)
            if event.event
            in {"execution_completed", "execution_failed", "execution_interrupted"}
        ),
        None,
    )
    if runner_terminal is not None:
        return runner_terminal
    process_terminals = [
        event for event in selected.events if event.event == "process_completed"
    ]
    interrupted = [event for event in process_terminals if event.signal is not None]
    failed = [event for event in process_terminals if event.exit_code not in {None, 0}]
    preferred = interrupted or failed or process_terminals
    return preferred[-1] if preferred else None


def _terminal_text(event: ExecutionEvent) -> str:
    fields: list[str] = [event.event]
    status = event.extensions.get("status")
    if isinstance(status, str):
        fields.append(f"status={status}")
    if event.exit_code is not None:
        fields.append(f"exit_code={event.exit_code}")
    if event.signal is not None:
        fields.append(f"signal={event.signal}")
    if event.duration_seconds is not None:
        fields.append(f"duration={event.duration_seconds:.1f}s")
    error_type = event.extensions.get("error_type")
    if isinstance(error_type, str):
        fields.append(f"error_type={error_type}")
    if event.message is not None:
        fields.append(f"message={event.message}")
    return " ".join(fields)


def _rank_log_paths(selected: MonitorSnapshot) -> tuple[str, ...]:
    return tuple(
        str(selected.context.execution_dir / f"rank-{rank}.log")
        for rank in range(selected.context.metadata.world_size)
    )


def _execution_state(
    snapshot: MonitorSnapshot,
    now: datetime,
    stale_after_seconds: float,
) -> str:
    if snapshot.status == "running" and any(
        producer.effective_status(now, stale_after_seconds) == "stale"
        for key, producer in snapshot.producers.items()
        if key.source == "process"
    ):
        return "stale"
    return snapshot.status


def _lineage_text(snapshot: MonitorSnapshot) -> str:
    metadata = snapshot.context.metadata
    if metadata.parent_execution_id is not None:
        return (
            f"{_short_execution_id(metadata.parent_execution_id)} -> "
            f"ep{_optional(metadata.starting_epoch)}"
        )
    if metadata.resume_checkpoint is not None:
        return f"resume ? -> ep{_optional(metadata.starting_epoch)}"
    if metadata.previous_execution_id is not None:
        return f"fresh (prev {_short_execution_id(metadata.previous_execution_id)})"
    return "fresh"


def _current_phase(snapshot: MonitorSnapshot) -> str:
    running = [name for name, state in snapshot.phases.items() if state == "running"]
    if running:
        return running[-1]
    if snapshot.phases:
        return next(reversed(snapshot.phases))
    task = snapshot.current_task
    return task.phase if task is not None else "--"


def _attempt_duration(snapshot: MonitorSnapshot, now: datetime) -> str:
    terminal = _terminal_event(snapshot)
    if terminal is not None and terminal.duration_seconds is not None:
        return f"{terminal.duration_seconds:.1f}s"
    endpoint = now if snapshot.status == "running" else snapshot.updated_at
    return format_duration(max(0.0, (endpoint - snapshot.created_at).total_seconds())) or "0s"


def _downtime_before(executions: tuple[MonitorSnapshot, ...], index: int) -> str:
    if index == 0:
        return "--"
    previous = executions[index - 1]
    current = executions[index]
    return format_duration(
        max(0.0, (current.created_at - previous.updated_at).total_seconds())
    ) or "0s"


def _display_time(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone().strftime("%b %d %H:%M")


def _task_counts(task: TaskState | None) -> str:
    if task is None:
        return "--"
    return (
        f"{task.completed:,}/{task.total:,}"
        if task.total is not None
        else f"{task.completed:,}/?"
    )


def _task_rate(task: TaskState | None) -> str:
    if task is None or task.throughput is None:
        return "--"
    return f"{task.throughput:.1f} {_rate_unit(task.unit)}/s"


def _compact_task_rate(task: TaskState | None) -> str:
    if task is None or task.throughput is None:
        return "--"
    return f"{task.throughput:.1f} {_rate_unit(task.unit)}/s"


def _progress_text(progress: _ProgressView) -> str:
    counts = (
        f"{progress.completed:,}/{progress.total:,}"
        if progress.total is not None
        else f"{progress.completed:,}/?"
    )
    unit = f" {progress.unit}" if progress.unit else ""
    rate = (
        f" · {progress.throughput:.1f} {_rate_unit(progress.unit)}/s"
        if progress.throughput is not None
        else ""
    )
    eta = format_duration(progress.eta_seconds)
    eta_text = f" · ETA {eta}" if eta is not None else ""
    return f"{counts}{unit}{rate}{eta_text}"


def _rate_unit(unit: str | None) -> str:
    return "b" if unit == "microbatch" else unit or "unit"


def _logical_eta_text(
    snapshot: RunSnapshot,
    progress: _ProgressView | None,
) -> str | None:
    """Estimate a resume-aware optimizer horizon, falling back to task ETA."""
    coordinates = snapshot.logical_coordinates
    optimizer = coordinates.get("optimizer_step")
    optimizer_total = coordinates.get(
        "optimizer_step_total",
        coordinates.get("optimizer_total"),
    )
    if isinstance(optimizer, int) and isinstance(optimizer_total, int):
        remaining = max(optimizer_total - optimizer, 0)
        if remaining == 0 or snapshot.selected.status == "completed":
            return "COMPLETED"
        samples = _optimizer_interval_samples(snapshot)
        if not samples:
            return "-- (calibrating)"
        return f"~{format_duration(remaining * median(samples)) or '0s'}"
    if progress is not None:
        return format_duration(progress.eta_seconds) or "--"
    return None


def _optimizer_interval_samples(snapshot: RunSnapshot) -> tuple[float, ...]:
    """Measure recent positive optimizer-step intervals from one metric timeline."""
    for _name, points in sorted(snapshot.metric_history.items(), key=_metric_sort_key):
        observations = [
            (point.time, point.coordinates.get("optimizer_step"))
            for point in points
            if isinstance(point.coordinates.get("optimizer_step"), int)
        ]
        samples: list[float] = []
        previous: tuple[datetime, object] | None = None
        for observed_at, step in observations:
            if previous is None:
                previous = (observed_at, step)
                continue
            previous_at, previous_step = previous
            if not isinstance(step, int) or not isinstance(previous_step, int):
                previous = (observed_at, step)
                continue
            completed = step - previous_step
            elapsed = (observed_at - previous_at).total_seconds()
            if completed > 0 and elapsed > 0:
                samples.append(elapsed / completed)
            if completed != 0:
                previous = (observed_at, step)
        if samples:
            return tuple(samples[-32:])
    return ()


def _downsample(
    values: tuple[float, ...],
    width: int,
    *,
    aggregation: str,
) -> tuple[float, ...]:
    if len(values) <= width:
        return values
    sampled: list[float] = []
    for index in range(width):
        start = index * len(values) // width
        stop = (index + 1) * len(values) // width
        bucket = values[start:stop]
        sampled.append(sum(bucket) / len(bucket) if aggregation == "mean" else bucket[-1])
    return tuple(sampled)


def _ordered_coordinates(
    coordinates: dict[str, int | float | str],
) -> list[tuple[str, int | float | str]]:
    priority = {name: index for index, name in enumerate(_COORDINATE_ORDER)}
    return sorted(coordinates.items(), key=lambda item: (priority.get(item[0], 99), item[0]))


def _coordinate(value: object | None, total: object | None) -> str:
    return f"{_optional(value)}/{_optional(total)}"


def _short_execution_id(execution_id: str | None) -> str:
    return execution_id[-_SHORT_EXECUTION_ID_LENGTH:] if execution_id is not None else "--"


def _optional(value: object | None) -> str:
    return "--" if value is None else str(value)


def _age(value: datetime | None, now: datetime) -> str:
    if value is None:
        return "--"
    seconds = max(0.0, (now - value).total_seconds())
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _percentage(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "--"


def _memory_usage(host: PsutilViewerTelemetry) -> str:
    if host.memory_used_bytes is None and host.memory_total_bytes is None:
        return f"Util {_percentage(host.memory_percent)}"
    used = "--" if host.memory_used_bytes is None else f"{host.memory_used_bytes / 1024**3:.1f}"
    total = "--" if host.memory_total_bytes is None else f"{host.memory_total_bytes / 1024**3:.1f}"
    return f"{used}/{total} GiB ({_percentage(host.memory_percent)})"


def _load_metric(value: str, utilization_percent: float | None) -> Text:
    if utilization_percent is None:
        style = ""
    elif utilization_percent < _LIGHT_LOAD_PERCENT:
        style = "green"
    elif utilization_percent <= _HEAVY_LOAD_PERCENT:
        style = "yellow"
    else:
        style = "red"
    return Text(value, style=style)


def _warnings_panel(warnings: list[str]) -> Text:
    text = Text()
    for warning in warnings:
        text.append(f"! {warning}\n", style="yellow")
    text.append("Last valid state retained; other streams remain usable.", style="dim")
    return text


def _footer(pinned: bool) -> Text:
    instructions = (
        "r refresh · q quit"
        if pinned
        else "Enter exact/overview · ↑/↓ or j/k select · r refresh · q quit"
    )
    return Text(instructions, style="dim", justify="center")
