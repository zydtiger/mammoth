"""Passive discovery and state folding for Mammoth execution artifacts.

The CLI and optional interactive UI consume this module. It reads only core
metadata and JSONL events, so monitoring never imports a consuming project.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mammoth.core.events import (
    ExecutionEvent,
    ExecutionEventReadError,
    ExecutionEventTailReader,
)
from mammoth.core.execution import (
    ExecutionContext,
    ExecutionMetadata,
    join_execution_context,
)
from mammoth.core.layout import RunLayout

RunStatus = Literal["pending", "running", "completed", "failed", "interrupted"]
ScopeStatus = Literal["pending", "running", "completed", "failed", "skipped", "stale"]

_RANK_STREAM_PATTERN = re.compile(r"^rank-(?P<rank>[0-9]+)\.jsonl$")
_TERMINAL_RUN_STATUS: dict[str, RunStatus] = {
    "execution_completed": "completed",
    "execution_failed": "failed",
    "execution_interrupted": "interrupted",
}
_TERMINAL_SCOPE_STATUS: dict[str, ScopeStatus] = {
    "phase_completed": "completed",
    "phase_failed": "failed",
    "phase_skipped": "skipped",
    "task_completed": "completed",
    "task_failed": "failed",
    "task_skipped": "skipped",
    "process_completed": "completed",
}


@dataclass(frozen=True, slots=True)
class ProducerKey:
    """Stable stream identity for a runner or process rank."""

    source: str
    rank: int | None = None

    @property
    def label(self) -> str:
        """Return a concise human-readable producer label."""
        return "runner" if self.rank is None else f"rank-{self.rank}"


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One time-indexed metric value with opaque project coordinates."""

    time: datetime
    value: float
    producer: ProducerKey
    coordinates: Mapping[str, int | float | str]


@dataclass(slots=True)
class TaskState:
    """Folded state for one producer-owned task."""

    producer: ProducerKey
    phase: str
    task_id: str
    parent_task_id: str | None = None
    status: ScopeStatus = "pending"
    completed: int = 0
    total: int | None = None
    unit: str | None = None
    throughput: float | None = None
    message: str | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    coordinates: dict[str, int | float | str] = field(default_factory=dict)
    display_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def fraction(self) -> float | None:
        """Return bounded completion fraction when a nonzero total exists."""
        if self.total is None or self.total <= 0:
            return None
        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def eta_seconds(self) -> float | None:
        """Estimate remaining seconds from observed task work only."""
        if (
            self.status != "running"
            or self.total is None
            or self.completed <= 0
            or self.completed >= self.total
        ):
            return None
        if self.throughput is not None and self.throughput > 0:
            return (self.total - self.completed) / self.throughput
        if self.started_at is None or self.updated_at is None:
            return None
        elapsed = (self.updated_at - self.started_at).total_seconds()
        if elapsed <= 0:
            return None
        return elapsed * (self.total - self.completed) / self.completed


@dataclass(slots=True)
class ProducerState:
    """Folded process or runner state and its last observation time."""

    key: ProducerKey
    status: ScopeStatus = "pending"
    phase: str | None = None
    sequence: int = 0
    updated_at: datetime | None = None
    exit_code: int | None = None
    signal: int | str | None = None

    def effective_status(self, now: datetime, stale_after_seconds: float) -> ScopeStatus:
        """Return ``stale`` for a nonterminal producer with an old heartbeat."""
        if self.status != "running" or self.updated_at is None:
            return self.status
        age = (now - self.updated_at).total_seconds()
        return "stale" if age > stale_after_seconds else self.status


@dataclass(slots=True)
class MonitorSnapshot:
    """Complete project-neutral state reconstructed from one execution."""

    context: ExecutionContext
    status: RunStatus = "pending"
    phases: dict[str, ScopeStatus] = field(default_factory=dict)
    producers: dict[ProducerKey, ProducerState] = field(default_factory=dict)
    tasks: dict[tuple[ProducerKey, str], TaskState] = field(default_factory=dict)
    metric_history: dict[str, list[MetricPoint]] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    @property
    def execution_id(self) -> str:
        """Return the selected immutable attempt ID."""
        return self.context.metadata.execution_id

    @property
    def created_at(self) -> datetime:
        """Return the execution creation time as normalized UTC."""
        return parse_time(self.context.metadata.created_at)

    @property
    def updated_at(self) -> datetime:
        """Return the latest observed event time, falling back to creation."""
        observed = [
            producer.updated_at
            for producer in self.producers.values()
            if producer.updated_at is not None
        ]
        observed.extend(
            task.updated_at for task in self.tasks.values() if task.updated_at is not None
        )
        return max(observed, default=self.created_at)

    @property
    def duration_seconds(self) -> float:
        """Return the nonnegative observed execution duration."""
        return max(0.0, (self.updated_at - self.created_at).total_seconds())

    @property
    def current_task(self) -> TaskState | None:
        """Return the newest running task, or the newest observed task."""
        tasks = sorted(
            self.tasks.values(),
            key=lambda task: task.updated_at or self.created_at,
        )
        running = [task for task in tasks if task.status == "running"]
        return (running or tasks)[-1] if tasks else None

    @property
    def current_coordinates(self) -> dict[str, int | float | str]:
        """Return coordinates from the current task without assigning semantics."""
        task = self.current_task
        return dict(task.coordinates) if task is not None else {}


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """All valid execution snapshots for one logical run."""

    layout: RunLayout
    executions: tuple[MonitorSnapshot, ...]
    selected_execution_id: str
    warnings: tuple[str, ...] = ()

    @property
    def selected(self) -> MonitorSnapshot:
        """Return the selected execution snapshot."""
        for snapshot in self.executions:
            if snapshot.execution_id == self.selected_execution_id:
                return snapshot
        raise RuntimeError(f"Selected execution {self.selected_execution_id!r} disappeared")

    @property
    def metric_history(self) -> dict[str, tuple[MetricPoint, ...]]:
        """Return selected-lineage metric histories in chronological order."""
        selected = self.selected
        lineage_ids = {selected.execution_id, *selected.lineage}
        combined: dict[str, list[MetricPoint]] = {}
        for snapshot in self.executions:
            if snapshot.execution_id not in lineage_ids:
                continue
            for name, points in snapshot.metric_history.items():
                combined.setdefault(name, []).extend(points)
        return {
            name: tuple(sorted(points, key=lambda point: point.time))
            for name, points in combined.items()
        }


def discover_executions(layout: RunLayout) -> tuple[list[ExecutionContext], list[str]]:
    """Return valid attempts sorted by creation time plus isolated warnings."""
    contexts: list[ExecutionContext] = []
    warnings: list[str] = []
    if not layout.executions_dir.is_dir():
        return contexts, warnings
    for child in sorted(layout.executions_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        try:
            context = join_execution_context(
                layout.run_dir,
                child.name,
                expected_run_name=layout.run_name,
            )
        except (OSError, ValueError) as error:
            warnings.append(f"Ignored execution {child.name!r}: {error}")
            continue
        contexts.append(context)
    contexts.sort(
        key=lambda item: (parse_time(item.metadata.created_at), item.metadata.execution_id)
    )
    return contexts, warnings


def select_execution(layout: RunLayout, execution_id: str | None = None) -> ExecutionContext:
    """Select an exact attempt or the newest valid attempt without guessing names."""
    if execution_id is not None:
        return join_execution_context(
            layout.run_dir,
            execution_id,
            expected_run_name=layout.run_name,
        )
    contexts, _warnings = discover_executions(layout)
    if not contexts:
        raise FileNotFoundError(f"No valid executions found for {layout.run_dir}")
    return contexts[-1]


def execution_lineage(layout: RunLayout, metadata: ExecutionMetadata) -> tuple[str, ...]:
    """Follow explicit previous and parent links while rejecting cycles silently."""
    contexts, _warnings = discover_executions(layout)
    records = {context.metadata.execution_id: context.metadata for context in contexts}
    lineage: list[str] = []
    seen = {metadata.execution_id}
    pending = [metadata.previous_execution_id, metadata.parent_execution_id]
    while pending:
        execution_id = pending.pop(0)
        if execution_id is None or execution_id in seen:
            continue
        seen.add(execution_id)
        lineage.append(execution_id)
        parent = records.get(execution_id)
        if parent is not None:
            pending.extend((parent.previous_execution_id, parent.parent_execution_id))
    return tuple(lineage)


def fold_events(
    context: ExecutionContext,
    events: Iterable[ExecutionEvent],
    *,
    lineage: tuple[str, ...] = (),
    warnings: Iterable[str] = (),
) -> MonitorSnapshot:
    """Fold an arbitrary producer-event collection into deterministic state."""
    snapshot = MonitorSnapshot(context=context, lineage=lineage, warnings=list(warnings))
    ordered = sorted(
        events,
        key=lambda event: (
            parse_time(event.time),
            event.source,
            -1 if event.rank is None else event.rank,
            event.sequence,
        ),
    )
    for event in ordered:
        apply_event(snapshot, event)
    return snapshot


def apply_event(snapshot: MonitorSnapshot, event: ExecutionEvent) -> None:
    """Apply one validated observation to an existing snapshot."""
    observed_at = parse_time(event.time)
    producer_key = ProducerKey(event.source, event.rank)
    producer = snapshot.producers.setdefault(producer_key, ProducerState(producer_key))
    producer.sequence = event.sequence
    producer.updated_at = observed_at
    if event.phase is not None:
        producer.phase = event.phase

    if event.event == "execution_started":
        snapshot.status = "running"
        producer.status = "running"
    elif event.event in _TERMINAL_RUN_STATUS:
        snapshot.status = _TERMINAL_RUN_STATUS[event.event]
    elif event.event == "process_started":
        producer.status = "running"
    elif event.event == "process_completed":
        producer.status = "completed" if event.exit_code in {None, 0} else "failed"
        producer.exit_code = event.exit_code
        producer.signal = event.signal

    if event.phase is not None:
        if event.event == "phase_started":
            snapshot.phases[event.phase] = "running"
        elif event.event in _TERMINAL_SCOPE_STATUS and event.event.startswith("phase_"):
            snapshot.phases[event.phase] = _TERMINAL_SCOPE_STATUS[event.event]

    if event.task_id is not None:
        task_key = (producer_key, event.task_id)
        task = snapshot.tasks.setdefault(
            task_key,
            TaskState(
                producer=producer_key,
                phase=event.phase or "unknown",
                task_id=event.task_id,
                parent_task_id=event.parent_task_id,
            ),
        )
        task.updated_at = observed_at
        if task.started_at is None:
            task.started_at = observed_at
        if event.event == "task_started":
            task.status = "running"
        elif event.event in _TERMINAL_SCOPE_STATUS and event.event.startswith("task_"):
            task.status = _TERMINAL_SCOPE_STATUS[event.event]
        elif event.event == "progress":
            task.status = (
                "completed" if event.final and event.total == event.completed else "running"
            )
            task.completed = event.completed or 0
            task.total = event.total
            task.unit = event.unit
        if event.throughput is not None:
            task.throughput = event.throughput
        if event.message is not None:
            task.message = event.message
        task.coordinates.update(event.coordinates)
        task.display_metrics.update(event.display_metrics)

    if event.display_metrics:
        coordinates = combined_coordinates(event)
        for name, value in event.display_metrics.items():
            snapshot.metric_history.setdefault(name, []).append(
                MetricPoint(observed_at, value, producer_key, coordinates)
            )


def combined_coordinates(event: ExecutionEvent) -> dict[str, int | float | str]:
    """Return new coordinates plus compatible scalar legacy extension fields."""
    coordinates = dict(event.coordinates)
    for name in ("epoch", "batch", "global_step", "optimizer_step", "slide_id"):
        value = event.extensions.get(name)
        if isinstance(value, int | float | str) and not isinstance(value, bool):
            coordinates.setdefault(name, value)
    return coordinates


class ExecutionMonitor:
    """Incrementally tail all recognized streams for one immutable attempt."""

    def __init__(self, layout: RunLayout, execution_id: str | None = None) -> None:
        self.layout = layout
        self.context = select_execution(layout, execution_id)
        self._readers: dict[Path, ExecutionEventTailReader] = {}
        self._events: list[ExecutionEvent] = []
        self._warnings: list[str] = []
        self._failed_paths: set[Path] = set()

    def poll(self) -> MonitorSnapshot:
        """Read newly appended records and return a fresh deterministic fold."""
        for path in event_stream_paths(self.context):
            if path in self._failed_paths:
                continue
            reader = self._readers.setdefault(path, ExecutionEventTailReader(path))
            try:
                self._events.extend(reader.poll())
            except ExecutionEventReadError as error:
                self._events.extend(error.valid_events)
                self._warnings.append(str(error))
                self._failed_paths.add(path)
            except OSError as error:
                self._warnings.append(str(error))
                self._failed_paths.add(path)
        return fold_events(
            self.context,
            self._events,
            lineage=execution_lineage(self.layout, self.context.metadata),
            warnings=self._warnings,
        )


class RunMonitor:
    """Incrementally monitor every valid execution for one logical run."""

    def __init__(self, layout: RunLayout, execution_id: str | None = None) -> None:
        self.layout = layout
        self.execution_id = execution_id
        if execution_id is not None:
            select_execution(layout, execution_id)
        self._monitors: dict[str, ExecutionMonitor] = {}

    def poll(self, selected_execution_id: str | None = None) -> RunSnapshot:
        """Discover attempts, tail each stream, and select one execution."""
        contexts, warnings = discover_executions(self.layout)
        if not contexts:
            raise FileNotFoundError(f"No valid executions found for {self.layout.run_dir}")
        execution_ids = {context.metadata.execution_id for context in contexts}
        for execution_id in execution_ids:
            self._monitors.setdefault(
                execution_id,
                ExecutionMonitor(self.layout, execution_id),
            )
        snapshots = tuple(
            self._monitors[context.metadata.execution_id].poll() for context in contexts
        )
        if self.execution_id is not None and self.execution_id not in execution_ids:
            raise FileNotFoundError(
                f"Execution {self.execution_id!r} is no longer available for "
                f"{self.layout.run_dir}"
            )
        requested = self.execution_id or selected_execution_id
        selected = requested if requested in execution_ids else snapshots[-1].execution_id
        return RunSnapshot(
            layout=self.layout,
            executions=snapshots,
            selected_execution_id=selected,
            warnings=tuple(warnings),
        )


def event_stream_paths(context: ExecutionContext) -> list[Path]:
    """Discover only reserved runner and rank JSONL stream filenames."""
    paths: list[Path] = []
    runner = context.execution_dir / "runner.jsonl"
    if runner.is_file():
        paths.append(runner)
    rank_paths: list[tuple[int, Path]] = []
    for path in context.execution_dir.glob("rank-*.jsonl"):
        matched = _RANK_STREAM_PATTERN.fullmatch(path.name)
        if matched is not None and path.is_file():
            rank_paths.append((int(matched.group("rank")), path))
    paths.extend(path for _rank, path in sorted(rank_paths))
    return paths


def parse_time(value: str) -> datetime:
    """Parse a validated schema-v1 UTC timestamp."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
