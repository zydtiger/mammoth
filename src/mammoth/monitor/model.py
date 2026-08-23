"""Passive discovery and state folding for Mammoth execution artifacts.

The CLI and optional interactive UI consume this module. It reads only core
metadata and JSONL events, so monitoring never imports a consuming project.
"""

from __future__ import annotations

import bisect
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
ScopeStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "interrupted",
    "skipped",
    "stale",
]

_RANK_STREAM_PATTERN = re.compile(r"^rank-(?P<rank>[0-9]+)\.jsonl$")
FLEET_TAIL_WINDOW_BYTES = 128 * 1024
"""Bytes from the end of each stream that fleet/group roll-up folding reads.

Passed as ``ExecutionMonitor(..., tail_window_bytes=FLEET_TAIL_WINDOW_BYTES)``
by :mod:`mammoth.monitor.fleet`. See ``docs/MONITOR.md`` for the rationale
and the approximation this bound introduces; single-run folding never uses
this constant, so its exact full-history semantics are unaffected.
"""
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
    last_event: str | None = None
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
    events: tuple[ExecutionEvent, ...] = ()
    _phase_keys: dict[str, tuple[datetime, str, int, int]] = field(
        default_factory=dict, compare=False, repr=False
    )
    _metric_keys: dict[str, list[tuple[datetime, str, int, int]]] = field(
        default_factory=dict, compare=False, repr=False
    )

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
    def terminal_event_time(self) -> datetime | None:
        """Return when this execution reached a terminal outcome, if it has.

        Returns ``None`` unless ``status`` itself is terminal (``completed``,
        ``failed``, or ``interrupted``). A multi-rank execution can have one
        rank's own ``process_completed`` event long before every rank
        finishes, while ``status`` correctly stays ``running`` until they
        all have (see ``_finalize_run_status``); a folding source that used
        one rank's local terminal timestamp regardless of overall status
        would treat a still-running execution as finished. Prefers a
        runner-emitted terminal event; without one, falls back to an
        interrupted, then failed, then any process-terminal event. Mirrors
        the terminal-event selection dashboard rendering uses, exposed here
        as a timestamp for folding sources (such as fleet recency ordering)
        that need "when did this finish" without the full event object.
        """
        if self.status not in {"completed", "failed", "interrupted"}:
            return None
        runner_terminal = next(
            (event for event in reversed(self.events) if event.event in _TERMINAL_RUN_STATUS),
            None,
        )
        if runner_terminal is not None:
            return parse_time(runner_terminal.time)
        process_terminals = [event for event in self.events if event.event == "process_completed"]
        interrupted = [event for event in process_terminals if event.signal is not None]
        failed = [event for event in process_terminals if event.exit_code not in {None, 0}]
        preferred = interrupted or failed or process_terminals
        return parse_time(preferred[-1].time) if preferred else None

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
    def resume_lineage(self) -> tuple[MonitorSnapshot, ...]:
        """Return the selected attempt and resolved resume parents chronologically."""
        selected = self.selected
        by_id = {snapshot.execution_id: snapshot for snapshot in self.executions}
        reversed_lineage: list[MonitorSnapshot] = []
        current: MonitorSnapshot | None = selected
        seen: set[str] = set()
        while current is not None and current.execution_id not in seen:
            reversed_lineage.append(current)
            seen.add(current.execution_id)
            parent_id = current.context.metadata.parent_execution_id
            current = by_id.get(parent_id) if parent_id is not None else None
        return tuple(reversed(reversed_lineage))

    @property
    def logical_coordinates(self) -> dict[str, int | float | str]:
        """Return latest coordinates across the selected explicit resume lineage."""
        coordinates: dict[str, int | float | str] = {}
        for snapshot in self.resume_lineage:
            tasks = sorted(
                snapshot.tasks.values(),
                key=lambda task: task.updated_at or snapshot.created_at,
            )
            for task in tasks:
                coordinates.update(task.coordinates)
        return coordinates

    @property
    def logical_source_execution_id(self) -> str | None:
        """Return the execution that supplied the newest logical coordinates."""
        source: str | None = None
        latest: datetime | None = None
        for snapshot in self.resume_lineage:
            for task in snapshot.tasks.values():
                if not task.coordinates or task.updated_at is None:
                    continue
                if latest is None or task.updated_at >= latest:
                    latest = task.updated_at
                    source = snapshot.execution_id
        return source

    @property
    def metric_history(self) -> dict[str, tuple[MetricPoint, ...]]:
        """Return resume-aware metric histories in chronological order."""
        combined: dict[str, list[MetricPoint]] = {}
        for snapshot in self.resume_lineage:
            metadata = snapshot.context.metadata
            if metadata.parent_execution_id is not None:
                for name, points in combined.items():
                    combined[name] = [
                        point
                        for point in points
                        if _metric_precedes_resume(
                            point,
                            starting_global_step=metadata.starting_global_step,
                            starting_epoch=metadata.starting_epoch,
                        )
                    ]
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


def _event_sort_key(event: ExecutionEvent) -> tuple[datetime, str, int, int]:
    """Return the deterministic cross-stream fold order for one event.

    Ties on time break on source, then rank (runner sorts before any rank),
    then sequence, matching one producer's own append order.
    """
    return (
        parse_time(event.time),
        event.source,
        -1 if event.rank is None else event.rank,
        event.sequence,
    )


def fold_events(
    context: ExecutionContext,
    events: Iterable[ExecutionEvent],
    *,
    lineage: tuple[str, ...] = (),
    warnings: Iterable[str] = (),
) -> MonitorSnapshot:
    """Fold an arbitrary producer-event collection into deterministic state."""
    ordered = sorted(events, key=_event_sort_key)
    snapshot = MonitorSnapshot(
        context=context,
        lineage=lineage,
        warnings=list(warnings),
        events=tuple(ordered),
    )
    for event in ordered:
        apply_event(snapshot, event)
    _finalize_run_status(
        snapshot,
        has_runner_terminal=any(event.event in _TERMINAL_RUN_STATUS for event in ordered),
    )
    return snapshot


def _apply_if_newer(
    watermarks: dict[str, tuple[datetime, str, int, int]],
    values: dict[str, ScopeStatus],
    name: str,
    value: ScopeStatus,
    key: tuple[datetime, str, int, int],
) -> None:
    """Overwrite ``values[name]`` only if ``key`` is newer than its watermark.

    Keeps a cross-producer "true event order, last writer wins" register
    order-independent of *application* order: incremental folding applies
    each poll's newly read events sorted only among themselves, so a value
    shared across producers (a phase name multiple ranks or the runner can
    all touch) can otherwise see a chronologically-earlier event applied
    after a chronologically-later one already applied from an earlier poll
    (clock skew or a stalled writer flush can make an earlier-timestamped
    event become readable only later). Full folding's already-globally-
    sorted input always advances this watermark forward, so this adds no
    observable behavior there, only a cheap comparison.
    """
    if name not in watermarks or key > watermarks[name]:
        watermarks[name] = key
        values[name] = value


def _insert_metric_point(
    history: dict[str, list[MetricPoint]],
    keys: dict[str, list[tuple[datetime, str, int, int]]],
    name: str,
    point: MetricPoint,
    key: tuple[datetime, str, int, int],
) -> None:
    """Insert ``point`` at its correct chronological position for ``name``.

    Appends in the common case (O(1)): points normally arrive already in
    order, including under full folding's pre-sorted input. Falls back to a
    positional insert only when a point turns out older than the newest one
    already recorded for this metric — the same cross-producer visibility-
    order issue :func:`_apply_if_newer` guards against, but for an ordered
    history list rather than one current value.
    """
    points = history.setdefault(name, [])
    point_keys = keys.setdefault(name, [])
    if not point_keys or key >= point_keys[-1]:
        points.append(point)
        point_keys.append(key)
        return
    position = bisect.bisect_right(point_keys, key)
    points.insert(position, point)
    point_keys.insert(position, key)


def apply_event(snapshot: MonitorSnapshot, event: ExecutionEvent) -> None:
    """Apply one validated observation to an existing snapshot."""
    observed_at = parse_time(event.time)
    sort_key = _event_sort_key(event)
    producer_key = ProducerKey(event.source, event.rank)
    producer = snapshot.producers.setdefault(producer_key, ProducerState(producer_key))
    producer.sequence = event.sequence
    producer.updated_at = observed_at
    producer.last_event = event.event
    if event.phase is not None:
        producer.phase = event.phase

    if event.event == "execution_started":
        snapshot.status = "running"
        producer.status = "running"
    elif event.event in _TERMINAL_RUN_STATUS:
        snapshot.status = _TERMINAL_RUN_STATUS[event.event]
    elif event.event == "process_started":
        for task_key in [key for key in snapshot.tasks if key[0] == producer_key]:
            del snapshot.tasks[task_key]
        producer.status = "running"
    elif event.event == "process_completed":
        if event.signal is not None:
            producer.status = "interrupted"
        else:
            producer.status = "completed" if event.exit_code in {None, 0} else "failed"
        producer.exit_code = event.exit_code
        producer.signal = event.signal

    if snapshot.status == "pending" and event.event in {
        "process_started",
        "phase_started",
        "task_started",
        "progress",
        "heartbeat",
    }:
        snapshot.status = "running"

    if event.phase is not None:
        if event.event == "phase_started":
            _apply_if_newer(snapshot._phase_keys, snapshot.phases, event.phase, "running", sort_key)
        elif event.event in _TERMINAL_SCOPE_STATUS and event.event.startswith("phase_"):
            _apply_if_newer(
                snapshot._phase_keys,
                snapshot.phases,
                event.phase,
                _TERMINAL_SCOPE_STATUS[event.event],
                sort_key,
            )

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
        if event.throughput is not None:
            task.throughput = event.throughput
        if event.message is not None:
            task.message = event.message
        task.coordinates.update(combined_coordinates(event))
        task.display_metrics.update(event.display_metrics)

    if event.display_metrics:
        coordinates = combined_coordinates(event)
        for name, value in event.display_metrics.items():
            _insert_metric_point(
                snapshot.metric_history,
                snapshot._metric_keys,
                name,
                MetricPoint(observed_at, value, producer_key, coordinates),
                sort_key,
            )


def combined_coordinates(event: ExecutionEvent) -> dict[str, int | float | str]:
    """Return new coordinates plus compatible scalar legacy extension fields."""
    coordinates = dict(event.coordinates)
    for name in (
        "epoch",
        "epoch_total",
        "batch",
        "global_step",
        "optimizer_step",
        "optimizer_step_total",
        "optimizer_total",
        "slide_id",
    ):
        value = event.extensions.get(name)
        if isinstance(value, int | float | str) and not isinstance(value, bool):
            coordinates.setdefault(name, value)
    return coordinates


def _metric_precedes_resume(
    point: MetricPoint,
    *,
    starting_global_step: int | None,
    starting_epoch: int | None,
) -> bool:
    """Return whether a parent metric remains before a resumed child horizon."""
    global_step = point.coordinates.get("global_step")
    if starting_global_step is not None and isinstance(global_step, int):
        return global_step < starting_global_step
    epoch = point.coordinates.get("epoch")
    if starting_epoch is not None and isinstance(epoch, int):
        return epoch < starting_epoch
    return True


def _finalize_run_status(
    snapshot: MonitorSnapshot,
    *,
    has_runner_terminal: bool,
) -> None:
    """Infer direct-process terminal state when no runner terminal was emitted."""
    if has_runner_terminal:
        return
    processes = [
        producer for key, producer in snapshot.producers.items() if key.source == "process"
    ]
    if not processes:
        return
    if any(producer.status == "interrupted" for producer in processes):
        snapshot.status = "interrupted"
        return
    if any(producer.status == "failed" for producer in processes):
        snapshot.status = "failed"
        return
    expected = snapshot.context.metadata.world_size
    if len(processes) >= expected and all(producer.status == "completed" for producer in processes):
        snapshot.status = "completed"
    elif any(producer.status == "running" for producer in processes):
        snapshot.status = "running"


class ExecutionMonitor:
    """Incrementally tail all recognized streams for one immutable attempt.

    Maintains one persistent folded :class:`MonitorSnapshot` across polls,
    applying only newly read events (:func:`apply_event`) instead of
    re-folding the complete accumulated history, and never retains the
    complete raw event list: only terminal-class events
    (:attr:`~mammoth.core.events.ExecutionEvent.is_terminal`) are kept, in
    :attr:`MonitorSnapshot.events`, since that is the only subset any reader
    of that field consumes (:meth:`MonitorSnapshot.terminal_event_time` and
    the dashboard's phase-history and terminal-event rendering).

    Each poll's newly read events, across every stream, are sorted among
    themselves before being applied, then applied after everything already
    folded — but *application* order is not the same as *event-time* order
    across producers: a later poll can make a chronologically-earlier event
    visible only after a chronologically-later event from a different
    producer was already applied from an earlier poll (a stalled writer
    flush or cross-host clock skew, not merely a slow poller). Fields scoped
    to one producer or one producer's task are unaffected regardless — each
    producer's own stream is always read and applied in that producer's own
    strict order, matching :func:`fold_events` exactly no matter how polls
    are batched. Fields a phase name or a metric name can share *across*
    producers (:attr:`MonitorSnapshot.phases`, :attr:`MonitorSnapshot.
    metric_history`) apply each event only if it is newer, by true event
    order, than whatever last set that name (see ``_apply_if_newer`` /
    ``_insert_metric_point``), which is what keeps those exactly equivalent
    to a full re-fold regardless of poll batching — not merely appending or
    overwriting in the order events happened to become visible.

    ``tail_window_bytes``, when set, bounds only the *first* read of each
    underlying stream (see
    :class:`~mammoth.core.events.ExecutionEventTailReader`); this is what
    :mod:`mammoth.monitor.fleet` roll-up folding uses, never single-run
    folding, so single-run behavior and outputs are unchanged.
    """

    def __init__(
        self,
        layout: RunLayout,
        execution_id: str | None = None,
        *,
        tail_window_bytes: int | None = None,
    ) -> None:
        self.layout = layout
        self.context = select_execution(layout, execution_id)
        self._tail_window_bytes = tail_window_bytes
        self._readers: dict[Path, ExecutionEventTailReader] = {}
        self._warnings: list[str] = []
        self._failed_paths: set[Path] = set()
        self._snapshot = MonitorSnapshot(context=self.context)
        self._retained_events: list[ExecutionEvent] = []
        self._has_runner_terminal = False

    def poll(self) -> MonitorSnapshot:
        """Read newly appended records and apply them to the folded state."""
        new_events: list[ExecutionEvent] = []
        for path in event_stream_paths(self.context):
            if path in self._failed_paths:
                continue
            reader = self._readers.setdefault(
                path,
                ExecutionEventTailReader(path, tail_window_bytes=self._tail_window_bytes),
            )
            try:
                new_events.extend(reader.poll())
            except ExecutionEventReadError as error:
                new_events.extend(error.valid_events)
                self._warnings.append(str(error))
                self._failed_paths.add(path)
            except OSError as error:
                self._warnings.append(str(error))
                self._failed_paths.add(path)
        new_events.sort(key=_event_sort_key)
        for event in new_events:
            apply_event(self._snapshot, event)
            if event.is_terminal:
                bisect.insort(self._retained_events, event, key=_event_sort_key)
            if event.event in _TERMINAL_RUN_STATUS:
                self._has_runner_terminal = True
        _finalize_run_status(self._snapshot, has_runner_terminal=self._has_runner_terminal)
        self._snapshot.events = tuple(self._retained_events)
        self._snapshot.lineage = execution_lineage(self.layout, self.context.metadata)
        self._snapshot.warnings = list(self._warnings)
        return self._snapshot


class RunMonitor:
    """Incrementally monitor every valid execution for one logical run.

    ``lazy_first_poll``, when set, fully polls only the selected (or newest)
    execution on this monitor's *first* :meth:`poll` call; every other
    attempt gets a cheap unpolled placeholder (immutable metadata only, no
    event-stream read) for that one call. Every subsequent call polls every
    attempt exactly as before, so the attempt-history panel is complete
    again from the very next scheduled refresh. This exists only to let a
    fleet drill-down's first screen paint without waiting on every historical
    attempt's full-history read; standalone single-run monitoring never sets
    it, so its output is unaffected — see ``docs/MONITOR.md``.
    """

    def __init__(
        self,
        layout: RunLayout,
        execution_id: str | None = None,
        *,
        lazy_first_poll: bool = False,
    ) -> None:
        self.layout = layout
        self.execution_id = execution_id
        if execution_id is not None:
            select_execution(layout, execution_id)
        self._monitors: dict[str, ExecutionMonitor] = {}
        self._lazy_next_poll = lazy_first_poll

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
        if self.execution_id is not None and self.execution_id not in execution_ids:
            raise FileNotFoundError(
                f"Execution {self.execution_id!r} is no longer available for {self.layout.run_dir}"
            )
        requested = self.execution_id or selected_execution_id
        selected = requested if requested in execution_ids else contexts[-1].metadata.execution_id

        lazy = self._lazy_next_poll
        self._lazy_next_poll = False
        snapshots = tuple(
            self._monitors[context.metadata.execution_id].poll()
            if not lazy or context.metadata.execution_id == selected
            else MonitorSnapshot(context=context)
            for context in contexts
        )
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
