"""Web monitor with one frontend and two sources of entries.

``mammoth monitor web`` finds entries below configured roots and is read-only.
``mammoth monitor serve`` finds them from every run directory the queue service
ever recorded, and adds the Queue page.

The server folds the same execution artifacts as the terminal dashboard and adds
each execution's telemetry stream. It never writes artifacts or contacts
producers. One background sweep keeps bounded fleet roll-ups fresh; a full
:class:`~mammoth.monitor.model.RunMonitor` exists only for an opened run and is
released after it has been idle.

The only writes are queue cancel and priority changes, forwarded to the queue
service. Job submission is never exposed.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any, Union
from urllib.parse import parse_qs, urlsplit

from mammoth.core.identity import validate_run_name
from mammoth.core.layout import RunLayout
from mammoth.core.telemetry import TelemetryTailReader, telemetry_stream_path
from mammoth.monitor.documents import fleet_document, group_document, run_document
from mammoth.monitor.fleet import FleetMonitor, FleetSnapshot
from mammoth.monitor.model import RunMonitor, RunSnapshot, StagePoint
from mammoth.queue.api import QueueClient, ServiceUnavailableError
from mammoth.queue.model import (
    JobNotFoundError,
    JobStateError,
    JobValidationError,
    QueueError,
    default_socket_path,
)

logger = logging.getLogger(__name__)

MAX_ENTRY_DEPTH = 4
DISCOVERY_INTERVAL_SECONDS = 30.0
RUN_IDLE_SECONDS = 600.0
SUMMARY_SECONDS = 10.0
MAX_DOCUMENT_ROWS = 100
DEFAULT_MAX_POINTS = 1000
MAX_POINTS_LIMIT = 5000
MAX_WRITE_BYTES = 4096
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_WILDCARD_ADDRESS = "0.0.0.0"
_QUEUE_WRITE = re.compile(r"^/api/queue/([0-9a-f]{12})/(cancel|priority)$")
_STATIC_DIR = Path(__file__).with_name("web_static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/uPlot.iife.min.js": ("uPlot.iife.min.js", "text/javascript; charset=utf-8"),
    "/uPlot.min.css": ("uPlot.min.css", "text/css; charset=utf-8"),
}


def _is_run_dir(path: Path) -> bool:
    # os.path.isdir treats an unreadable path as absent instead of raising.
    return os.path.isdir(path / "logs" / "executions")


def discover_entries(roots: Sequence[Path], *, max_depth: int = MAX_ENTRY_DEPTH) -> dict[str, Path]:
    """Return ``{entry_id: path}`` for directories holding runs or group manifests.

    Run directories are never descended into, and hidden directories and
    symbolic links are skipped, so project artifacts stay out of the walk.
    """
    entries: dict[str, Path] = {}

    def walk(path: Path, entry_id: str, depth: int) -> None:
        try:
            children = sorted(
                (child for child in os.scandir(path) if child.is_dir(follow_symlinks=False)),
                key=lambda child: child.name,
            )
        except OSError:
            return
        runs = [child for child in children if _is_run_dir(Path(child.path))]
        if runs or os.path.isdir(path / ".mammoth" / "groups"):
            entries[entry_id] = path
        if depth >= max_depth:
            return
        run_paths = {child.path for child in runs}
        for child in children:
            if child.path not in run_paths and not child.name.startswith("."):
                walk(Path(child.path), f"{entry_id}/{child.name}", depth + 1)

    used: set[str] = set()
    for root in roots:
        path = Path(root).resolve()
        name, suffix = path.name or "root", 2
        while name in used:
            name, suffix = f"{path.name}-{suffix}", suffix + 1
        used.add(name)
        walk(path, name, 0)
    return entries


def _is_wildcard(host: str) -> bool:
    try:
        return socket.gethostbyname(host or _WILDCARD_ADDRESS) == _WILDCARD_ADDRESS
    except OSError:
        return False  # Binding reports an unresolvable host.


def queue_entries(run_dirs: Iterable[str]) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """Return entries holding queued runs that have started, and each entry's run names.

    Directories that do not exist, such as deleted runs, are skipped.
    """
    entries: dict[str, Path] = {}
    names: dict[str, set[str]] = {}
    for run_dir in run_dirs:
        path = Path(run_dir)
        if _is_run_dir(path):
            entry_id = queue_entry_id(path.parent)
            entries[entry_id] = path.parent
            names.setdefault(entry_id, set()).add(path.name)
    return entries, names


def queue_entry_id(path: Path) -> str:
    """Return a stable readable ID, unique because it derives from the full path."""
    try:
        return str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def downsample(
    times: Sequence[float],
    values: Sequence[Union[float, None]],
    xs: Union[Sequence[float], None],
    max_points: int,
) -> tuple[list[float], list[Union[float, None]], Union[list[float], None]]:
    """Keep each bucket's minimum and maximum so spikes survive downsampling."""
    count = len(times)
    if count <= max_points:
        return list(times), list(values), list(xs) if xs is not None else None
    buckets = max(1, max_points // 2)
    kept: list[int] = []
    for bucket in range(buckets):
        start, end = bucket * count // buckets, (bucket + 1) * count // buckets
        present = [index for index in range(start, end) if values[index] is not None]
        if not present:
            kept.append(start)
            continue
        low = min(present, key=lambda index: values[index])  # type: ignore[arg-type,return-value]
        high = max(present, key=lambda index: values[index])  # type: ignore[arg-type,return-value]
        kept.extend(sorted({low, high}))
    return (
        [times[index] for index in kept],
        [values[index] for index in kept],
        [xs[index] for index in kept] if xs is not None else None,
    )


@dataclass
class _Column:
    """One telemetry series stored as parallel time and value columns."""

    times: list[float] = field(default_factory=list)
    values: list[Union[float, None]] = field(default_factory=list)


class _ExecutionTelemetry:
    """Fold one execution's telemetry stream into named columns."""

    def __init__(self, execution_dir: Path) -> None:
        self.reader = TelemetryTailReader(telemetry_stream_path(execution_dir))
        self.columns: dict[str, _Column] = {}
        self.unavailable: dict[str, str] = {}

    def _append(self, name: str, observed_at: float, value: Any) -> None:
        column = self.columns.setdefault(name, _Column())
        number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        column.times.append(observed_at)
        column.values.append(float(number) if number is not None else None)

    def poll(self) -> None:
        for record in self.reader.poll():
            observed_at, kind = record.get("time"), record.get("kind")
            if not isinstance(observed_at, (int, float)):
                continue
            if kind == "gpu":
                prefix = f"telemetry/gpu/{record.get('gpu')}"
                self._append(f"{prefix}/utilization", observed_at, record.get("utilization"))
                self._append(f"{prefix}/memory_mib", observed_at, record.get("memory_mib"))
                self._append(f"{prefix}/power_watts", observed_at, record.get("power_watts"))
                self.unavailable.pop("gpu", None)
            elif kind == "cpu":
                self._append("telemetry/cpu/main_cores", observed_at, record.get("main_cores"))
                self._append("telemetry/cpu/worker_cores", observed_at, record.get("worker_cores"))
            elif kind == "status":
                self.unavailable[str(record.get("source"))] = str(record.get("detail"))


@dataclass(frozen=True)
class _Span:
    """One stretch of a lane in one stage; ``end`` is ``None`` while it is open."""

    start: float
    end: Union[float, None]
    stage: str
    lane: str
    producer: str


def _stage_spans(snapshot: RunSnapshot, *, live: bool) -> list[_Span]:
    """Return every lane's spans across the lineage, oldest first.

    A stage ends at the next transition on its producer and lane within its own
    attempt, so a resume gap never counts as stage time. An attempt's last stage
    ends at that attempt's last observation, or stays open when ``live``.
    """
    spans: list[_Span] = []
    for attempt in snapshot.resume_lineage:
        last_seen = attempt.updated_at.timestamp()
        open_ended = live and attempt.execution_id == snapshot.selected_execution_id
        lanes: dict[tuple[str, str], list[StagePoint]] = {}
        for point in attempt.stage_history:
            lanes.setdefault((point.producer.label, point.lane), []).append(point)
        for (producer, lane), points in lanes.items():
            for index, point in enumerate(points):
                start = point.time.timestamp()
                end: Union[float, None]
                if index + 1 < len(points):
                    end = points[index + 1].time.timestamp()
                else:
                    end = None if open_ended else max(start, last_seen)
                spans.append(_Span(start, end, point.stage, lane, producer))
    spans.sort(key=lambda span: span.start)
    return spans


def _compact_spans(spans: Sequence[_Span], max_points: int, now: float) -> list[_Span]:
    """Bound each lane's span count while keeping every stage's share of the time.

    Each time bucket shows one stage, chosen by the time each stage is still owed,
    so a repeating pattern shows in proportion instead of aliasing to one stage.
    """
    lanes: dict[tuple[str, str], list[_Span]] = {}
    for span in spans:
        lanes.setdefault((span.producer, span.lane), []).append(span)
    budget = max(1, max_points // max(1, len(lanes)))
    compacted: list[_Span] = []
    for (producer, lane), items in lanes.items():
        if len(items) <= budget:
            compacted.extend(items)
            continue
        origin = items[0].start
        stop = max(span.end if span.end is not None else now for span in items)
        width = max(stop - origin, 1e-9) / budget
        totals: list[dict[str, float]] = [{} for _ in range(budget)]
        for span in items:
            end = span.end if span.end is not None else now
            first = min(budget - 1, int((span.start - origin) / width))
            last = min(budget - 1, int((end - origin) / width))
            for bucket in range(first, last + 1):
                low = origin + bucket * width
                overlap = min(end, low + width) - max(span.start, low)
                if overlap > 0:
                    totals[bucket][span.stage] = totals[bucket].get(span.stage, 0.0) + overlap
        merged: list[_Span] = []
        owed: dict[str, float] = {}
        for bucket, stages in enumerate(totals):
            if not stages:
                continue
            for name, overlap in stages.items():
                owed[name] = owed.get(name, 0.0) + overlap
            stage = max(owed, key=lambda name: owed[name])
            owed[stage] -= sum(stages.values())
            low = origin + bucket * width
            high = low + width
            if merged and merged[-1].stage == stage and merged[-1].end == low:
                merged[-1] = _Span(merged[-1].start, high, stage, lane, producer)
            else:
                merged.append(_Span(low, high, stage, lane, producer))
        if items[-1].end is None and merged:
            last_span = merged[-1]
            merged[-1] = _Span(last_span.start, None, last_span.stage, lane, producer)
        compacted.extend(merged)
    compacted.sort(key=lambda span: span.start)
    return compacted


def _stage_summary(
    spans: Sequence[_Span],
    columns: dict[str, list[_Column]],
    now: float,
) -> list[dict[str, Any]]:
    """Return per-lane time share, duration, and resource use for each stage."""
    gpu = sorted(
        (when, value)
        for name, parts in columns.items()
        if name.startswith("telemetry/gpu/") and name.endswith("/utilization")
        for part in parts
        for when, value in zip(part.times, part.values)
        if value is not None
    )
    resource_samples: dict[str, list[tuple[float, float]]] = {"gpu": gpu}
    for label, name in (("main", "main_cores"), ("workers", "worker_cores")):
        resource_samples[label] = sorted(
            (when, value)
            for part in columns.get(f"telemetry/cpu/{name}", [])
            for when, value in zip(part.times, part.values)
            if value is not None
        )
    lanes: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    for span in spans:
        end = span.end if span.end is not None else max(span.start, now)
        lanes.setdefault((span.producer, span.lane), {}).setdefault(span.stage, []).append(
            (span.start, end)
        )
    rows: list[dict[str, Any]] = []
    for (producer, lane), stages in lanes.items():
        total = sum(end - start for runs in stages.values() for start, end in runs) or 1.0
        for stage, runs in stages.items():
            within = {
                label: _values_within(samples, runs) for label, samples in resource_samples.items()
            }
            elapsed = sum(end - start for start, end in runs)
            rows.append(
                {
                    "producer": producer,
                    "lane": lane,
                    "stage": stage,
                    "count": len(runs),
                    "share": elapsed / total,
                    "mean_seconds": elapsed / len(runs),
                    "gpu_mean": _mean(within["gpu"]),
                    "gpu_median": median(within["gpu"]) if within["gpu"] else None,
                    "main_cores": _mean(within["main"]),
                    "worker_cores": _mean(within["workers"]),
                }
            )
    return sorted(
        rows, key=lambda row: (row["producer"], row["lane"] != "main", row["lane"], -row["share"])
    )


def _values_within(
    samples: Sequence[tuple[float, float]], runs: Iterable[tuple[float, float]]
) -> list[float]:
    """Return sorted-sample values falling inside any of the given spans."""
    times = [sample[0] for sample in samples]
    values: list[float] = []
    for start, end in runs:
        low = bisect.bisect_left(times, start)
        high = bisect.bisect_left(times, end)
        values.extend(sample[1] for sample in samples[low:high])
    return values


def _mean(values: Sequence[float]) -> Union[float, None]:
    return sum(values) / len(values) if values else None


class MonitorUnavailableError(Exception):
    """A run exists but its artifacts could not be read right now."""


class _RunView:
    """Own one opened run's full monitor and telemetry readers."""

    def __init__(self, layout: RunLayout, *, refresh_seconds: float) -> None:
        self.layout = layout
        self.lock = threading.Lock()
        self.monitor = RunMonitor(layout)
        self.telemetry: dict[str, _ExecutionTelemetry] = {}
        self.refresh_seconds = refresh_seconds
        self.last_access = time.monotonic()
        self._polled_at: Union[float, None] = None
        self._selected: Union[str, None] = None
        self._snapshot: Union[RunSnapshot, None] = None
        self._spans: Union[list[_Span], None] = None
        self._spans_key: Union[tuple[Union[float, None], str, bool], None] = None
        self._summary: Union[list[dict[str, Any]], None] = None
        self._summary_key: Union[tuple[str, bool], None] = None
        self._summary_at = 0.0

    def snapshot(self, execution: Union[str, None]) -> Union[RunSnapshot, None]:
        """Return a snapshot at most one refresh old; call with ``lock`` held.

        ``None`` means the run has no valid execution. A transient read failure
        raises, so a viewer keeps the page and retries instead of seeing "not found".
        """
        try:
            snapshot = self._snapshot_or_raise(execution)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise MonitorUnavailableError(str(error)) from error
        if execution is not None and execution != snapshot.selected_execution_id:
            return None
        return snapshot

    def stage_spans(self, snapshot: RunSnapshot, *, live: bool) -> list[_Span]:
        """Return spans once per poll, so viewers of one run share the work."""
        key = (self._polled_at, snapshot.selected_execution_id, live)
        if self._spans is None or self._spans_key != key:
            self._spans, self._spans_key = _stage_spans(snapshot, live=live), key
        return self._spans

    def stage_summary(self, snapshot: RunSnapshot, *, live: bool) -> list[dict[str, Any]]:
        """Return the stage summary, recomputed at most every ``SUMMARY_SECONDS``.

        Its cost grows with run length, and it changes slowly on a long run.
        """
        now = time.monotonic()
        key = (snapshot.selected_execution_id, live)
        if (
            self._summary is None
            or self._summary_key != key
            or now - self._summary_at >= SUMMARY_SECONDS
        ):
            rows = _stage_summary(
                self.stage_spans(snapshot, live=live),
                self.lineage_columns(snapshot),
                time.time(),
            )
            self._summary, self._summary_key, self._summary_at = rows, key, now
        return self._summary

    def _snapshot_or_raise(self, execution: Union[str, None]) -> RunSnapshot:
        now = time.monotonic()
        self.last_access = now
        if (
            self._snapshot is None
            or self._polled_at is None
            or now - self._polled_at >= self.refresh_seconds
            or execution != self._selected
        ):
            self._snapshot = self.monitor.poll(execution)
            self._polled_at, self._selected = now, execution
            for attempt in self._snapshot.resume_lineage:
                reader = self.telemetry.setdefault(
                    attempt.execution_id, _ExecutionTelemetry(attempt.context.execution_dir)
                )
                reader.poll()
        return self._snapshot

    def lineage_columns(self, snapshot: RunSnapshot) -> dict[str, list[_Column]]:
        """Return every telemetry column of the selected lineage, oldest attempt first."""
        columns: dict[str, list[_Column]] = {}
        for attempt in snapshot.resume_lineage:
            telemetry = self.telemetry.get(attempt.execution_id)
            if telemetry is not None:
                for name, column in telemetry.columns.items():
                    columns.setdefault(name, []).append(column)
        return columns


class MonitorService:
    """Discover entries, keep fleet roll-ups fresh, and answer web queries."""

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
        queue_client: Union[QueueClient, None] = None,
    ) -> None:
        self.roots = tuple(roots)
        self.queue_client = queue_client
        self._queue_runs: dict[str, set[str]] = {}
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self._lock = threading.Lock()
        self._entries: dict[str, Path] = {}
        self._fleets: dict[str, FleetMonitor] = {}
        self._snapshots: dict[str, FleetSnapshot] = {}
        self._views: dict[tuple[str, str], _RunView] = {}
        self._discovered_at: Union[float, None] = None
        self._swept_once = False
        self._sweep_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sweep_forever, name="mammoth-web-sweep")
        self._thread.daemon = True

    def start(self) -> None:
        """Start the background fleet sweep."""
        self._thread.start()

    def queue(self) -> Union[dict[str, Any], None]:
        """Return queue state with run links, or say why the service cannot answer."""
        if self.queue_client is None:
            return None
        try:
            jobs = self.queue_client.list_jobs()
            resources = self.queue_client.resources()
        except QueueError as error:
            return {"available": False, "error": str(error), "jobs": [], "resources": None}
        with self._lock:
            queue_runs = dict(self._queue_runs)
        for job in jobs:
            job["link"] = None
            if job.get("run_dir"):
                path = Path(job["run_dir"])
                entry = queue_entry_id(path.parent)
                if path.name in queue_runs.get(entry, ()):
                    job["link"] = {"entry": entry, "run": path.name}
        return {"available": True, "error": None, "jobs": jobs, "resources": resources}

    def _discover(self) -> bool:
        """Refresh entries and return whether the source answered."""
        if self.queue_client is None:
            entries = discover_entries(self.roots)
            with self._lock:
                self._entries = entries
            return True
        try:
            run_dirs = self.queue_client.run_dirs()
        except QueueError as error:
            # Keep what is already shown; runs stay readable while the service is down.
            logger.warning("Could not read run directories from the queue: %s", error)
            return False
        entries, names = queue_entries(run_dirs)
        with self._lock:
            self._entries = entries
            self._queue_runs = names
        return True

    def stop(self) -> None:
        """Stop the background fleet sweep."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def sweep(self) -> None:
        """Refresh discovery when due, then poll every entry's bounded fleet monitor."""
        with self._sweep_lock:
            self._sweep()

    def _sweep(self) -> None:
        now = time.monotonic()
        due = self._discovered_at is None or now - self._discovered_at >= DISCOVERY_INTERVAL_SECONDS
        # A failure, such as the queue still starting, retries on the next sweep.
        if due and self._discover():
            self._discovered_at = now
        with self._lock:
            entries = dict(self._entries)
            queue_runs = dict(self._queue_runs)
            for removed in set(self._snapshots) - set(entries):
                del self._snapshots[removed]
        for removed in set(self._fleets) - set(entries):
            del self._fleets[removed]
        for entry_id, path in entries.items():
            if self._stop.is_set():
                return
            fleet = self._fleets.setdefault(entry_id, FleetMonitor(path))
            if self.queue_client is not None:
                fleet.run_names = queue_runs.get(entry_id, set())
            try:
                snapshot = fleet.poll(stale_after_seconds=self.stale_after_seconds)
            except Exception as error:
                logger.warning("Could not poll entry %s: %s", path, error)
                continue
            with self._lock:
                self._snapshots[entry_id] = snapshot
        self._release_idle_views()
        with self._lock:
            self._swept_once = True

    def _sweep_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception:
                logger.exception("Fleet sweep failed; retrying")
            self._stop.wait(self.interval_seconds)

    def _release_idle_views(self) -> None:
        now = time.monotonic()
        with self._lock:
            for key, view in list(self._views.items()):
                if now - view.last_access >= RUN_IDLE_SECONDS:
                    del self._views[key]

    def fleet(self) -> dict[str, Any]:
        """Return every entry's groups and runs that have recorded executions."""
        with self._lock:
            snapshots = dict(self._snapshots)
            paths = dict(self._entries)
            ready = self._swept_once
        documents: list[dict[str, Any]] = []
        for entry_id, snapshot in snapshots.items():
            if entry_id not in paths:
                continue
            document = fleet_document(snapshot)
            document["loose_runs"] = [
                run
                for run in document["loose_runs"]
                if _is_run_dir(paths[entry_id] / run["run_name"])
            ]
            documents.append({"id": entry_id, **document})
        return {"entries": documents, "ready": ready, "time": time.time()}

    def group(self, entry_id: str, group_id: str) -> Union[dict[str, Any], None]:
        """Return one group's document, or ``None`` when it is unknown."""
        with self._lock:
            snapshot = self._snapshots.get(entry_id)
        if snapshot is None:
            return None
        for group in snapshot.groups:
            if group.group_id == group_id:
                return {"entry": entry_id, **group_document(group)}
        return None

    def _view(self, entry_id: str, run_name: str) -> Union[_RunView, None]:
        with self._lock:
            path = self._entries.get(entry_id)
            view = self._views.get((entry_id, run_name))
            if view is not None and path is not None and _is_run_dir(path / run_name):
                return view
            self._views.pop((entry_id, run_name), None)
            if path is None:
                return None
            try:
                layout = RunLayout(path, validate_run_name(run_name))
            except ValueError:
                return None
            if not _is_run_dir(layout.run_dir):
                return None
            view = _RunView(layout, refresh_seconds=self.interval_seconds / 2)
            self._views[(entry_id, run_name)] = view
            return view

    def _is_live(self, snapshot: RunSnapshot) -> bool:
        """Return whether the selected attempt is running and still reporting."""
        selected = snapshot.selected
        return (
            selected.status in {"pending", "running"}
            and time.time() - selected.updated_at.timestamp() <= self.stale_after_seconds
        )

    def run(
        self, entry_id: str, run_name: str, execution: Union[str, None]
    ) -> Union[dict[str, Any], None]:
        """Return one run's document with stage summary and telemetry state."""
        view = self._view(entry_id, run_name)
        if view is None:
            return None
        with view.lock:
            snapshot = view.snapshot(execution)
            if snapshot is None:
                return None
            document = run_document(snapshot, stale_after_seconds=self.stale_after_seconds)
            columns = view.lineage_columns(snapshot)
            summary = view.stage_summary(snapshot, live=self._is_live(snapshot))
            document["entry"] = entry_id
            document["stage_summary"] = summary[:MAX_DOCUMENT_ROWS]
            document["stages"] = document["stages"][:MAX_DOCUMENT_ROWS]
            document["display_items"] = sorted(
                document["display_items"], key=lambda item: item["time"], reverse=True
            )[:MAX_DOCUMENT_ROWS]
            document["telemetry_series"] = sorted(columns)
            telemetry = view.telemetry.get(snapshot.selected_execution_id)
            document["telemetry_unavailable"] = dict(telemetry.unavailable) if telemetry else {}
            document["warnings"].extend(
                warning
                for attempt in snapshot.resume_lineage
                if attempt.execution_id in view.telemetry
                for warning in view.telemetry[attempt.execution_id].reader.warnings
            )
            document["time"] = time.time()
            return document

    def series(
        self,
        entry_id: str,
        run_name: str,
        execution: Union[str, None],
        names: Sequence[str],
        *,
        since: float,
        max_points: int,
        after: Union[Sequence[float], None] = None,
    ) -> Union[dict[str, Any], None]:
        """Return requested points newer than ``since``, or newer than each ``after``.

        ``after`` aligns with ``names`` because series advance independently: one
        shared cursor would let a quiet series make every other series resend history.
        """
        view = self._view(entry_id, run_name)
        if view is None:
            return None
        with view.lock:
            snapshot = view.snapshot(execution)
            if snapshot is None:
                return None
            metrics = snapshot.metric_history
            columns = view.lineage_columns(snapshot)
            cursor = since
            series: dict[str, Any] = {}
            for position, name in enumerate(names):
                threshold = after[position] if after is not None else since
                if name in columns:
                    times: list[float] = []
                    values: list[Union[float, None]] = []
                    for part in columns[name]:
                        start = bisect.bisect_right(part.times, threshold)
                        times.extend(part.times[start:])
                        values.extend(part.values[start:])
                    xs: Union[list[float], None] = None
                    axis = "time"
                elif name in metrics:
                    points = metrics[name]
                    newest = len(points)
                    while newest > 0 and points[newest - 1].time.timestamp() > threshold:
                        newest -= 1
                    kept = points[newest:]
                    # With nothing to send, skip scanning a long history for the axis.
                    has_epoch = bool(kept) and all(
                        isinstance(point.coordinates.get("epoch"), (int, float))
                        and not isinstance(point.coordinates.get("epoch"), bool)
                        for point in points
                    )
                    times = [point.time.timestamp() for point in kept]
                    values = [point.value for point in kept]
                    xs = (
                        [float(point.coordinates["epoch"]) for point in kept] if has_epoch else None
                    )
                    axis = "epoch" if has_epoch else "time"
                else:
                    continue
                if times:
                    cursor = max(cursor, times[-1])
                sampled_times, sampled_values, sampled_xs = downsample(
                    times, values, xs, max_points
                )
                series[name] = {
                    "axis": axis,
                    "t": sampled_times,
                    "y": sampled_values,
                    "x": sampled_xs,
                }
            return {"cursor": cursor, "series": series}

    def stages(
        self,
        entry_id: str,
        run_name: str,
        execution: Union[str, None],
        *,
        since: float,
        max_points: int,
    ) -> Union[dict[str, Any], None]:
        """Return bounded stage spans overlapping the time after ``since``.

        A span that began before ``since`` and is still running at that moment is
        included, so a window opened inside a long stage still shows it.
        """
        view = self._view(entry_id, run_name)
        if view is None:
            return None
        with view.lock:
            snapshot = view.snapshot(execution)
            if snapshot is None:
                return None
            spans = view.stage_spans(snapshot, live=self._is_live(snapshot))
        kept = [span for span in spans if span.end is None or span.end > since]
        kept = _compact_spans(kept, max_points, time.time())
        return {
            "t": [span.start for span in kept],
            "end": [span.end for span in kept],
            "stage": [span.stage for span in kept],
            "lane": [span.lane for span in kept],
            "producer": [span.producer for span in kept],
        }


class _Handler(BaseHTTPRequestHandler):
    """Serve packaged static files and read-only JSON queries."""

    server: MonitorWebServer
    protocol_version = "HTTP/1.1"
    timeout = 60

    def do_POST(self) -> None:
        """Forward one queue cancel or priority change; nothing else is writable."""
        # Reading first keeps a reused connection's next request line intact.
        body = self._read_body()
        if body is None:
            self._json({"error": "invalid or oversized Content-Length"}, status=400)
            return
        client = self.server.service.queue_client
        if client is None:
            self._json({"error": "not found"}, status=404)
            return
        refusal = self._write_refusal()
        if refusal is not None:
            self._json({"error": refusal}, status=403)
            return
        match = _QUEUE_WRITE.match(urlsplit(self.path).path)
        if match is None:
            self._json({"error": "not found"}, status=404)
            return
        job_id, action = match.groups()
        try:
            if action == "cancel":
                self._json(client.cancel(job_id))
            else:
                self._json(client.set_priority(job_id, _write_document(body).get("priority")))
        except JobValidationError as error:
            self._json({"error": str(error)}, status=400)
        except JobNotFoundError as error:
            self._json({"error": str(error)}, status=404)
        except JobStateError as error:
            self._json({"error": str(error)}, status=409)
        except ServiceUnavailableError as error:
            self._json({"error": str(error)}, status=503)
        except QueueError as error:
            self._json({"error": str(error)}, status=500)

    def _write_refusal(self) -> Union[str, None]:
        """Allow writes only from this page, on the address the server is bound to."""
        if not self.server.writes_enabled:
            return "queue writes are disabled on a non-loopback bind without --allow-queue-writes"
        host = self.headers.get("Host")
        names = _LOOPBACK_NAMES if self.server.loopback_only else {self.server_host_name}
        # An exact address also stops DNS rebinding, where Origin and Host would agree.
        if host is None or urlsplit(f"//{host}").hostname not in names:
            return "unexpected Host header"
        origin = self.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != host:
            return "unexpected Origin header"
        # A cross-site form cannot send this type without a CORS preflight.
        if self.headers.get_content_type() != "application/json":
            return "unexpected Content-Type header"
        return None

    @property
    def server_host_name(self) -> str:
        """Return the bound address, which writes must name exactly."""
        return str(self.server.server_address[0])

    def _read_body(self) -> Union[bytes, None]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_WRITE_BYTES:
            self.close_connection = True
            return None
        return self.rfile.read(length)

    def do_GET(self) -> None:
        """Route one GET request; every route is read-only."""
        if not self._host_allowed():
            self._json({"error": "unexpected Host header"}, status=403)
            return
        url = urlsplit(self.path)
        if url.path in _STATIC_FILES:
            self._static(*_STATIC_FILES[url.path])
            return
        query = parse_qs(url.query)
        try:
            document = self._query(url.path, query)
        except ValueError as error:
            self._json({"error": str(error)}, status=400)
            return
        except MonitorUnavailableError as error:
            self._json({"error": str(error)}, status=503)
            return
        except Exception:
            logger.exception("Could not answer %s", self.path)
            self._json({"error": "internal error"}, status=500)
            return
        if document is None:
            self._json({"error": "not found"}, status=404)
            return
        self._json(document)

    def _host_allowed(self) -> bool:
        """Refuse a foreign Host on loopback, which blocks DNS rebinding from a browser."""
        host = self.headers.get("Host")
        if host is None or not self.server.loopback_only:
            return True
        return urlsplit(f"//{host}").hostname in _LOOPBACK_NAMES

    def _query(self, path: str, query: dict[str, list[str]]) -> Union[dict[str, Any], None]:
        service = self.server.service

        def single(name: str) -> Union[str, None]:
            return query[name][0] if name in query else None

        def required(name: str) -> str:
            value = single(name)
            if not value:
                raise ValueError(f"missing query parameter {name!r}")
            return value

        def finite(name: str, text: Union[str, float]) -> float:
            value = float(text)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            return value

        def number(name: str, default: float) -> float:
            return finite(name, single(name) or default)

        max_points = int(min(max(2.0, number("max_points", DEFAULT_MAX_POINTS)), MAX_POINTS_LIMIT))
        if path == "/api/fleet":
            return service.fleet()
        if path == "/api/mode":
            return {
                "queue": service.queue_client is not None,
                "writable": self.server.writes_enabled,
            }
        if path == "/api/queue":
            document = service.queue()
            if document is not None:
                document["writable"] = self.server.writes_enabled
            return document
        if path == "/api/group":
            return service.group(required("entry"), required("group"))
        if path == "/api/run":
            return service.run(required("entry"), required("run"), single("execution"))
        if path == "/api/series":
            names = query.get("name", [])
            after = [finite("after", item) for item in query["after"]] if "after" in query else None
            if after is not None and len(after) != len(names):
                raise ValueError("after must have one value per name")
            return service.series(
                required("entry"),
                required("run"),
                single("execution"),
                names,
                since=number("since", 0.0),
                max_points=max_points,
                after=after,
            )
        if path == "/api/stages":
            return service.stages(
                required("entry"),
                required("run"),
                single("execution"),
                since=number("since", 0.0),
                max_points=max_points,
            )
        return None

    def _static(self, filename: str, content_type: str) -> None:
        self._send((_STATIC_DIR / filename).read_bytes(), content_type)

    def _json(self, document: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(document, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json", status=status)

    def _send(self, body: bytes, content_type: str, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep routine requests out of the server's terminal output."""
        logger.debug(format, *args)


def _write_document(body: bytes) -> dict[str, Any]:
    try:
        document = json.loads(body or b"{}")
    except ValueError as error:
        raise JobValidationError(f"Request body is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise JobValidationError("Request body must be a JSON object")
    return document


class MonitorWebServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to one :class:`MonitorService`."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: MonitorService,
        *,
        allow_queue_writes: bool = False,
    ) -> None:
        super().__init__(address, _Handler)
        self.service = service
        self.loopback_only = address[0] in _LOOPBACK_NAMES
        self.writes_enabled = service.queue_client is not None and (
            self.loopback_only or allow_queue_writes
        )

    @property
    def server_name_for_url(self) -> str:
        """Return the bound host as text."""
        return str(self.server_address[0])

    @property
    def url(self) -> str:
        """Return the bound address as a URL."""
        return f"http://{self.server_name_for_url}:{self.server_address[1]}/"

    def shutdown_monitor(self) -> None:
        """Stop background work and release the listening socket."""
        self.service.stop()
        self.server_close()


def create_web_server(
    roots: Sequence[Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    interval_seconds: float = 2.0,
    stale_after_seconds: float = 90.0,
) -> MonitorWebServer:
    """Bind the read-only server and start its sweep; the caller runs ``serve_forever``."""
    service = MonitorService(
        roots,
        interval_seconds=interval_seconds,
        stale_after_seconds=stale_after_seconds,
    )
    server = MonitorWebServer((host, port), service)
    service.start()
    return server


def create_queue_monitor_server(
    queue_client: Union[QueueClient, None] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    interval_seconds: float = 2.0,
    stale_after_seconds: float = 90.0,
    allow_queue_writes: bool = False,
) -> MonitorWebServer:
    """Bind the queue-backed server and start its sweep; the caller runs ``serve_forever``."""
    if allow_queue_writes and _is_wildcard(host):
        raise ValueError("--allow-queue-writes needs one specific --host address")
    service = MonitorService(
        (),
        interval_seconds=interval_seconds,
        stale_after_seconds=stale_after_seconds,
        queue_client=queue_client or QueueClient(default_socket_path()),
    )
    server = MonitorWebServer((host, port), service, allow_queue_writes=allow_queue_writes)
    service.start()
    return server
