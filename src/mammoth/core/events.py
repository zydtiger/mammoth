"""Versioned framework-independent execution events for Mammoth run artifacts.

Process-local writers persist these records alongside immutable contexts from
:mod:`mammoth.core.execution`; later reader and dashboard layers replay them.
This module stays in ``mammoth.core`` so event infrastructure never depends on
PyTorch.
"""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, Self, cast

from mammoth.core.execution import (
    ExecutionContext,
    sanitize_metadata_fields,
)
from mammoth.core.identity import validate_execution_id

EXECUTION_EVENT_SCHEMA_VERSION = 1
MAX_DISPLAY_METRICS = 16
RUNNER_EVENT_STREAM_FILENAME = "runner.jsonl"
PROCESS_EVENT_STREAM_TEMPLATE = "rank-{rank}.jsonl"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
_APPEND_GUARD_BYTES = 4096

logger = logging.getLogger(__name__)

EventName = Literal[
    "execution_started",
    "execution_completed",
    "execution_failed",
    "execution_interrupted",
    "phase_started",
    "phase_completed",
    "phase_failed",
    "phase_skipped",
    "process_started",
    "process_completed",
    "task_started",
    "progress",
    "task_completed",
    "task_failed",
    "task_skipped",
    "heartbeat",
]
EventSource = Literal["runner", "process"]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ImmutableJsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["ImmutableJsonValue", ...]
    | Mapping[str, "ImmutableJsonValue"]
)

EVENT_NAMES: frozenset[str] = frozenset(
    {
        "execution_started",
        "execution_completed",
        "execution_failed",
        "execution_interrupted",
        "phase_started",
        "phase_completed",
        "phase_failed",
        "phase_skipped",
        "process_started",
        "process_completed",
        "task_started",
        "progress",
        "task_completed",
        "task_failed",
        "task_skipped",
        "heartbeat",
    }
)

_EXECUTION_EVENTS = frozenset(
    {
        "execution_started",
        "execution_completed",
        "execution_failed",
        "execution_interrupted",
    }
)
_PHASE_EVENTS = frozenset({"phase_started", "phase_completed", "phase_failed", "phase_skipped"})
_PROCESS_EVENTS = frozenset({"process_started", "process_completed"})
_TASK_EVENTS = frozenset(
    {"task_started", "progress", "task_completed", "task_failed", "task_skipped"}
)
_TERMINAL_EVENTS = frozenset(
    {
        "execution_completed",
        "execution_failed",
        "execution_interrupted",
        "phase_completed",
        "phase_failed",
        "phase_skipped",
        "process_completed",
        "task_completed",
        "task_failed",
        "task_skipped",
    }
)
_KNOWN_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "time",
        "execution_id",
        "run_name",
        "source",
        "rank",
        "world_size",
        "phase",
        "task_id",
        "parent_task_id",
        "event",
        "coordinates",
        "throughput",
        "duration_seconds",
        "exit_code",
        "signal",
        "completed",
        "total",
        "unit",
        "final",
        "message",
        "display_metrics",
    }
)
_WRITER_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "time",
        "execution_id",
        "run_name",
        "source",
        "rank",
        "world_size",
        "event",
    }
)


@dataclass(frozen=True)
class ExecutionEvent:
    """One validated schema-v1 record written by a process-local event writer."""

    schema_version: int
    sequence: int
    time: str
    execution_id: str
    run_name: str
    source: EventSource
    event: EventName
    rank: int | None = None
    world_size: int | None = None
    phase: str | None = None
    task_id: str | None = None
    parent_task_id: str | None = None
    coordinates: Mapping[str, int | float | str] = field(default_factory=dict)
    throughput: float | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    signal: int | str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    final: bool | None = None
    message: str | None = None
    display_metrics: Mapping[str, float] = field(default_factory=dict)
    extensions: Mapping[str, ImmutableJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce the v1 identity, applicability, and optional-data contract."""
        self._validate_identity()
        self._validate_applicability()
        self._validate_optional_data()
        if self.message is not None:
            sanitized_message = sanitize_metadata_fields({"message": self.message})["message"]
            object.__setattr__(self, "message", cast(str, sanitized_message))
        metrics = _validated_display_metrics(self.display_metrics)
        coordinates = _validated_coordinates(self.coordinates)
        extensions = _validated_extensions(self.extensions)
        object.__setattr__(self, "display_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))
        object.__setattr__(self, "extensions", MappingProxyType(extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the flattened JSON object consumed by writers and readers."""
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "time": self.time,
            "execution_id": self.execution_id,
            "run_name": self.run_name,
            "source": self.source,
            "event": self.event,
        }
        optional: dict[str, JsonValue] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "phase": self.phase,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "throughput": self.throughput,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "final": self.final,
            "message": self.message,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if self.display_metrics:
            payload["display_metrics"] = dict(self.display_metrics)
        if self.coordinates:
            payload["coordinates"] = dict(self.coordinates)
        payload.update({key: _thaw_json(value) for key, value in self.extensions.items()})
        return payload

    def to_json_line(self) -> bytes:
        """Serialize one complete UTF-8 JSONL record for process-local writers."""
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionEvent:
        """Validate a JSON object while preserving unknown optional v1 fields."""
        if not isinstance(payload, Mapping):
            raise ValueError("execution event must be a JSON object.")
        if any(not isinstance(key, str) or not key for key in payload):
            raise ValueError("execution event field names must be non-empty strings.")
        extensions = {key: value for key, value in payload.items() if key not in _KNOWN_FIELDS}
        return cls(
            schema_version=_required_int(payload, "schema_version", minimum=1),
            sequence=_required_int(payload, "sequence", minimum=1),
            time=_required_string(payload, "time"),
            execution_id=_required_string(payload, "execution_id"),
            run_name=_required_string(payload, "run_name"),
            source=cast(EventSource, _required_string(payload, "source")),
            event=cast(EventName, _required_string(payload, "event")),
            rank=_optional_int(payload, "rank", minimum=0),
            world_size=_optional_int(payload, "world_size", minimum=1),
            phase=_optional_string(payload, "phase"),
            task_id=_optional_string(payload, "task_id"),
            parent_task_id=_optional_string(payload, "parent_task_id"),
            coordinates=_optional_mapping(payload, "coordinates"),
            throughput=_optional_number(payload, "throughput", minimum=0.0),
            duration_seconds=_optional_number(payload, "duration_seconds", minimum=0.0),
            exit_code=_optional_int(payload, "exit_code"),
            signal=_optional_signal(payload),
            completed=_optional_int(payload, "completed", minimum=0),
            total=_optional_int(payload, "total", minimum=0),
            unit=_optional_string(payload, "unit"),
            final=_optional_bool(payload, "final"),
            message=_optional_string(payload, "message"),
            display_metrics=_optional_mapping(payload, "display_metrics"),
            extensions=extensions,
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether this event closes or skips a lifecycle scope."""
        return self.event in _TERMINAL_EVENTS

    def _validate_identity(self) -> None:
        _validate_int("schema_version", self.schema_version, minimum=1)
        if self.schema_version != EXECUTION_EVENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported execution event schema version: "
                f"{self.schema_version!r}; expected {EXECUTION_EVENT_SCHEMA_VERSION}."
            )
        _validate_int("sequence", self.sequence, minimum=1)
        _parse_utc_time(self.time)
        validate_execution_id(self.execution_id)
        _validate_nonempty_string("run_name", self.run_name)
        _validate_nonempty_string("source", self.source)
        if self.source not in {"runner", "process"}:
            raise ValueError(f"source must be 'runner' or 'process', got {self.source!r}.")
        _validate_nonempty_string("event", self.event)
        if self.event not in EVENT_NAMES:
            raise ValueError(f"Unknown execution event: {self.event!r}.")
        if self.source == "process":
            _validate_int("rank", self.rank, minimum=0)
            _validate_int("world_size", self.world_size, minimum=1)
            assert self.rank is not None and self.world_size is not None
            if self.rank >= self.world_size:
                raise ValueError(f"rank={self.rank} is outside event world_size={self.world_size}.")
        elif self.rank is not None or self.world_size is not None:
            raise ValueError("runner events must omit rank and world_size.")

    def _validate_applicability(self) -> None:
        if self.event in _EXECUTION_EVENTS:
            if self.phase is not None:
                raise ValueError(f"{self.event} must omit phase.")
        else:
            _validate_nonempty_string("phase", self.phase)
        if self.event in _TASK_EVENTS:
            _validate_nonempty_string("task_id", self.task_id)
        elif self.event == "heartbeat":
            if self.task_id is not None:
                _validate_nonempty_string("task_id", self.task_id)
        elif self.task_id is not None:
            raise ValueError(f"{self.event} must omit task_id.")
        if self.parent_task_id is not None:
            if self.task_id is None:
                raise ValueError("parent_task_id requires task_id.")
            _validate_nonempty_string("parent_task_id", self.parent_task_id)
            if self.parent_task_id == self.task_id:
                raise ValueError("parent_task_id must differ from task_id.")
        if self.event in _PROCESS_EVENTS and self.source != "process":
            raise ValueError(f"{self.event} requires source='process'.")
        progress_values = (self.completed, self.total, self.unit, self.final)
        if self.event == "progress":
            _validate_int("completed", self.completed, minimum=0)
            if self.total is not None:
                _validate_int("total", self.total, minimum=0)
            if self.final is not None and not isinstance(self.final, bool):
                raise ValueError(f"final must be a boolean, got {self.final!r}.")
            if (
                self.total is not None
                and self.completed is not None
                and self.completed > self.total
            ):
                raise ValueError("completed must not exceed total.")
        elif any(value is not None for value in progress_values):
            raise ValueError(f"{self.event} must omit progress-only fields.")

    def _validate_optional_data(self) -> None:
        for name in ("unit", "message"):
            text_value = cast(str | None, getattr(self, name))
            if text_value is not None:
                _validate_nonempty_string(name, text_value)
        for name in ("throughput", "duration_seconds"):
            numeric_value = cast(float | None, getattr(self, name))
            if numeric_value is not None:
                _validate_finite_number(name, numeric_value, minimum=0.0)
                object.__setattr__(self, name, float(numeric_value))
        if self.throughput is not None and self.event not in {
            "progress",
            "heartbeat",
            *_TERMINAL_EVENTS,
        }:
            raise ValueError(
                "throughput is valid only for progress, heartbeat, or terminal events."
            )
        if self.duration_seconds is not None and self.event not in _TERMINAL_EVENTS:
            raise ValueError("duration_seconds requires a terminal event.")
        if self.exit_code is not None:
            _validate_int("exit_code", self.exit_code)
            if self.event not in _TERMINAL_EVENTS:
                raise ValueError("exit_code requires a terminal event.")
        if self.signal is not None:
            if not isinstance(self.signal, int | str) or isinstance(self.signal, bool):
                raise ValueError("signal must be an integer or non-empty string.")
            if isinstance(self.signal, str):
                _validate_nonempty_string("signal", self.signal)
            if self.event not in {"execution_interrupted", "process_completed"}:
                raise ValueError(
                    "signal is valid only for interrupted executions or completed processes."
                )
        if self.display_metrics and self.event not in {"progress", "heartbeat"}:
            raise ValueError("display_metrics are valid only for progress or heartbeat events.")


class ExecutionEventWriter:
    """Own one append-only event stream without cross-process synchronization."""

    def __init__(
        self,
        path: Path,
        *,
        execution_id: str,
        run_name: str,
        source: EventSource,
        rank: int | None = None,
        world_size: int | None = None,
        progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
        utc_clock: Callable[[], str] | None = None,
        failure_logger: logging.Logger = logger,
    ) -> None:
        """Validate writer identity and open its reserved process-local stream."""
        self.path = Path(path)
        self.execution_id = validate_execution_id(execution_id)
        _validate_nonempty_string("run_name", run_name)
        self.run_name = run_name
        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.progress_interval_seconds = _validate_interval(
            "progress_interval_seconds", progress_interval_seconds
        )
        if self.progress_interval_seconds < DEFAULT_PROGRESS_INTERVAL_SECONDS:
            raise ValueError(
                "progress_interval_seconds must be at least "
                f"{DEFAULT_PROGRESS_INTERVAL_SECONDS}, got {progress_interval_seconds!r}."
            )
        self.heartbeat_interval_seconds = _validate_interval(
            "heartbeat_interval_seconds", heartbeat_interval_seconds
        )
        self._monotonic_clock = monotonic_clock
        self._utc_clock = utc_clock or utc_event_time
        self._failure_logger = failure_logger
        self._stream: BinaryIO | None = None
        self._sequence = 0
        self._pending_progress: ExecutionEvent | None = None
        self._last_progress_emitted: float | None = None
        self._last_activity = _monotonic_time(self._monotonic_clock)
        self._failure_logged = False
        self._closed = False
        self._lock = threading.RLock()
        self._validate_writer_identity()
        opened_stream: BinaryIO | None = None
        try:
            opened_stream = _open_event_stream(self.path)
            self._sequence = _existing_stream_sequence(
                opened_stream,
                self.path,
                execution_id=self.execution_id,
                run_name=self.run_name,
                source=self.source,
                rank=self.rank,
                world_size=self.world_size,
            )
            self._stream = opened_stream
        except Exception as error:
            if opened_stream is not None:
                with suppress(Exception):
                    opened_stream.close()
            self._disable_after_io_failure("open", error)

    @classmethod
    def for_process(
        cls,
        context: ExecutionContext,
        *,
        rank: int,
        world_size: int | None = None,
        **options: Any,
    ) -> Self:
        """Open the reserved ``rank-N.jsonl`` stream for one execution rank."""
        effective_world_size = context.metadata.world_size if world_size is None else world_size
        return cls(
            process_event_stream_path(context, rank, world_size=effective_world_size),
            execution_id=context.metadata.execution_id,
            run_name=context.metadata.run_name,
            source="process",
            rank=rank,
            world_size=effective_world_size,
            **options,
        )

    @classmethod
    def for_runner(
        cls,
        context: ExecutionContext,
        **options: Any,
    ) -> Self:
        """Open the reserved optional ``runner.jsonl`` orchestration stream."""
        return cls(
            runner_event_stream_path(context),
            execution_id=context.metadata.execution_id,
            run_name=context.metadata.run_name,
            source="runner",
            **options,
        )

    @property
    def enabled(self) -> bool:
        """Return whether this writer may still append records."""
        with self._lock:
            return self._stream is not None and not self._closed and not self._failure_logged

    @property
    def sequence(self) -> int:
        """Return the last sequence successfully flushed by this writer."""
        with self._lock:
            return self._sequence

    def emit(self, event: EventName, **fields: Any) -> ExecutionEvent | None:
        """Validate and immediately flush a lifecycle or heartbeat event."""
        with self._lock:
            if event == "progress":
                return self.emit_progress(**fields)
            if event == "heartbeat":
                return self.emit_heartbeat(**fields)
            if not self.enabled:
                return None
            observed_at = self._utc_clock()
            candidate = self._build_event(event, observed_at=observed_at, fields=fields)
            if (
                candidate.is_terminal
                and self._pending_progress is not None
                and _terminal_closes_progress(candidate, self._pending_progress)
            ):
                self.flush_progress(final=True)
                if not self.enabled:
                    return None
                candidate = _with_sequence(candidate, self._sequence + 1)
            return self._write_event(
                candidate,
                monotonic_now=_monotonic_time(self._monotonic_clock),
            )

    def emit_progress(
        self,
        *,
        phase: str,
        task_id: str,
        completed: int,
        total: int | None = None,
        final: bool = False,
        **fields: Any,
    ) -> ExecutionEvent | None:
        """Emit, replace, or immediately finalize one throttled progress update."""
        with self._lock:
            if not self.enabled:
                return None
            if not isinstance(final, bool):
                raise ValueError(f"final must be a boolean, got {final!r}.")
            payload = {
                "phase": phase,
                "task_id": task_id,
                "completed": completed,
                **fields,
            }
            if total is not None:
                payload["total"] = total
            if final:
                payload["final"] = True
            observed_at = self._utc_clock()
            candidate = self._build_event(
                "progress",
                observed_at=observed_at,
                fields=payload,
            )
            if (
                self._pending_progress is not None
                and not _same_progress_scope(self._pending_progress, candidate)
            ):
                self.flush_progress(final=False)
                if not self.enabled:
                    return None
            monotonic_now = _monotonic_time(self._monotonic_clock)
            self._last_activity = monotonic_now
            if (
                final
                or self._last_progress_emitted is None
                or monotonic_now - self._last_progress_emitted >= self.progress_interval_seconds
            ):
                self._pending_progress = None
                return self._write_event(candidate, monotonic_now=monotonic_now)
            self._pending_progress = candidate
            return None

    def flush_progress(self, *, final: bool = True) -> ExecutionEvent | None:
        """Immediately flush the latest replaceable progress observation, if any."""
        with self._lock:
            if not isinstance(final, bool):
                raise ValueError(f"final must be a boolean, got {final!r}.")
            pending = self._pending_progress
            if pending is None or not self.enabled:
                return None
            self._pending_progress = None
            candidate = _with_sequence(
                pending,
                self._sequence + 1,
                final=True if final else pending.final,
            )
            return self._write_event(
                candidate,
                monotonic_now=_monotonic_time(self._monotonic_clock),
            )

    def emit_heartbeat(
        self,
        *,
        phase: str,
        task_id: str | None = None,
        force: bool = False,
        **fields: Any,
    ) -> ExecutionEvent | None:
        """Emit a heartbeat only after the configured idle interval unless forced."""
        with self._lock:
            if not self.enabled:
                return None
            if not isinstance(force, bool):
                raise ValueError(f"force must be a boolean, got {force!r}.")
            monotonic_now = _monotonic_time(self._monotonic_clock)
            if not force and monotonic_now - self._last_activity < self.heartbeat_interval_seconds:
                return None
            if self._pending_progress is not None:
                self.flush_progress(final=False)
                if not force or not self.enabled:
                    return None
            payload = {"phase": phase, **fields}
            if task_id is not None:
                payload["task_id"] = task_id
            candidate = self._build_event(
                "heartbeat",
                observed_at=self._utc_clock(),
                fields=payload,
            )
            return self._write_event(candidate, monotonic_now=monotonic_now)

    @contextmanager
    def periodic_heartbeats(
        self,
        *,
        phase: str,
        task_id: str | None = None,
        message: str | None = None,
    ) -> Iterator[None]:
        """Emit idle heartbeats while the caller is blocked in long-running work."""
        stop = threading.Event()
        poll_seconds = min(1.0, self.heartbeat_interval_seconds)

        def emit_until_stopped() -> None:
            while not stop.wait(poll_seconds):
                fields: dict[str, Any] = {}
                if message is not None:
                    fields["message"] = message
                self.emit_heartbeat(
                    phase=phase,
                    task_id=task_id,
                    **fields,
                )

        heartbeat_thread = threading.Thread(
            target=emit_until_stopped,
            name=f"mammoth-event-heartbeat-{self.rank if self.rank is not None else 'runner'}",
            daemon=True,
        )
        thread_started = False
        try:
            heartbeat_thread.start()
            thread_started = True
        except Exception as error:
            self._failure_logger.warning(
                "Could not start periodic execution-event heartbeats for %s: %s",
                self.path,
                error,
            )
        try:
            yield
        finally:
            if thread_started:
                stop.set()
                heartbeat_thread.join()

    def close(self) -> None:
        """Flush final pending progress and close without propagating I/O failures."""
        with self._lock:
            if self._closed:
                return
            if self.enabled:
                self.flush_progress(final=True)
            stream = self._stream
            self._stream = None
            self._closed = True
            if stream is None:
                return
            try:
                stream.close()
            except Exception as error:
                self._disable_after_io_failure("close", error)

    def __enter__(self) -> ExecutionEventWriter:
        """Return this process-local writer for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Flush and close the event stream without masking computation errors."""
        self.close()

    def _validate_writer_identity(self) -> None:
        _validate_nonempty_string("source", self.source)
        if self.source == "runner":
            if self.rank is not None or self.world_size is not None:
                raise ValueError("runner writers must omit rank and world_size.")
            expected_name = RUNNER_EVENT_STREAM_FILENAME
        elif self.source == "process":
            _validate_int("rank", self.rank, minimum=0)
            _validate_int("world_size", self.world_size, minimum=1)
            assert self.rank is not None and self.world_size is not None
            if self.rank >= self.world_size:
                raise ValueError(
                    f"rank={self.rank} is outside writer world_size={self.world_size}."
                )
            expected_name = PROCESS_EVENT_STREAM_TEMPLATE.format(rank=self.rank)
        else:
            raise ValueError(f"source must be 'runner' or 'process', got {self.source!r}.")
        if self.path.name != expected_name:
            raise ValueError(
                f"{self.source} writer path must end with {expected_name!r}, got {self.path}."
            )

    def _build_event(
        self,
        event: EventName,
        *,
        observed_at: str,
        fields: Mapping[str, Any],
    ) -> ExecutionEvent:
        collisions = _WRITER_IDENTITY_FIELDS.intersection(fields)
        if collisions:
            raise ValueError(
                f"writer-owned event fields cannot be overridden: {', '.join(sorted(collisions))}."
            )
        payload: dict[str, Any] = {
            "schema_version": EXECUTION_EVENT_SCHEMA_VERSION,
            "sequence": self._sequence + 1,
            "time": observed_at,
            "execution_id": self.execution_id,
            "run_name": self.run_name,
            "source": self.source,
            "event": event,
            **fields,
        }
        if self.source == "process":
            payload["rank"] = self.rank
            payload["world_size"] = self.world_size
        return ExecutionEvent.from_dict(payload)

    def _write_event(
        self,
        event: ExecutionEvent,
        *,
        monotonic_now: float,
    ) -> ExecutionEvent | None:
        stream = self._stream
        if stream is None or not self.enabled:
            return None
        if event.sequence != self._sequence + 1:
            event = _with_sequence(event, self._sequence + 1)
        record = event.to_json_line()
        try:
            written = stream.write(record)
            if written != len(record):
                raise OSError(
                    f"short event-stream append: wrote {written!r} of {len(record)} bytes"
                )
            stream.flush()
        except Exception as error:
            self._disable_after_io_failure("write", error)
            return None
        self._sequence = event.sequence
        self._last_activity = monotonic_now
        if event.event == "progress":
            self._last_progress_emitted = monotonic_now
        return event

    def _disable_after_io_failure(self, operation: str, error: Exception) -> None:
        if self._failure_logged:
            return
        self._failure_logged = True
        stream = self._stream
        self._stream = None
        if stream is not None:
            with suppress(Exception):
                stream.close()
        with suppress(Exception):
            self._failure_logger.error(
                "Execution event writer disabled after %s failure for %s: %s",
                operation,
                self.path,
                error,
                exc_info=True,
            )


class ExecutionEventReadError(ValueError):
    """A complete JSONL record failed with precise stream location context."""

    def __init__(
        self,
        path: Path,
        line_number: int,
        detail: str,
        *,
        valid_events: tuple[ExecutionEvent, ...] = (),
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = detail
        self.valid_events = valid_events
        super().__init__(f"Malformed execution event in {path} at line {line_number}: {detail}")


class ExecutionEventTailReader:
    """Poll an append-only event stream while retaining an incomplete final line."""

    def __init__(self, path: Path, *, allow_missing: bool = True) -> None:
        self.path = Path(path)
        self.allow_missing = allow_missing
        self._file_identity: tuple[int, int] | None = None
        self._offset = 0
        self._append_guard = b""
        self._buffer = b""
        self._line_number = 0
        self._next_sequence = 1
        self._stream_identity: (
            tuple[
                str,
                str,
                EventSource,
                int | None,
                int | None,
            ]
            | None
        ) = None
        self._error: ExecutionEventReadError | None = None

    @property
    def line_number(self) -> int:
        """Return the number of complete records consumed, including a failed line."""
        return self._line_number

    @property
    def offset(self) -> int:
        """Return the byte offset already read from the active file identity."""
        return self._offset

    def poll(self) -> list[ExecutionEvent]:
        """Return records appended since the previous poll without blocking."""
        if self._error is not None:
            raise self._error
        try:
            appended = self._read_appended_bytes()
        except FileNotFoundError:
            if self.allow_missing and self._file_identity is None:
                return []
            raise
        if not appended:
            return []

        complete_lines, self._buffer = _split_complete_lines(self._buffer + appended)
        events: list[ExecutionEvent] = []
        for line in complete_lines:
            self._line_number += 1
            try:
                event = _decode_event_line(line)
                self._validate_stream_record(event)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                ValueError,
            ) as error:
                self._fail(self._line_number, str(error), valid_events=tuple(events))
            events.append(event)
        return events

    def _read_appended_bytes(self) -> bytes:
        flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise OSError(f"Execution event stream must be a regular file: {self.path}")
            identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            if self._file_identity is not None and identity != self._file_identity:
                self._fail(
                    self._line_number + 1,
                    "active stream file identity changed",
                )
            if descriptor_stat.st_size < self._offset:
                self._fail(
                    self._line_number + 1,
                    f"active stream was truncated from offset {self._offset}",
                )
            guard_offset = self._offset - len(self._append_guard)
            if (
                self._append_guard
                and os.pread(
                    descriptor,
                    len(self._append_guard),
                    guard_offset,
                )
                != self._append_guard
            ):
                self._fail(
                    self._line_number + 1,
                    "consumed stream prefix changed",
                )
            os.lseek(descriptor, self._offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = descriptor_stat.st_size - self._offset
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    self._fail(
                        self._line_number + 1,
                        "active stream changed while polling",
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            appended = b"".join(chunks)
        except BaseException:
            with suppress(Exception):
                os.close(descriptor)
            raise
        os.close(descriptor)
        self._file_identity = identity
        self._offset += len(appended)
        self._append_guard = (self._append_guard + appended)[-_APPEND_GUARD_BYTES:]
        return appended

    def _validate_stream_record(self, event: ExecutionEvent) -> None:
        if event.sequence != self._next_sequence:
            raise ValueError(
                f"sequence {event.sequence} does not match expected {self._next_sequence}"
            )
        identity = (
            event.execution_id,
            event.run_name,
            event.source,
            event.rank,
            event.world_size,
        )
        if self._stream_identity is None:
            self._stream_identity = identity
        elif identity != self._stream_identity:
            raise ValueError("event identity differs from earlier records in the stream")
        self._next_sequence += 1

    def _fail(
        self,
        line_number: int,
        detail: str,
        *,
        valid_events: tuple[ExecutionEvent, ...] = (),
    ) -> None:
        error = ExecutionEventReadError(
            self.path,
            line_number,
            detail,
            valid_events=valid_events,
        )
        self._error = error
        raise error


def iter_execution_events(path: Path) -> Iterator[ExecutionEvent]:
    """Yield a stable historical snapshot and ignore an incomplete trailing append."""
    reader = ExecutionEventTailReader(path, allow_missing=False)
    try:
        events = reader.poll()
    except ExecutionEventReadError as error:
        yield from error.valid_events
        raise
    yield from events


def read_execution_events(path: Path) -> list[ExecutionEvent]:
    """Return a historical snapshot from :func:`iter_execution_events`."""
    return list(iter_execution_events(path))


def _split_complete_lines(data: bytes) -> tuple[list[bytes], bytes]:
    parts = data.split(b"\n")
    return parts[:-1], parts[-1]


def _decode_event_line(line: bytes) -> ExecutionEvent:
    payload = json.loads(line.decode("utf-8", errors="strict"))
    return ExecutionEvent.from_dict(payload)


def process_event_stream_path(
    context: ExecutionContext,
    rank: int,
    *,
    world_size: int | None = None,
) -> Path:
    """Return the reserved process-owned JSONL path for one validated rank."""
    context.rank_log_path(rank, world_size=world_size)
    return context.execution_dir / PROCESS_EVENT_STREAM_TEMPLATE.format(rank=rank)


def runner_event_stream_path(context: ExecutionContext) -> Path:
    """Return the optional experiment-runner JSONL path for one execution."""
    return context.execution_dir / RUNNER_EVENT_STREAM_FILENAME


def _with_sequence(
    event: ExecutionEvent,
    sequence: int,
    *,
    final: bool | None = None,
) -> ExecutionEvent:
    changes: dict[str, Any] = {"sequence": sequence}
    if final is not None:
        changes["final"] = final
    return replace(event, **changes)


def _validate_interval(name: str, value: Any) -> float:
    _validate_finite_number(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}.")
    return float(value)


def _terminal_closes_progress(
    terminal: ExecutionEvent,
    progress: ExecutionEvent,
) -> bool:
    if terminal.event.startswith("task_"):
        return terminal.phase == progress.phase and terminal.task_id == progress.task_id
    if terminal.event.startswith("phase_"):
        return terminal.phase == progress.phase
    return terminal.event.startswith(("process_", "execution_"))


def _same_progress_scope(
    first: ExecutionEvent,
    second: ExecutionEvent,
) -> bool:
    """Return whether two replaceable observations belong to the same task."""
    return first.phase == second.phase and first.task_id == second.task_id


def _monotonic_time(clock: Callable[[], float]) -> float:
    value = clock()
    _validate_finite_number("monotonic clock", value)
    return float(value)


def _open_event_stream(path: Path) -> BinaryIO:
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(f"Execution event stream must be a regular file: {path}")
        return cast(BinaryIO, os.fdopen(descriptor, "r+b", buffering=0))
    except Exception:
        os.close(descriptor)
        raise


def _existing_stream_sequence(
    stream: BinaryIO,
    path: Path,
    *,
    execution_id: str,
    run_name: str,
    source: EventSource,
    rank: int | None,
    world_size: int | None,
) -> int:
    expected_sequence = 1
    stream.seek(0)
    for line_number, line in enumerate(stream, start=1):
        if not line.endswith(b"\n"):
            raise ValueError(
                f"Cannot append after incomplete event record in {path} at line {line_number}."
            )
        try:
            event = _decode_event_line(line[:-1])
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Cannot append after malformed event record in {path} "
                f"at line {line_number}: {error}"
            ) from error
        if event.sequence != expected_sequence:
            raise ValueError(
                f"Cannot append after sequence {event.sequence} in {path} at line "
                f"{line_number}; expected {expected_sequence}."
            )
        if (
            event.execution_id != execution_id
            or event.run_name != run_name
            or event.source != source
            or event.rank != rank
            or event.world_size != world_size
        ):
            raise ValueError(
                f"Cannot append event identity mismatch in {path} at line {line_number}."
            )
        expected_sequence += 1
    stream.seek(0, os.SEEK_END)
    return expected_sequence - 1


def utc_event_time() -> str:
    """Return a schema-compatible UTC timestamp for event producers."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validated_display_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("display_metrics must be a mapping.")
    if len(metrics) > MAX_DISPLAY_METRICS:
        raise ValueError(f"display_metrics may contain at most {MAX_DISPLAY_METRICS} entries.")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        _validate_nonempty_string("display metric name", name)
        _validate_finite_number(f"display_metrics[{name!r}]", value)
        validated[name] = float(value)
    return validated


def _validated_coordinates(
    coordinates: Mapping[str, int | float | str],
) -> dict[str, int | float | str]:
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinates must be a mapping.")
    validated: dict[str, int | float | str] = {}
    for name, value in coordinates.items():
        _validate_nonempty_string("coordinate name", name)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(
                f"coordinates[{name!r}] must be an integer, finite number, or string."
            )
        if isinstance(value, float):
            _validate_finite_number(f"coordinates[{name!r}]", value)
        if isinstance(value, str):
            _validate_nonempty_string(f"coordinates[{name!r}]", value)
        validated[name] = value
    return validated


def _validated_extensions(extensions: Mapping[str, Any]) -> dict[str, ImmutableJsonValue]:
    if not isinstance(extensions, Mapping):
        raise ValueError("extensions must be a mapping.")
    collisions = _KNOWN_FIELDS.intersection(extensions)
    if collisions:
        raise ValueError(f"extensions duplicate known fields: {', '.join(sorted(collisions))}.")
    sanitized = sanitize_metadata_fields(extensions)
    return {key: _freeze_json(value) for key, value in sanitized.items()}


def _freeze_json(value: Any) -> ImmutableJsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return cast(None | bool | int | float | str, value)


def _thaw_json(value: ImmutableJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _parse_utc_time(value: str) -> datetime:
    _validate_nonempty_string("time", value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid UTC event timestamp: {value!r}.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"Event timestamp must use UTC: {value!r}.")
    return parsed


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    _validate_nonempty_string(key, value)
    return cast(str, value)


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    _validate_nonempty_string(key, value)
    return cast(str, value)


def _required_int(payload: Mapping[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = payload.get(key)
    _validate_int(key, value, minimum=minimum)
    return cast(int, value)


def _optional_int(
    payload: Mapping[str, Any], key: str, *, minimum: int | None = None
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    _validate_int(key, value, minimum=minimum)
    return cast(int, value)


def _optional_number(
    payload: Mapping[str, Any], key: str, *, minimum: float | None = None
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    _validate_finite_number(key, value, minimum=minimum)
    return float(value)


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean, got {value!r}.")
    return value


def _optional_signal(payload: Mapping[str, Any]) -> int | str | None:
    value = payload.get("signal")
    if value is None:
        return None
    if not isinstance(value, int | str) or isinstance(value, bool):
        raise ValueError("signal must be an integer or non-empty string.")
    return value


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _validate_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}.")


def _validate_int(name: str, value: Any, *, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}.")


def _validate_finite_number(name: str, value: Any, *, minimum: float | None = None) -> None:
    is_finite = False
    if isinstance(value, int | float) and not isinstance(value, bool):
        with suppress(OverflowError):
            is_finite = math.isfinite(value)
    if not is_finite:
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}.")
