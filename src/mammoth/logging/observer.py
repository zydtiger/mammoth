"""Producer-facing observation facade faning records out to independent sinks.

Workflows and trainers use ``RunObserver``. Sink failures are isolated so an
observability outage never terminates the caller's workload.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import Any, Protocol

from mammoth.core.events import EventName
from mammoth.logging.model import Media, Observation

logger = logging.getLogger(__name__)


class ObservationSink(Protocol):
    """Backend contract consumed by :class:`RunObserver`."""

    def observe(self, observation: Observation) -> None:
        """Consume one immutable observation."""

    def flush(self) -> None:
        """Publish buffered records."""

    def close(self) -> None:
        """Flush and release backend resources."""


class RunObserver:
    """Fan lifecycle, progress, metrics, and media out to configured sinks."""

    def __init__(self, sinks: Sequence[ObservationSink] = ()) -> None:
        self._sinks = tuple(sinks)
        self._disabled: set[int] = set()
        self._closed = False

    @property
    def disabled_sink_count(self) -> int:
        """Return the number of backends disabled after an exception."""
        return len(self._disabled)

    def emit(
        self,
        event: EventName,
        *,
        metrics: Mapping[str, float] | None = None,
        display_metrics: Mapping[str, float] | None = None,
        media: Mapping[str, Media] | None = None,
        **fields: Any,
    ) -> Observation:
        """Validate and dispatch one observation without propagating sink I/O errors."""
        if self._closed:
            raise RuntimeError("RunObserver is closed")
        dense_metrics = {} if metrics is None else metrics
        displayed = dense_metrics if display_metrics is None else display_metrics
        observation = Observation(
            event=event,
            fields=fields,
            metrics=dense_metrics,
            display_metrics=displayed,
            media={} if media is None else media,
        )
        for sink in self._sinks:
            identifier = id(sink)
            if identifier in self._disabled:
                continue
            try:
                sink.observe(observation)
            except Exception:
                self._disabled.add(identifier)
                logger.exception("Disabled observation sink %s after failure", type(sink).__name__)
                with suppress(Exception):
                    sink.close()
        return observation

    def progress(
        self,
        *,
        phase: str,
        task_id: str,
        completed: int,
        total: int | None = None,
        metrics: Mapping[str, float] | None = None,
        display_metrics: Mapping[str, float] | None = None,
        coordinates: Mapping[str, int | float | str] | None = None,
        final: bool = False,
        unit: str | None = None,
        message: str | None = None,
        media: Mapping[str, Media] | None = None,
    ) -> Observation:
        """Dispatch a progress observation to JSONL and dense-history sinks."""
        fields: dict[str, Any] = {
            "phase": phase,
            "task_id": task_id,
            "completed": completed,
            "final": final,
        }
        if total is not None:
            fields["total"] = total
        if coordinates:
            fields["coordinates"] = coordinates
        if unit is not None:
            fields["unit"] = unit
        if message is not None:
            fields["message"] = message
        return self.emit(
            "progress",
            metrics=metrics,
            display_metrics=display_metrics,
            media=media,
            **fields,
        )

    def heartbeat(
        self,
        *,
        phase: str,
        task_id: str | None = None,
        metrics: Mapping[str, float] | None = None,
        display_metrics: Mapping[str, float] | None = None,
        coordinates: Mapping[str, int | float | str] | None = None,
        force: bool = False,
        message: str | None = None,
    ) -> Observation:
        """Dispatch a throttled or explicitly forced producer heartbeat."""
        fields: dict[str, Any] = {"phase": phase, "force": force}
        if task_id is not None:
            fields["task_id"] = task_id
        if coordinates:
            fields["coordinates"] = coordinates
        if message is not None:
            fields["message"] = message
        return self.emit(
            "heartbeat",
            metrics=metrics,
            display_metrics=display_metrics,
            **fields,
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Emit balanced phase lifecycle records around caller-owned work."""
        self.emit("phase_started", phase=name)
        try:
            yield
        except BaseException as error:
            self.emit("phase_failed", phase=name, message=str(error))
            raise
        else:
            self.emit("phase_completed", phase=name)

    @contextmanager
    def task(
        self,
        phase: str,
        task_id: str,
        *,
        parent_task_id: str | None = None,
    ) -> Iterator[None]:
        """Emit balanced task lifecycle records around caller-owned work."""
        fields: dict[str, Any] = {"phase": phase, "task_id": task_id}
        if parent_task_id is not None:
            fields["parent_task_id"] = parent_task_id
        self.emit("task_started", **fields)
        try:
            yield
        except BaseException as error:
            self.emit("task_failed", message=str(error), **fields)
            raise
        else:
            self.emit("task_completed", **fields)

    def flush(self) -> None:
        """Flush every enabled sink while isolating failures."""
        for sink in self._sinks:
            identifier = id(sink)
            if identifier in self._disabled:
                continue
            try:
                sink.flush()
            except Exception:
                self._disabled.add(identifier)
                logger.exception("Disabled observation sink %s during flush", type(sink).__name__)

    def close(self) -> None:
        """Close every sink once in reverse construction order."""
        if self._closed:
            return
        self._closed = True
        for sink in reversed(self._sinks):
            with suppress(Exception):
                sink.close()

    def __enter__(self) -> RunObserver:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
