"""Producer-facing observation facade faning records out to independent sinks.

Workflows and trainers use ``RunObserver``. Sink failures are isolated so an
observability outage never terminates the caller's workload.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import Any, Protocol

from mammoth.core.events import DEFAULT_HEARTBEAT_INTERVAL_SECONDS, EventName
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

    def __init__(
        self,
        sinks: Sequence[ObservationSink] = (),
        *,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sinks = tuple(sinks)
        self._disabled: set[int] = set()
        self._closed = False
        self._heartbeat_interval_seconds = _validate_heartbeat_interval(
            heartbeat_interval_seconds
        )
        self._monotonic_clock = monotonic_clock
        self._state_lock = threading.Lock()
        self._dispatch_lock = threading.RLock()
        self._heartbeat_threads: dict[threading.Thread, threading.Event] = {}
        self._close_complete = threading.Event()
        self._closing_thread_id: int | None = None
        self._last_activity = monotonic_clock()

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
        logical_step: int | None = None,
        **fields: Any,
    ) -> Observation:
        """Validate and dispatch one observation without propagating sink I/O errors."""
        with self._dispatch_lock:
            if "unit" in fields:
                raise TypeError("unit is no longer supported; progress values are unitless.")
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("RunObserver is closed")
                self._last_activity = self._monotonic_clock()
            dense_metrics = {} if metrics is None else metrics
            displayed = dense_metrics if display_metrics is None else display_metrics
            observation = Observation(
                event=event,
                fields=fields,
                metrics=dense_metrics,
                display_metrics=displayed,
                media={} if media is None else media,
                logical_step=logical_step,
            )
            for sink in self._sinks:
                identifier = id(sink)
                if identifier in self._disabled:
                    continue
                try:
                    sink.observe(observation)
                except Exception:
                    self._disabled.add(identifier)
                    logger.exception(
                        "Disabled observation sink %s after failure",
                        type(sink).__name__,
                    )
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
        message: str | None = None,
        media: Mapping[str, Media] | None = None,
        logical_step: int | None = None,
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
        if message is not None:
            fields["message"] = message
        return self.emit(
            "progress",
            metrics=metrics,
            display_metrics=display_metrics,
            media=media,
            logical_step=logical_step,
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
        logical_step: int | None = None,
    ) -> Observation:
        """Dispatch a producer heartbeat for backend-specific retention."""
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
            logical_step=logical_step,
            **fields,
        )

    @contextmanager
    def periodic_heartbeats(
        self,
        *,
        phase: str,
        task_id: str | None = None,
        message: str | None = None,
    ) -> Iterator[None]:
        """Emit observer heartbeats while caller-owned work is otherwise idle."""
        stop = threading.Event()
        poll_seconds = min(1.0, self._heartbeat_interval_seconds)

        def emit_until_stopped() -> None:
            while not stop.wait(poll_seconds):
                with self._dispatch_lock:
                    if not self._reserve_periodic_heartbeat():
                        continue
                    try:
                        self.heartbeat(
                            phase=phase,
                            task_id=task_id,
                            message=message,
                        )
                    except RuntimeError:
                        if self._closed:
                            return
                        logger.exception("Could not emit a periodic observer heartbeat")
                    except Exception:
                        logger.exception("Could not emit a periodic observer heartbeat")

        heartbeat_thread = threading.Thread(
            target=emit_until_stopped,
            name="mammoth-observer-heartbeat",
            daemon=True,
        )
        thread_started = False
        with self._state_lock:
            if self._closed:
                raise RuntimeError("RunObserver is closed")
            try:
                heartbeat_thread.start()
                thread_started = True
                self._heartbeat_threads[heartbeat_thread] = stop
            except Exception as error:
                logger.warning("Could not start periodic observer heartbeats: %s", error)
        try:
            yield
        finally:
            if thread_started:
                stop.set()
                if heartbeat_thread is not threading.current_thread():
                    heartbeat_thread.join()
                with self._state_lock:
                    self._heartbeat_threads.pop(heartbeat_thread, None)

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
        with self._dispatch_lock:
            for sink in self._sinks:
                identifier = id(sink)
                if identifier in self._disabled:
                    continue
                try:
                    sink.flush()
                except Exception:
                    self._disabled.add(identifier)
                    logger.exception(
                        "Disabled observation sink %s during flush",
                        type(sink).__name__,
                    )

    def close(self) -> None:
        """Close every sink once in reverse construction order."""
        current_thread_id = threading.get_ident()
        with self._state_lock:
            if self._closed:
                owns_close = False
                should_wait = (
                    not self._close_complete.is_set()
                    and self._closing_thread_id != current_thread_id
                )
                heartbeat_threads: tuple[threading.Thread, ...] = ()
            else:
                owns_close = True
                self._closed = True
                self._closing_thread_id = current_thread_id
                heartbeat_threads = tuple(self._heartbeat_threads)
                for stop in self._heartbeat_threads.values():
                    stop.set()
                should_wait = False
        if should_wait:
            self._close_complete.wait()
            return
        if not owns_close:
            return
        try:
            for heartbeat_thread in heartbeat_threads:
                if heartbeat_thread is not threading.current_thread():
                    heartbeat_thread.join()
            with self._dispatch_lock:
                for sink in reversed(self._sinks):
                    with suppress(Exception):
                        sink.close()
        finally:
            self._close_complete.set()

    def _reserve_periodic_heartbeat(self) -> bool:
        with self._state_lock:
            if self._closed:
                return False
            monotonic_now = self._monotonic_clock()
            if monotonic_now - self._last_activity < self._heartbeat_interval_seconds:
                return False
            self._last_activity = monotonic_now
            return True

    def __enter__(self) -> RunObserver:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def _validate_heartbeat_interval(value: float) -> float:
    """Return one positive finite observer heartbeat interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("heartbeat_interval_seconds must be a positive finite number")
    return float(value)
