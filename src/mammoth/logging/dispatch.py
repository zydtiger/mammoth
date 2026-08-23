"""Bounded per-sink background dispatch for immutable observations.

The dispatcher is deliberately owned by :class:`mammoth.logging.RunObserver`.
It keeps each concrete sink single-threaded while allowing independent sinks to
make progress independently.  Ordinary JSONL progress may be coalesced before
it reaches its worker; dense-history sinks retain FIFO backpressure instead.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from mammoth.core.pipeline import (
    BackgroundPipelineError,
    BackgroundPipelineSubmission,
    BoundedBackgroundPipeline,
)
from mammoth.logging.model import Observation


class _Sink(Protocol):
    """Minimal sink surface consumed by one background worker."""

    def observe(self, observation: Observation) -> None:
        """Consume one immutable observation."""

    def flush(self) -> None:
        """Publish buffered records."""

    def close(self) -> None:
        """Release sink resources."""


class AsyncObservationSink:
    """Dispatch one sink through an ordered bounded background pipeline.

    The worker never shares its concrete sink with another worker.  When
    ``coalesce_progress`` is enabled, only non-final progress records that
    have not reached the worker are replaceable; every other record applies
    normal bounded FIFO backpressure.
    """

    def __init__(
        self,
        sink: _Sink,
        *,
        max_pending: int,
        coalesce_progress: bool,
    ) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.sink = sink
        self.max_pending = max_pending
        self.coalesce_progress = coalesce_progress
        self._lock = threading.RLock()
        self._closed = False
        self._failure: BaseException | None = None
        self._pending_progress: OrderedDict[tuple[str, str], Observation] = OrderedDict()
        self._pipeline: BoundedBackgroundPipeline[Callable[[], None], None] = (
            BoundedBackgroundPipeline(
                _run_operation,
                max_pending=max_pending,
                thread_name_prefix=f"mammoth-observation-{type(sink).__name__.lower()}",
            )
        )

    @property
    def failed(self) -> bool:
        """Return whether the worker has observed a sink failure."""
        with self._lock:
            return self._failure is not None

    def observe(self, observation: Observation) -> None:
        """Submit one observation, coalescing only eligible progress records."""
        with self._lock:
            self._require_open()
            self._raise_failure()
            if self._is_replaceable_progress(observation):
                self._retain_progress(observation)
                self._drain_available_progress()
                return
            barrier = self._requires_barrier(observation)
            if barrier:
                self._drain_progress()
            self._submit_observation(observation, wait=barrier)
            if barrier:
                self._submit_operation(self.sink.flush, wait=True)

    def flush(self) -> None:
        """Deliver coalesced progress, then wait for a sink flush barrier."""
        with self._lock:
            if self._closed:
                return
            self._raise_failure()
            self._drain_progress()
            self._submit_operation(self.sink.flush, wait=True)

    def close(self) -> None:
        """Drain work and close the concrete sink on its owning worker.

        Each step below records its own failure independently. An earlier
        step raising must never skip the final ``sink.close`` submission: a
        prior fire-and-forget dispatch can still be sitting in the pipeline
        as an unacknowledged failure when this runs, which would otherwise
        make every later submission attempt on this sink -- including the
        close itself -- re-raise that same stale failure and return without
        ever releasing the concrete sink's resources.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._drain_progress(allow_closed=True)
            except BaseException as error:
                self._remember_failure(error)
            try:
                self._submit_operation(self.sink.flush, wait=True, allow_closed=True)
            except BaseException as error:
                self._remember_failure(error)
            try:
                self._submit_operation(self.sink.close, wait=True, allow_closed=True)
            except BaseException as error:
                self._remember_failure(error)
        self._close_pipeline()

    def _retain_progress(self, observation: Observation) -> None:
        scope = _progress_scope(observation)
        if scope in self._pending_progress:
            self._pending_progress.move_to_end(scope)
            self._pending_progress[scope] = observation
            return
        if len(self._pending_progress) >= self.max_pending:
            _, oldest = self._pending_progress.popitem(last=False)
            self._submit_observation(oldest, wait=False)
        self._pending_progress[scope] = observation

    def _drain_progress(self, *, allow_closed: bool = False) -> None:
        while self._pending_progress:
            _, observation = self._pending_progress.popitem(last=False)
            self._submit_observation(observation, wait=False, allow_closed=allow_closed)

    def _drain_available_progress(self) -> None:
        while self._pending_progress and self._pipeline.pending_count < self.max_pending:
            _, observation = self._pending_progress.popitem(last=False)
            self._submit_observation(observation, wait=False)

    def _submit_observation(
        self,
        observation: Observation,
        *,
        wait: bool,
        allow_closed: bool = False,
    ) -> None:
        self._submit_operation(
            lambda: self.sink.observe(observation),
            wait=wait,
            allow_closed=allow_closed,
        )

    def _submit_operation(
        self,
        operation: Callable[[], None],
        *,
        wait: bool,
        allow_closed: bool = False,
    ) -> None:
        if not allow_closed:
            self._require_open()
        submission = self._submit_past_stale_failures(operation)
        if wait:
            try:
                submission.result()
            except BaseException as error:
                self._remember_failure(error)
                raise self._dispatch_error() from error

    def _submit_past_stale_failures(
        self,
        operation: Callable[[], None],
    ) -> BackgroundPipelineSubmission[Callable[[], None], None]:
        """Submit one operation, clearing any stale unacknowledged failures first.

        A fire-and-forget dispatch that failed on the worker is not
        acknowledged until its completion callback runs on the pipeline's own
        callback thread. Until then, the pipeline synchronously re-raises
        that same recorded failure for every new submission attempt on this
        sink, regardless of what the new operation is. More than one such
        failure can accumulate this way before the callback thread catches
        up. Acknowledging each one as it is found and retrying lets a
        genuinely new operation -- notably ``close()``'s flush and close
        steps -- still run instead of being permanently blocked by failures
        the caller has already recorded.
        """
        while True:
            try:
                return self._pipeline.submit(operation, on_done=self._complete_submission)
            except BackgroundPipelineError as error:
                self._remember_failure(error.cause)
                with suppress(Exception):
                    self._pipeline.acknowledge(error.submission)

    def _complete_submission(
        self,
        submission: BackgroundPipelineSubmission[Callable[[], None], None],
    ) -> None:
        completed = False
        try:
            submission.result()
            completed = True
        except BaseException as error:
            with self._lock:
                self._remember_failure(error)
        finally:
            try:
                self._pipeline.acknowledge(submission)
            except BaseException as error:
                with self._lock:
                    self._remember_failure(error)
        if completed:
            with self._lock:
                if not self._closed and self._failure is None:
                    try:
                        self._drain_available_progress()
                    except BaseException as error:
                        self._remember_failure(error)

    def _close_pipeline(self) -> None:
        """Acknowledge close-time failures before retrying worker shutdown."""
        while True:
            try:
                self._pipeline.close()
                return
            except BackgroundPipelineError as error:
                with self._lock:
                    self._remember_failure(error.cause)
                try:
                    self._pipeline.acknowledge(error.submission)
                except BaseException as acknowledge_error:
                    with self._lock:
                        self._remember_failure(acknowledge_error)
                    return
            except BaseException as error:
                with self._lock:
                    self._remember_failure(error)
                return

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("asynchronous observation sink is closed")

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._dispatch_error() from self._failure

    def _remember_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error

    def _dispatch_error(self) -> RuntimeError:
        if self._failure is None:
            return RuntimeError("asynchronous observation dispatch failed")
        return RuntimeError(
            f"asynchronous observation dispatch failed: {type(self._failure).__name__}: "
            f"{self._failure}"
        )

    def _is_replaceable_progress(self, observation: Observation) -> bool:
        return (
            self.coalesce_progress
            and observation.event == "progress"
            and not bool(observation.fields.get("final", False))
        )

    @staticmethod
    def _requires_barrier(observation: Observation) -> bool:
        return observation.event != "progress" or bool(observation.fields.get("final", False))


def _progress_scope(observation: Observation) -> tuple[str, str]:
    """Return the validated task scope of a progress observation."""
    phase = observation.fields.get("phase")
    task_id = observation.fields.get("task_id")
    if not isinstance(phase, str) or not isinstance(task_id, str):
        raise ValueError("progress observations require string phase and task_id")
    return phase, task_id


def _run_operation(operation: Callable[[], None]) -> None:
    """Run one sink operation in the pipeline's dedicated worker."""
    operation()
