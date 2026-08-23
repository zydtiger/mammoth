"""Run bounded ordered background work for higher Mammoth layers and consumers.

The framework-neutral :mod:`mammoth.core` package exposes this worker lifecycle
to checkpoint publication, execution sessions, and consuming projects without
depending on PyTorch or project-specific artifact types.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread, current_thread
from types import TracebackType
from typing import Literal

type _SubmissionState = Literal["queued", "running", "completed", "failed"]


class BackgroundPipelineSubmission[InputT, ResultT]:
    """Provide read-only access to one accepted input and its worker outcome."""

    __slots__ = (
        "_callback_dispatch",
        "_callback_lock",
        "_callbacks",
        "_callbacks_closed",
        "_error",
        "_future",
        "_input",
        "_pipeline_token",
        "_state",
        "_state_finalized",
    )

    def __init__(
        self,
        input_value: InputT,
        pipeline_token: object,
        callback_dispatch: Callable[
            [
                BackgroundPipelineSubmission[InputT, ResultT],
                Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
                bool,
            ],
            None,
        ],
    ) -> None:
        self._input = input_value
        self._future: Future[ResultT] = Future()
        self._state: _SubmissionState = "queued"
        self._state_finalized = Event()
        self._pipeline_token = pipeline_token
        self._error: BackgroundPipelineError[InputT, ResultT] | None = None
        self._callback_dispatch = callback_dispatch
        self._callback_lock = RLock()
        self._callbacks: list[
            tuple[
                Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
                bool,
            ]
        ] = []
        self._callbacks_closed = False

    def done(self) -> bool:
        """Return whether the worker has produced a result or failure."""
        return self._future.done() and self._state_finalized.is_set()

    @property
    def input(self) -> InputT:
        """Return the immutable input association for this submission."""
        return self._input

    def result(self, timeout: float | None = None) -> ResultT:
        """Wait for and return the worker result, or raise its original failure."""
        try:
            result = self._future.result(timeout=timeout)
        except BaseException:
            self._state_finalized.wait()
            raise
        self._state_finalized.wait()
        return result

    def add_done_callback(
        self,
        callback: Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
        *,
        after_idle: bool = True,
    ) -> None:
        """Invoke a best-effort callback after pipeline outcome state is final."""
        if not callable(callback):
            raise TypeError("done callback must be callable")
        if not isinstance(after_idle, bool):
            raise TypeError("after_idle must be a boolean")

        with self._callback_lock:
            if not self._callbacks_closed:
                self._callbacks.append((callback, after_idle))
                return
        self._callback_dispatch(self, callback, after_idle)

    def _release_done_callbacks(
        self,
    ) -> tuple[
        tuple[
            Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
            bool,
        ],
        ...,
    ]:
        with self._callback_lock:
            self._callbacks_closed = True
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            return callbacks

    def __repr__(self) -> str:
        return f"BackgroundPipelineSubmission(input={self.input!r})"


@dataclass(frozen=True, slots=True)
class BackgroundPipelineResult[InputT, ResultT]:
    """Associate one successfully completed result with its submission."""

    submission: BackgroundPipelineSubmission[InputT, ResultT]
    result: ResultT

    @property
    def input(self) -> InputT:
        """Return the input whose worker call produced this result."""
        return self.submission.input


class BackgroundPipelineError[InputT, ResultT](RuntimeError):
    """Attribute one background worker failure to its accepted submission."""

    def __init__(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
        cause: BaseException,
    ) -> None:
        self._submission = submission
        self._cause = cause
        super().__init__(f"Background work failed: {type(cause).__name__}: {cause}")

    @property
    def submission(self) -> BackgroundPipelineSubmission[InputT, ResultT]:
        """Return the immutable submission whose worker call failed."""
        return self._submission

    @property
    def cause(self) -> BaseException:
        """Return the immutable original worker failure."""
        return self._cause

    @property
    def input(self) -> InputT:
        """Return the input whose worker call failed."""
        return self.submission.input


class BoundedBackgroundPipeline[InputT, ResultT]:
    """Execute accepted inputs in order on one worker with bounded backpressure.

    Callers retain ownership until :meth:`submit` accepts an input. The pipeline
    then owns that submission until the caller observes its typed result or
    attributed failure and explicitly calls :meth:`acknowledge`. Accepted work
    is never cancelled.
    """

    def __init__(
        self,
        worker: Callable[[InputT], ResultT],
        *,
        max_pending: int = 1,
        thread_name_prefix: str = "mammoth-background",
    ) -> None:
        if not callable(worker):
            raise TypeError("worker must be callable")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
            raise ValueError("thread_name_prefix must be a non-empty string")
        self.worker = worker
        self.max_pending = max_pending
        self._submissions: deque[BackgroundPipelineSubmission[InputT, ResultT]] = deque()
        self._pipeline_token = object()
        self._accepting = True
        self._closed = False
        self._worker_stop = False
        self._callback_stop = False
        self._callback_queue: deque[
            tuple[
                BackgroundPipelineSubmission[InputT, ResultT],
                Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
            ]
        ] = deque()
        self._idle_callback_queue: deque[
            tuple[
                BackgroundPipelineSubmission[InputT, ResultT],
                Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
            ]
        ] = deque()
        self._deferred_interrupt: KeyboardInterrupt | SystemExit | None = None
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._worker_thread = Thread(
            target=self._worker_loop,
            name=f"{thread_name_prefix}_0",
            daemon=True,
        )
        self._callback_thread = Thread(
            target=self._callback_loop,
            name=f"{thread_name_prefix}-callbacks",
            daemon=True,
        )
        self._callback_thread.start()
        self._worker_thread.start()

    @property
    def pending_count(self) -> int:
        """Return the number of queued or running accepted submissions."""
        with self._condition:
            self._raise_next_failure()
            return self._active_count()

    def wait_for_submission_slot(self) -> None:
        """Apply backpressure before a caller prepares its next input."""
        with self._condition:
            self._require_accepting()
            self._await_submission_slot()

    def submit(
        self,
        input_value: InputT,
        *,
        on_done: Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object] | None = None,
    ) -> BackgroundPipelineSubmission[InputT, ResultT]:
        """Transfer one input to the ordered worker after bounded backpressure."""
        if on_done is not None and not callable(on_done):
            raise TypeError("on_done must be callable or None")
        with self._condition:
            self._require_accepting()
            self._await_submission_slot()
            submission = BackgroundPipelineSubmission[InputT, ResultT](
                input_value,
                self._pipeline_token,
                self._queue_done_callback,
            )
            if on_done is not None:
                submission.add_done_callback(on_done, after_idle=False)
            try:
                self._submissions.append(submission)
                self._condition.notify_all()
                return self._complete_submission_handoff(submission)
            except (KeyboardInterrupt, SystemExit) as error:
                if not any(item is submission for item in self._submissions):
                    raise
                self._condition.notify_all()
                if self._deferred_interrupt is None:
                    self._deferred_interrupt = error
            return submission

    def owns(self, submission: BackgroundPipelineSubmission[InputT, ResultT]) -> bool:
        """Return whether an accepted submission still belongs to this pipeline."""
        with self._condition:
            return any(item is submission for item in self._submissions)

    def owns_input(self, input_value: InputT) -> bool:
        """Return whether this pipeline owns a submission for the exact input object."""
        with self._condition:
            return any(item.input is input_value for item in self._submissions)

    def take_deferred_interrupt(self) -> KeyboardInterrupt | SystemExit | None:
        """Return and clear an interruption delivered after work acceptance."""
        with self._condition:
            interrupt = self._deferred_interrupt
            self._deferred_interrupt = None
            return interrupt

    def take_completed(self) -> tuple[BackgroundPipelineResult[InputT, ResultT], ...]:
        """Return unacknowledged ordered results without waiting for active work."""
        with self._condition:
            self._raise_next_failure()
            return self._completed_results()

    def acknowledge(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
    ) -> None:
        """Release one observed completed result or attributed failure."""
        with self._condition:
            if submission._pipeline_token is not self._pipeline_token:
                raise ValueError("submission belongs to another background pipeline")
            if submission._state in {"queued", "running"}:
                raise RuntimeError("cannot acknowledge background work before completion")
            if not any(item is submission for item in self._submissions):
                return
            try:
                self._submissions.remove(submission)
            except (KeyboardInterrupt, SystemExit) as error:
                if any(item is submission for item in self._submissions):
                    raise
                if self._deferred_interrupt is None:
                    self._deferred_interrupt = error

    def flush(self) -> tuple[BackgroundPipelineResult[InputT, ResultT], ...]:
        """Wait for accepted work, then expose unacknowledged results or failure."""
        with self._condition:
            while self._active_count() > 0:
                self._condition.wait()
            self._raise_next_failure()
            return self._completed_results()

    def close(self) -> tuple[BackgroundPipelineResult[InputT, ResultT], ...]:
        """Flush accepted work and stop the worker exactly once."""
        with self._condition:
            if self._closed:
                return self._completed_results()
            self._accepting = False
        try:
            completed = self.flush()
        except BaseException:
            self._shutdown_worker()
            self._shutdown_callbacks()
            raise
        self._shutdown_worker()
        self._shutdown_callbacks()
        with self._condition:
            self._closed = True
        return completed

    def __enter__(self) -> BoundedBackgroundPipeline[InputT, ResultT]:
        """Return this open pipeline."""
        with self._condition:
            self._require_accepting()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close without replacing an active caller exception."""
        del exc_type, traceback
        cleanup_errors: list[BaseException] = []
        while not self._closed:
            try:
                completed = self.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                if isinstance(cleanup_error, BackgroundPipelineError):
                    self._acknowledge_for_cleanup(
                        cleanup_error.submission,
                        cleanup_errors,
                    )
                continue
            for result in completed:
                self._acknowledge_for_cleanup(result.submission, cleanup_errors)
        if not cleanup_errors:
            return
        if exc_value is None:
            first_error = cleanup_errors[0]
            for later_error in cleanup_errors[1:]:
                first_error.add_note(
                    "Later background pipeline cleanup failure: "
                    f"{type(later_error).__name__}: {later_error}"
                )
            raise first_error
        for recorded_error in cleanup_errors:
            exc_value.add_note(
                "Background pipeline cleanup failed: "
                f"{type(recorded_error).__name__}: {recorded_error}"
            )

    def _worker_loop(self) -> None:
        """Claim accepted submissions in order until pipeline shutdown."""
        while True:
            with self._condition:
                submission = self._next_queued_submission()
                while submission is None:
                    if self._worker_stop:
                        return
                    self._condition.wait()
                    submission = self._next_queued_submission()
                submission._state = "running"
            try:
                result = self.worker(submission.input)
            except BaseException as cause:
                with suppress(BaseException):
                    submission._future.set_exception(cause)
                with self._condition:
                    submission._state = "failed"
                    submission._error = BackgroundPipelineError(submission, cause)
                    submission._state_finalized.set()
                    self._condition.notify_all()
                self._dispatch_submission_callbacks(submission)
            else:
                with suppress(BaseException):
                    submission._future.set_result(result)
                with self._condition:
                    submission._state = "completed"
                    submission._state_finalized.set()
                    self._condition.notify_all()
                self._dispatch_submission_callbacks(submission)

    def _callback_loop(self) -> None:
        """Dispatch completed-submission callbacks without blocking the worker."""
        while True:
            with self._condition:
                while True:
                    if self._callback_queue:
                        submission, callback = self._callback_queue.popleft()
                        break
                    if self._idle_callback_queue and self._active_count() == 0:
                        submission, callback = self._idle_callback_queue.popleft()
                        break
                    if self._callback_stop:
                        return
                    self._condition.wait()
            with suppress(BaseException):
                callback(submission)

    def _dispatch_submission_callbacks(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
    ) -> None:
        callbacks = submission._release_done_callbacks()
        if not callbacks:
            return
        with self._condition:
            for callback, after_idle in callbacks:
                queue = self._idle_callback_queue if after_idle else self._callback_queue
                queue.append((submission, callback))
            self._condition.notify_all()

    def _queue_done_callback(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
        callback: Callable[[BackgroundPipelineSubmission[InputT, ResultT]], object],
        after_idle: bool,
    ) -> None:
        run_inline = False
        with self._condition:
            if self._callback_stop and not self._callback_thread.is_alive():
                run_inline = True
            else:
                queue = self._idle_callback_queue if after_idle else self._callback_queue
                queue.append((submission, callback))
                self._condition.notify_all()
        if run_inline:
            with suppress(BaseException):
                callback(submission)

    def _next_queued_submission(
        self,
    ) -> BackgroundPipelineSubmission[InputT, ResultT] | None:
        return next(
            (submission for submission in self._submissions if submission._state == "queued"),
            None,
        )

    def _active_count(self) -> int:
        return sum(submission._state in {"queued", "running"} for submission in self._submissions)

    def _await_submission_slot(self) -> None:
        self._raise_next_failure()
        while self._active_count() >= self.max_pending:
            self._condition.wait()
            self._require_accepting()
            self._raise_next_failure()

    def _complete_submission_handoff(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
    ) -> BackgroundPipelineSubmission[InputT, ResultT]:
        return submission

    def _raise_next_failure(self) -> None:
        for submission in self._submissions:
            if submission._state != "failed" or submission._error is None:
                continue
            raise submission._error from submission._error.cause

    def _completed_results(self) -> tuple[BackgroundPipelineResult[InputT, ResultT], ...]:
        return tuple(
            BackgroundPipelineResult(submission, submission.result())
            for submission in self._submissions
            if submission._state == "completed"
        )

    def _acknowledge_for_cleanup(
        self,
        submission: BackgroundPipelineSubmission[InputT, ResultT],
        cleanup_errors: list[BaseException],
    ) -> None:
        while self.owns(submission):
            try:
                self.acknowledge(submission)
            except BaseException as error:
                cleanup_errors.append(error)

    def _require_accepting(self) -> None:
        if not self._accepting or self._closed:
            raise RuntimeError("background pipeline is closing or already closed")

    def _shutdown_worker(self) -> None:
        with self._condition:
            self._worker_stop = True
            self._condition.notify_all()
        if current_thread() is not self._worker_thread:
            self._worker_thread.join()

    def _shutdown_callbacks(self) -> None:
        with self._condition:
            self._callback_stop = True
            self._condition.notify_all()
        if current_thread() is not self._callback_thread:
            self._callback_thread.join()
