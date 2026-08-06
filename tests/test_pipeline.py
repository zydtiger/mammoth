"""Verify the framework-neutral bounded background pipeline contract."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import pytest

from mammoth.core import BackgroundPipelineError, BoundedBackgroundPipeline


def test_pipeline_validates_configuration() -> None:
    """Only callable workers and positive pending bounds are accepted."""
    with pytest.raises(TypeError, match="worker must be callable"):
        BoundedBackgroundPipeline(42)  # type: ignore[arg-type]
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_pending"):
            BoundedBackgroundPipeline(lambda item: item, max_pending=value)  # type: ignore[arg-type]
    pipeline = BoundedBackgroundPipeline(lambda item: item)
    with pytest.raises(TypeError, match="on_done"):
        pipeline.submit(1, on_done=42)  # type: ignore[arg-type]
    assert pipeline.close() == ()


def test_pipeline_executes_in_order_and_applies_bounded_backpressure() -> None:
    """A second submission waits until the only pending slot is released."""
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    second_finished = threading.Event()
    submissions = []

    def work(value: int) -> int:
        if value == 1:
            first_started.set()
            assert release_first.wait(timeout=5.0)
        else:
            second_started.set()
            assert release_second.wait(timeout=5.0)
        return value * 10

    pipeline = BoundedBackgroundPipeline(work, max_pending=1)
    first = pipeline.submit(1)
    assert first_started.wait(timeout=5.0)

    def submit_second() -> None:
        submissions.append(pipeline.submit(2))
        second_finished.set()

    thread = threading.Thread(target=submit_second)
    thread.start()
    try:
        assert not second_finished.wait(timeout=0.1)
    finally:
        release_first.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert second_started.wait(timeout=5.0)
    assert [item.input for item in pipeline.take_completed()] == [1]
    pipeline.acknowledge(first)
    release_second.set()
    final = pipeline.close()
    assert [(item.input, item.result) for item in final] == [(2, 20)]
    pipeline.acknowledge(submissions[0])
    assert pipeline.close() == ()


def test_pipeline_attributes_results_and_failures_to_submissions() -> None:
    """Typed completions and raised failures retain their accepted inputs."""
    release = threading.Event()

    def work(value: str) -> int:
        assert release.wait(timeout=5.0)
        if value == "broken":
            raise OSError("worker failed")
        return len(value)

    pipeline = BoundedBackgroundPipeline(work, max_pending=2)
    successful = pipeline.submit("good")
    failed = pipeline.submit("broken")
    release.set()

    with pytest.raises(BackgroundPipelineError, match="worker failed") as raised:
        pipeline.flush()

    assert raised.value.submission is failed
    assert raised.value.input == "broken"
    assert isinstance(raised.value.__cause__, OSError)
    pipeline.acknowledge(failed)
    completed = pipeline.take_completed()
    assert completed[0].submission is successful
    pipeline.acknowledge(successful)
    assert pipeline.close() == ()


def test_pipeline_keeps_later_failures_observable() -> None:
    """Flush retries surface every accepted failure with its own input."""
    release = threading.Event()

    def fail(value: str) -> int:
        assert release.wait(timeout=5.0)
        raise OSError(f"{value} failed")

    pipeline = BoundedBackgroundPipeline(fail, max_pending=2)
    first = pipeline.submit("first")
    second = pipeline.submit("second")
    release.set()

    with pytest.raises(BackgroundPipelineError, match="first failed") as first_error:
        pipeline.flush()
    pipeline.acknowledge(first_error.value.submission)
    with pytest.raises(BackgroundPipelineError, match="second failed") as second_error:
        pipeline.flush()
    pipeline.acknowledge(second_error.value.submission)

    assert first_error.value.submission is first
    assert second_error.value.submission is second
    assert pipeline.close() == ()


def test_pipeline_interrupt_before_acceptance_keeps_input_with_caller() -> None:
    """An interruption before queue insertion cannot transfer ownership."""

    class InterruptingDeque(deque[Any]):
        def append(self, value: Any) -> None:
            del value
            raise KeyboardInterrupt("before acceptance")

    pipeline = BoundedBackgroundPipeline(lambda value: value + 1)
    pipeline._submissions = InterruptingDeque()

    with pytest.raises(KeyboardInterrupt, match="before acceptance"):
        pipeline.submit(5)

    assert pipeline.pending_count == 0
    assert pipeline.close() == ()


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt("accepted interrupt"), SystemExit(9)],
)
def test_pipeline_defers_interruption_immediately_after_acceptance(
    interruption: KeyboardInterrupt | SystemExit,
) -> None:
    """An interrupt after queue insertion cannot orphan accepted work."""

    class InterruptingDeque(deque[Any]):
        def append(self, value: Any) -> None:
            super().append(value)
            raise interruption

    pipeline = BoundedBackgroundPipeline(lambda value: value + 1)
    pipeline._submissions = InterruptingDeque()
    submission = pipeline.submit(5)

    assert pipeline.owns(submission)
    assert pipeline.take_deferred_interrupt() is interruption
    completed = pipeline.close()
    assert [(item.input, item.result) for item in completed] == [(5, 6)]
    pipeline.acknowledge(submission)


def test_pipeline_defers_interruption_before_accepted_handoff_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption at the final accepted-handoff edge is deferred."""
    interruption = KeyboardInterrupt("handoff interrupted")
    pipeline = BoundedBackgroundPipeline(lambda value: value + 1)

    def interrupt_handoff(submission: Any) -> Any:
        del submission
        raise interruption

    monkeypatch.setattr(pipeline, "_complete_submission_handoff", interrupt_handoff)
    submission = pipeline.submit(8)

    assert pipeline.owns(submission)
    assert pipeline.take_deferred_interrupt() is interruption
    completed = pipeline.close()
    assert [(item.input, item.result) for item in completed] == [(8, 9)]
    pipeline.acknowledge(submission)


def test_pipeline_reports_ownership_for_the_exact_input_object() -> None:
    """Callers can resolve an ambiguous post-submit interrupt by input identity."""
    first = [1]
    equal_but_distinct = [1]
    pipeline = BoundedBackgroundPipeline(lambda value: value)
    submission = pipeline.submit(first)

    assert pipeline.owns_input(first)
    assert not pipeline.owns_input(equal_but_distinct)
    pipeline.flush()
    assert pipeline.owns_input(first)
    pipeline.acknowledge(submission)
    assert not pipeline.owns_input(first)
    assert pipeline.close() == ()


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt("acknowledgement interrupted"), SystemExit(11)],
)
def test_pipeline_defers_interruption_after_acknowledgement_removes_submission(
    interruption: KeyboardInterrupt | SystemExit,
) -> None:
    """An interrupted removal cannot consume an outcome without a deferred signal."""

    class InterruptingDeque(deque[Any]):
        def remove(self, value: Any) -> None:
            super().remove(value)
            raise interruption

    pipeline = BoundedBackgroundPipeline(lambda value: value + 1)
    submission = pipeline.submit(5)
    assert submission.result(timeout=5.0) == 6
    pipeline._submissions = InterruptingDeque(pipeline._submissions)

    pipeline.acknowledge(submission)

    assert not pipeline.owns(submission)
    assert pipeline.take_deferred_interrupt() is interruption
    assert pipeline.close() == ()


def test_pipeline_blocked_submit_cannot_cross_concurrent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submitter woken by shutdown cannot transfer new work to a stopped worker."""
    started = threading.Event()
    release_worker = threading.Event()
    submission_reached_slot = threading.Event()
    allow_submission_to_resume = threading.Event()

    def work(value: int) -> int:
        started.set()
        assert release_worker.wait(timeout=5.0)
        return value

    pipeline = BoundedBackgroundPipeline(work, max_pending=1)
    first = pipeline.submit(1)
    assert started.wait(timeout=5.0)
    original_wait = pipeline._condition.wait

    def pause_woken_submitter(timeout: float | None = None) -> bool:
        result = original_wait(timeout=timeout)
        if threading.current_thread().name != "blocked-submit":
            return result
        pipeline._condition.release()
        try:
            submission_reached_slot.set()
            assert allow_submission_to_resume.wait(timeout=5.0)
        finally:
            pipeline._condition.acquire()
        return result

    monkeypatch.setattr(pipeline._condition, "wait", pause_woken_submitter)
    errors: list[BaseException] = []

    def submit_second() -> None:
        try:
            pipeline.submit(2)
        except BaseException as error:
            errors.append(error)

    submitter = threading.Thread(target=submit_second, name="blocked-submit")
    submitter.start()
    release_worker.set()
    assert submission_reached_slot.wait(timeout=5.0)
    completed = pipeline.close()
    allow_submission_to_resume.set()
    submitter.join(timeout=5.0)

    assert not submitter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert [item.input for item in completed] == [1]
    assert pipeline.pending_count == 0
    pipeline.acknowledge(first)


def test_pipeline_flush_retry_observes_work_after_wait_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted backpressure leaves accepted work for a later flush."""
    started = threading.Event()
    release = threading.Event()

    def work(value: int) -> int:
        started.set()
        assert release.wait(timeout=5.0)
        return value + 1

    pipeline = BoundedBackgroundPipeline(work)
    submission = pipeline.submit(4)
    assert started.wait(timeout=5.0)
    original_wait = pipeline._condition.wait
    calls = 0

    def interrupt_once(timeout: float | None = None) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("flush wait interrupted")
        return original_wait(timeout=timeout)

    monkeypatch.setattr(pipeline._condition, "wait", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="flush wait interrupted"):
        pipeline.flush()

    assert pipeline.owns(submission)
    release.set()
    completed = pipeline.flush()
    assert [(item.input, item.result) for item in completed] == [(4, 5)]
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_backpressure_wait_retry_preserves_prior_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted slot waiting neither loses prior work nor accepts new work."""
    started = threading.Event()
    release = threading.Event()

    def work(value: int) -> int:
        started.set()
        assert release.wait(timeout=5.0)
        return value

    pipeline = BoundedBackgroundPipeline(work, max_pending=1)
    submission = pipeline.submit(1)
    assert started.wait(timeout=5.0)
    original_wait = pipeline._condition.wait

    def interrupt_wait(timeout: float | None = None) -> bool:
        del timeout
        raise KeyboardInterrupt("backpressure interrupted")

    monkeypatch.setattr(pipeline._condition, "wait", interrupt_wait)
    with pytest.raises(KeyboardInterrupt, match="backpressure interrupted"):
        pipeline.wait_for_submission_slot()

    assert pipeline.owns(submission)
    monkeypatch.setattr(pipeline._condition, "wait", original_wait)
    release.set()
    pipeline.flush()
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_close_retry_retains_unacknowledged_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close interruption retains its result until explicit acknowledgment."""
    pipeline = BoundedBackgroundPipeline(lambda value: value + 1)
    submission = pipeline.submit(4)
    original_flush = pipeline.flush
    calls = 0

    def interrupt_once() -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("close interrupted")
        return original_flush()

    monkeypatch.setattr(pipeline, "flush", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="close interrupted"):
        pipeline.close()

    assert pipeline.owns(submission)
    completed = pipeline.close()
    assert [(item.input, item.result) for item in completed] == [(4, 5)]
    assert pipeline.close() == completed
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


@pytest.mark.parametrize(
    "worker_error",
    [KeyboardInterrupt("worker interrupt"), SystemExit(11)],
)
def test_pipeline_attributes_worker_process_exceptions(
    worker_error: KeyboardInterrupt | SystemExit,
) -> None:
    """Worker process-level failures retain their accepted input identity."""

    def fail(_: str) -> int:
        raise worker_error

    pipeline = BoundedBackgroundPipeline(fail)
    submission = pipeline.submit("failed-input")

    with pytest.raises(BackgroundPipelineError) as raised:
        pipeline.flush()

    assert raised.value.submission is submission
    assert raised.value.cause is worker_error
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_failure_remains_after_interrupted_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A main-thread interruption cannot erase an attributed worker failure."""
    pipeline = BoundedBackgroundPipeline(lambda _: (_ for _ in ()).throw(OSError("failed")))
    submission = pipeline.submit("input")
    original_flush = pipeline.flush
    calls = 0

    def interrupt_once() -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SystemExit(17)
        return original_flush()

    monkeypatch.setattr(pipeline, "flush", interrupt_once)
    with pytest.raises(SystemExit) as interrupted:
        pipeline.flush()
    assert interrupted.value.code == 17

    with pytest.raises(BackgroundPipelineError, match="failed") as raised:
        pipeline.flush()
    assert raised.value.submission is submission
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_does_not_publish_failure_before_future_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure observation waits until submission completion is internally final."""
    worker_started = threading.Event()
    release_worker = threading.Event()
    finalization_started = threading.Event()
    allow_finalization = threading.Event()
    flush_finished = threading.Event()
    errors: list[BaseException] = []

    def fail(_: int) -> int:
        worker_started.set()
        assert release_worker.wait(timeout=5.0)
        raise OSError("worker failed")

    pipeline = BoundedBackgroundPipeline(fail)
    submission = pipeline.submit(1)
    assert worker_started.wait(timeout=5.0)
    real_set_exception = submission._future.set_exception

    def block_finalization(error: BaseException) -> None:
        finalization_started.set()
        assert allow_finalization.wait(timeout=5.0)
        real_set_exception(error)

    monkeypatch.setattr(submission._future, "set_exception", block_finalization)

    def flush() -> None:
        try:
            pipeline.flush()
        except BaseException as error:
            errors.append(error)
        finally:
            flush_finished.set()

    waiter = threading.Thread(target=flush)
    waiter.start()
    release_worker.set()
    assert finalization_started.wait(timeout=5.0)
    assert not flush_finished.wait(timeout=0.1)
    assert not submission.done()
    allow_finalization.set()
    waiter.join(timeout=5.0)

    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], BackgroundPipelineError)
    assert submission.done()
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_result_waits_until_acknowledgment_state_is_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned result can always be acknowledged immediately."""
    release_worker = threading.Event()
    future_finalized = threading.Event()
    allow_state_finalization = threading.Event()
    result_finished = threading.Event()
    pipeline = BoundedBackgroundPipeline(lambda value: release_worker.wait(5.0) or value)
    submission = pipeline.submit(1)
    real_set_result = submission._future.set_result

    def pause_after_future(result: int) -> None:
        real_set_result(result)
        future_finalized.set()
        assert allow_state_finalization.wait(timeout=5.0)

    monkeypatch.setattr(submission._future, "set_result", pause_after_future)
    results: list[int] = []

    def read_result() -> None:
        results.append(submission.result())
        pipeline.acknowledge(submission)
        result_finished.set()

    reader = threading.Thread(target=read_result)
    reader.start()
    release_worker.set()
    assert future_finalized.wait(timeout=5.0)
    assert not result_finished.wait(timeout=0.1)
    allow_state_finalization.set()
    reader.join(timeout=5.0)

    assert results == [1]
    assert not pipeline.owns(submission)
    assert pipeline.close() == ()


def test_pipeline_submission_exposes_read_only_completion() -> None:
    """Callers can wait for outcomes without mutating pipeline-owned state."""
    pipeline = BoundedBackgroundPipeline(lambda value: value * 2)
    submission = pipeline.submit(3)

    assert submission.result(timeout=5.0) == 6
    assert submission.done()
    assert not hasattr(submission, "future")
    assert not hasattr(submission, "cancel")
    assert not hasattr(submission, "set_result")
    pipeline.flush()
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_done_callback_runs_after_final_state_and_cannot_corrupt_it() -> None:
    """Callback re-entry sees completion and callback failure stays isolated."""
    callback_results: list[tuple[int, ...]] = []
    callback_finished = threading.Event()
    pipeline = BoundedBackgroundPipeline(lambda value: value * 2)

    def inspect_and_interrupt(_: Any) -> None:
        callback_results.append(tuple(item.result for item in pipeline.flush()))
        callback_finished.set()
        raise KeyboardInterrupt("callback interrupted")

    submission = pipeline.submit(3, on_done=inspect_and_interrupt)

    assert submission.result(timeout=5.0) == 6
    assert callback_finished.wait(timeout=5.0)
    assert callback_results == [(6,)]
    completed = pipeline.flush()
    assert [item.result for item in completed] == [6]
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_failed_acknowledgment_defers_interruption_after_removal() -> None:
    """A failed outcome stays attributable when acknowledgment removal is interrupted."""

    class InterruptingDeque(deque[Any]):
        def remove(self, value: Any) -> None:
            super().remove(value)
            raise KeyboardInterrupt("acknowledgment interrupted")

    pipeline = BoundedBackgroundPipeline(
        lambda _: (_ for _ in ()).throw(OSError("publication failed"))
    )
    submission = pipeline.submit(6)
    with pytest.raises(BackgroundPipelineError, match="publication failed") as raised:
        pipeline.flush()
    pipeline._submissions = InterruptingDeque(pipeline._submissions)

    pipeline.acknowledge(raised.value.submission)

    assert raised.value.input == 6
    assert isinstance(raised.value.cause, OSError)
    deferred = pipeline.take_deferred_interrupt()
    assert isinstance(deferred, KeyboardInterrupt)
    assert "acknowledgment interrupted" in str(deferred)
    pipeline.acknowledge(submission)
    assert not pipeline.owns(submission)
    assert pipeline.close() == ()


def test_pipeline_failure_attribution_is_read_only() -> None:
    """Callers cannot replace an attributed submission or its original cause."""
    pipeline = BoundedBackgroundPipeline(lambda _: (_ for _ in ()).throw(OSError("failed")))
    submission = pipeline.submit("input")

    with pytest.raises(BackgroundPipelineError) as raised:
        pipeline.flush()

    with pytest.raises(AttributeError):
        raised.value.submission = submission
    with pytest.raises(AttributeError):
        raised.value.cause = RuntimeError("replacement")
    assert raised.value.input == "input"
    assert isinstance(raised.value.cause, OSError)
    pipeline.acknowledge(submission)
    assert pipeline.close() == ()


def test_pipeline_context_keeps_workload_error_primary() -> None:
    """Every worker cleanup failure is attached without replacing caller failure."""
    workload_error = RuntimeError("caller failed")

    def fail(value: int) -> int:
        raise OSError(f"background {value} failed")

    with (
        pytest.raises(RuntimeError, match="caller failed") as raised,
        BoundedBackgroundPipeline(fail, max_pending=2) as pipeline,
    ):
        pipeline.submit(1)
        pipeline.submit(2)
        raise workload_error

    assert raised.value is workload_error
    notes = "\n".join(workload_error.__notes__)
    assert "background 1 failed" in notes
    assert "background 2 failed" in notes
