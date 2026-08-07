from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from mammoth.core import RunLayout, create_execution_context, read_execution_events
from mammoth.core.events import ExecutionEventWriter
from mammoth.logging import (
    JsonlEventSink,
    Media,
    Observation,
    RunObserver,
    claim_process_text_log,
    create_execution_logging,
    create_execution_observability,
)
from mammoth.logging.tensorboard import NullSummaryWriter, TensorBoardSink
from mammoth.logging.text import create_process_text_handler


def execution_context(tmp_path: Path):
    layout = RunLayout(tmp_path, "logging-run").prepare()
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("phase",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
        execution_id="attempt",
    )


class RecordingSink:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.flushed = 0
        self.closed = 0
        self.observed = threading.Event()

    def observe(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.observed.set()

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


class FailingSink(RecordingSink):
    def observe(self, observation: Observation) -> None:
        raise OSError("backend unavailable")


class BlockingSerialSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self._active = threading.Lock()
        self.heartbeat_entered = threading.Event()
        self.release_heartbeat = threading.Event()

    def observe(self, observation: Observation) -> None:
        if not self._active.acquire(blocking=False):
            raise RuntimeError("concurrent sink access")
        try:
            super().observe(observation)
            if observation.event == "heartbeat":
                self.heartbeat_entered.set()
                assert self.release_heartbeat.wait(timeout=1.0)
        finally:
            self._active.release()


class FakeSummaryWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.images: list[tuple[str, Any, int, dict[str, Any]]] = []
        self.text: list[tuple[str, str, int, dict[str, Any]]] = []
        self.histograms: list[tuple[str, Any, int, dict[str, Any]]] = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def add_image(self, tag: str, value: Any, global_step: int, **kwargs: Any) -> None:
        self.images.append((tag, value, global_step, kwargs))

    def add_text(self, tag: str, value: str, global_step: int, **kwargs: Any) -> None:
        self.text.append((tag, value, global_step, kwargs))

    def add_histogram(self, tag: str, value: Any, global_step: int, **kwargs: Any) -> None:
        self.histograms.append((tag, value, global_step, kwargs))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


def test_progress_fans_out_distinct_jsonl_and_dense_history(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    jsonl = JsonlEventSink(ExecutionEventWriter.for_process(context, rank=0))
    recording = RecordingSink()
    with RunObserver((jsonl, recording)) as observer:
        observer.progress(
            phase="phase",
            task_id="batch",
            completed=1,
            total=2,
            metrics={"train/loss": 0.25, "train/lr": 0.001},
            display_metrics={"train/loss": 0.25},
            coordinates={"global_step": 7},
        )

    event = read_execution_events(context.execution_dir / "rank-0.jsonl")[0]
    assert event.display_metrics == {"train/loss": 0.25}
    assert "metrics" not in event.to_dict()
    assert "unit" not in event.to_dict()
    assert recording.observations[0].metrics == {
        "train/loss": 0.25,
        "train/lr": 0.001,
    }


def test_observer_isolates_failed_sink_and_keeps_healthy_sinks() -> None:
    failing = FailingSink()
    recording = RecordingSink()
    observer = RunObserver((failing, recording))

    observer.emit("execution_started")
    observer.emit("execution_completed")

    assert observer.disabled_sink_count == 1
    assert [item.event for item in recording.observations] == [
        "execution_started",
        "execution_completed",
    ]
    assert failing.closed == 1


def test_observer_rejects_retired_unit_without_disabling_jsonl_sink(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    jsonl = JsonlEventSink(ExecutionEventWriter.for_process(context, rank=0))
    observer = RunObserver((jsonl,))

    with pytest.raises(TypeError, match="unit is no longer supported"):
        observer.emit(
            "progress",
            phase="phase",
            task_id="task",
            completed=1,
            unit="items",
        )
    observer.emit("progress", phase="phase", task_id="task", completed=1, total=2)
    observer.close()

    assert observer.disabled_sink_count == 0
    events = read_execution_events(context.execution_dir / "rank-0.jsonl")
    assert [event.completed for event in events] == [1]


def test_phase_and_task_contexts_emit_balanced_terminal_records() -> None:
    recording = RecordingSink()
    observer = RunObserver((recording,))
    with observer.phase("phase"), observer.task("phase", "task"):
        pass
    with pytest.raises(RuntimeError, match="boom"), observer.task("phase", "failed"):
        raise RuntimeError("boom")

    assert [item.event for item in recording.observations] == [
        "phase_started",
        "task_started",
        "task_completed",
        "phase_completed",
        "task_started",
        "task_failed",
    ]


def test_observer_manages_periodic_heartbeats_while_idle() -> None:
    recording = RecordingSink()
    observer = RunObserver((recording,), heartbeat_interval_seconds=0.01)

    with observer.periodic_heartbeats(phase="phase", task_id="task", message="working"):
        assert recording.observed.wait(timeout=1.0)
    observer.close()

    heartbeat = recording.observations[0]
    assert heartbeat.event == "heartbeat"
    assert heartbeat.fields == {
        "phase": "phase",
        "force": False,
        "task_id": "task",
        "message": "working",
    }


def test_periodic_heartbeat_serializes_sink_fanout_with_foreground_emit() -> None:
    sink = BlockingSerialSink()
    observer = RunObserver((sink,), heartbeat_interval_seconds=0.01)
    foreground_done = threading.Event()

    def emit_foreground() -> None:
        observer.emit("phase_started", phase="phase")
        foreground_done.set()

    with observer.periodic_heartbeats(phase="phase"):
        assert sink.heartbeat_entered.wait(timeout=1.0)
        foreground = threading.Thread(target=emit_foreground)
        foreground.start()
        assert not foreground_done.wait(timeout=0.05)
        sink.release_heartbeat.set()
        assert foreground_done.wait(timeout=1.0)
        foreground.join()
    observer.close()

    assert observer.disabled_sink_count == 0
    assert [item.event for item in sink.observations] == ["heartbeat", "phase_started"]


def test_close_stops_active_periodic_heartbeat_threads() -> None:
    observer = RunObserver(heartbeat_interval_seconds=1.0)
    existing_threads = set(threading.enumerate())
    heartbeat_scope = observer.periodic_heartbeats(phase="phase")
    heartbeat_scope.__enter__()
    heartbeat_thread = next(
        thread
        for thread in threading.enumerate()
        if thread not in existing_threads and thread.name == "mammoth-observer-heartbeat"
    )

    observer.close()

    assert not heartbeat_thread.is_alive()
    heartbeat_scope.__exit__(None, None, None)


def test_tensorboard_sink_uses_project_metric_names_and_logical_step(tmp_path: Path) -> None:
    writer = FakeSummaryWriter()
    sink = TensorBoardSink(tmp_path, writer=writer)
    observation = Observation(
        event="progress",
        fields={"coordinates": {"global_step": 12}},
        metrics={"custom/loss": 0.5},
        media={
            "sample": Media("image", [[1]], {"dataformats": "HW"}),
            "note": Media("text", "hello"),
        },
    )

    sink.observe(observation)
    sink.flush()
    sink.close()

    assert writer.scalars == [("custom/loss", 0.5, 12)]
    assert writer.images == [("sample", [[1]], 12, {"dataformats": "HW"})]
    assert writer.text == [("note", "hello", 12, {})]
    assert writer.flush_count == 1
    assert writer.close_count == 1


def test_tensorboard_sink_prefers_explicit_logical_step(tmp_path: Path) -> None:
    writer = FakeSummaryWriter()
    sink = TensorBoardSink(tmp_path, writer=writer)

    sink.observe(
        Observation(
            event="progress",
            fields={"coordinates": {"global_step": 12}},
            metrics={"custom/loss": 0.5},
            logical_step=37,
        )
    )

    assert writer.scalars == [("custom/loss", 0.5, 37)]


def test_tensorboard_sink_is_noop_on_secondary_rank(tmp_path: Path) -> None:
    sink = TensorBoardSink(tmp_path, rank=1, primary_rank=0)
    assert isinstance(sink.writer, NullSummaryWriter)
    sink.observe(Observation(event="progress", metrics={"loss": 1.0}))
    assert not any(tmp_path.iterdir())


def test_process_text_handler_writes_plain_diagnostics(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    handler = create_process_text_handler(context, rank=0)
    named_logger = logging.getLogger("mammoth-test-text")
    named_logger.setLevel(logging.INFO)
    named_logger.propagate = False
    named_logger.addHandler(handler)
    try:
        named_logger.info("plain diagnostic")
    finally:
        named_logger.removeHandler(handler)
        handler.close()

    contents = context.rank_log_path(0).read_text()
    assert "plain diagnostic" in contents
    assert "\x1b[" not in contents


def test_process_text_log_lease_rejects_concurrent_owner(tmp_path: Path) -> None:
    path = tmp_path / "rank-0.log"
    with claim_process_text_log(path), pytest.raises(RuntimeError, match="already owned"):
        claim_process_text_log(path)
    with claim_process_text_log(path):
        pass
    assert path.is_file()


def test_process_text_handler_close_is_idempotent(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    handler = create_process_text_handler(context, rank=0)

    handler.close()
    handler.close()

    with claim_process_text_log(context.rank_log_path(0)):
        pass


def test_execution_logging_composes_jsonl_text_and_additional_sinks(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    recording = RecordingSink()
    bundle = create_execution_logging(context, rank=0, additional_sinks=(recording,))
    named_logger = logging.getLogger("mammoth-test-execution-bundle")
    named_logger.setLevel(logging.INFO)
    named_logger.propagate = False
    named_logger.addHandler(bundle.text_handler)
    try:
        bundle.observer.emit("process_started", phase="phase")
        named_logger.info("bundle diagnostic")
    finally:
        named_logger.removeHandler(bundle.text_handler)
        bundle.close()

    events = read_execution_events(context.execution_dir / "rank-0.jsonl")
    assert [event.event for event in events] == ["process_started"]
    assert [observation.event for observation in recording.observations] == ["process_started"]
    assert "bundle diagnostic" in context.rank_log_path(0).read_text()


def test_execution_observability_does_not_claim_text_log(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    recording = RecordingSink()

    with create_execution_observability(
        context,
        rank=0,
        additional_sinks=(recording,),
    ) as observability:
        observability.observer.emit("process_started", phase="phase")
        assert not context.rank_log_path(0).exists()

    events = read_execution_events(context.execution_dir / "rank-0.jsonl")
    assert [event.event for event in events] == ["process_started"]
    assert [observation.event for observation in recording.observations] == ["process_started"]
