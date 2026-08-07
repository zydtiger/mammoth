from __future__ import annotations

import json
from pathlib import Path

import pytest

from mammoth.core import RunLayout, create_execution_context
from mammoth.core.events import (
    ExecutionEvent,
    ExecutionEventReadError,
    ExecutionEventTailReader,
    ExecutionEventWriter,
    read_execution_events,
)


def execution_context(tmp_path: Path):
    layout = RunLayout(tmp_path, "event-run").prepare()
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("work",),
        world_size=1,
        execution_mode="single",
        command=("python", "worker.py"),
        execution_id="attempt",
    )


def test_process_writer_round_trips_lifecycle_progress_and_coordinates(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        writer.emit("execution_started")
        writer.emit("process_started", phase="work")
        writer.emit("task_started", phase="work", task_id="item")
        writer.emit_progress(
            phase="work",
            task_id="item",
            completed=1,
            total=2,
            coordinates={"epoch": 0, "split": "train"},
            display_metrics={"loss": 0.5},
        )
        writer.emit("task_completed", phase="work", task_id="item")
        writer.emit("process_completed", phase="work", exit_code=0)
        writer.emit("execution_completed")

    events = read_execution_events(context.execution_dir / "rank-0.jsonl")

    assert [event.sequence for event in events] == list(range(1, 8))
    progress = events[3]
    assert progress.coordinates == {"epoch": 0, "split": "train"}
    assert progress.display_metrics == {"loss": 0.5}
    assert "unit" not in progress.to_dict()
    assert events[-1].is_terminal


def test_progress_is_replaceable_and_terminal_event_flushes_latest(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    clock_values = iter((0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    writer = ExecutionEventWriter.for_process(
        context,
        rank=0,
        progress_interval_seconds=1.0,
        monotonic_clock=lambda: next(clock_values),
    )
    first = writer.emit_progress(phase="work", task_id="item", completed=1, total=3)
    pending = writer.emit_progress(phase="work", task_id="item", completed=2, total=3)
    writer.emit("task_completed", phase="work", task_id="item")
    writer.close()

    events = read_execution_events(context.execution_dir / "rank-0.jsonl")
    assert first is not None
    assert pending is None
    assert [(event.event, event.completed) for event in events] == [
        ("progress", 1),
        ("progress", 2),
        ("task_completed", None),
    ]


def test_new_progress_writers_reject_retired_unit_keyword(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    with ExecutionEventWriter.for_process(context, rank=0) as writer:
        with pytest.raises(TypeError, match="unit is no longer supported"):
            writer.emit_progress(
                phase="work",
                task_id="item",
                completed=1,
                unit="items",
            )
        with pytest.raises(TypeError, match="unit is no longer supported"):
            writer.emit(
                "progress",
                phase="work",
                task_id="item",
                completed=1,
                unit="items",
            )


def test_legacy_progress_unit_is_readable_but_not_reserialized() -> None:
    payload = {
        "schema_version": 1,
        "sequence": 1,
        "time": "2026-01-01T00:00:00Z",
        "execution_id": "attempt",
        "run_name": "run",
        "source": "process",
        "event": "progress",
        "rank": 0,
        "world_size": 1,
        "phase": "segment",
        "task_id": "slide",
        "completed": 1,
        "total": 2,
        "unit": "patches",
        "epoch": 4,
        "slide_id": "case-1",
        "scheduling_mode": "dynamic",
    }

    event = ExecutionEvent.from_dict(payload)

    assert event.extensions["epoch"] == 4
    assert event.extensions["slide_id"] == "case-1"
    serialized = event.to_dict()
    assert serialized["scheduling_mode"] == "dynamic"
    assert "unit" not in serialized


def test_tail_reader_holds_partial_record_and_detects_prefix_mutation(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    stream = context.execution_dir / "rank-0.jsonl"
    writer = ExecutionEventWriter.for_process(context, rank=0)
    writer.emit("execution_started")
    writer.close()
    complete = stream.read_bytes()
    stream.write_bytes(complete + b'{"schema_version":1')
    reader = ExecutionEventTailReader(stream)

    assert [event.event for event in reader.poll()] == ["execution_started"]
    assert reader.poll() == []
    stream.write_bytes(b"X" + stream.read_bytes()[1:])
    with pytest.raises(ExecutionEventReadError, match="prefix changed"):
        reader.poll()


def test_malformed_complete_record_has_precise_line_context(tmp_path: Path) -> None:
    path = tmp_path / "rank-0.jsonl"
    path.write_bytes(b"{}\n")

    with pytest.raises(ExecutionEventReadError) as captured:
        read_execution_events(path)

    assert captured.value.path == path
    assert captured.value.line_number == 1


def test_writer_sanitizes_extensions_and_disables_on_open_failure(tmp_path: Path) -> None:
    context = execution_context(tmp_path)
    writer = ExecutionEventWriter.for_process(context, rank=0)
    writer.emit(
        "heartbeat",
        phase="work",
        force=True,
        api_token="secret",
        endpoint="https://user:pass@example.test/path?token=x",
    )
    writer.close()
    payload = json.loads((context.execution_dir / "rank-0.jsonl").read_text())
    assert payload["api_token"] == "<redacted>"
    assert "secret" not in json.dumps(payload)
    assert "pass" not in json.dumps(payload)

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file")
    disabled = ExecutionEventWriter(
        blocked_parent / "rank-0.jsonl",
        execution_id="attempt",
        run_name="run",
        source="process",
        rank=0,
        world_size=1,
    )
    assert not disabled.enabled
    assert disabled.emit("execution_started") is None
