"""Coverage for framework-neutral direct execution session composition."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import mammoth.execution as execution_module
from mammoth.core import (
    BackgroundPipelineError,
    claim_logical_run_lease,
    create_execution_context,
    read_execution_events,
)
from mammoth.execution import (
    NULL_EXECUTION_OBSERVER,
    ExecutionObserver,
    ExecutionSession,
    ExecutionSpec,
    SessionExecutionObserver,
)
from mammoth.logging import Observation


class RecordingSink:
    """Record neutral observer ownership without adding a framework dependency."""

    def __init__(self, order: list[str], *, fail_close: bool = False) -> None:
        self.order = order
        self.fail_close = fail_close

    def observe(self, observation: Observation) -> None:
        del observation

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.order.append("observer")
        if self.fail_close:
            raise OSError("observer cleanup failed")


def spec_for(tmp_path: Path, name: str = "neutral-run") -> ExecutionSpec:
    """Build a stable direct-session specification for one temporary run."""
    return ExecutionSpec(
        run_dir=tmp_path / name,
        run_name=name,
        invocation_kind="unit-test",
        intended_phases=("train", "validate"),
        execution_id=f"{name}-attempt",
        runtime={"caller": "neutral-test"},
    )


def test_neutral_api_imports_and_constructs_when_torch_is_unavailable(tmp_path: Path) -> None:
    """The direct session public path must not import optional PyTorch."""
    fake_torch = tmp_path / "torch.py"
    fake_torch.write_text("raise RuntimeError('torch must not be imported')\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    run_dir = tmp_path / "subprocess-run"
    script = f"""
from pathlib import Path
from mammoth.execution import ExecutionSession, ExecutionSpec

with ExecutionSession.create(
    ExecutionSpec(Path({str(run_dir)!r}), "subprocess-run", "test", ("work",))
) as session:
    with session.phase_scope("work"):
        pass
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (run_dir / "logs" / "executions").is_dir()


def test_direct_create_records_lifecycle_and_releases_lease(tmp_path: Path) -> None:
    """A direct single-process session owns create, logging, lifecycle, and lease."""
    spec = spec_for(tmp_path)
    with ExecutionSession.create(spec) as session, session.phase_scope("train"):
        pass

    events = read_execution_events(
        spec.run_dir / "logs" / "executions" / "neutral-run-attempt" / "rank-0.jsonl"
    )
    assert [event.event for event in events] == [
        "process_started",
        "phase_started",
        "phase_completed",
        "process_completed",
    ]
    assert events[-1].exit_code == 0
    with claim_logical_run_lease(spec.run_dir):
        pass


def test_direct_attach_strictly_validates_workflow_child_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct workflow children attach without leasing and reject a join mismatch."""
    spec = spec_for(tmp_path, "attached-run")
    context = create_execution_context(
        spec.run_dir,
        run_name=spec.run_name,
        invocation_kind=spec.invocation_kind,
        intended_phases=spec.intended_phases,
        world_size=1,
        execution_mode="single",
        command=("python", "workflow.py"),
        execution_id="attached-run-attempt",
        runtime={"caller": "neutral-test"},
    )
    monkeypatch.setenv("MAMMOTH_EXECUTION_ID", context.metadata.execution_id)
    monkeypatch.setenv("MAMMOTH_RUN_NAME", spec.run_name)
    monkeypatch.setenv("MAMMOTH_INVOCATION_KIND", spec.invocation_kind)
    monkeypatch.setenv("MAMMOTH_PHASE", "train")

    with ExecutionSession.attach(spec) as session:
        assert session.context == context
        with claim_logical_run_lease(spec.run_dir):
            pass

    monkeypatch.setenv("MAMMOTH_PHASE", "other")
    with pytest.raises(ValueError, match="MAMMOTH_PHASE"):
        ExecutionSession.attach(spec)


@pytest.mark.parametrize(
    ("action", "expected_phase_event", "expected_exit_code"),
    [
        ("skip", "phase_skipped", 0),
        ("fail", "phase_failed", 1),
        ("interrupt", "phase_failed", 130),
    ],
)
def test_neutral_session_records_skip_failure_and_interruption(
    tmp_path: Path,
    action: str,
    expected_phase_event: str,
    expected_exit_code: int,
) -> None:
    """Terminal lifecycle states remain framework-neutral and deterministic."""
    spec = spec_for(tmp_path, f"{action}-run")
    session = ExecutionSession.create(spec)
    session.start_phase("validate")
    if action == "skip":
        session.skip_phase("not requested")
        session.close()
    elif action == "fail":
        failure = RuntimeError("validation failed")
        session.fail_phase(failure)
        session.close(error=failure)
    else:
        interruption = KeyboardInterrupt("validation interrupted")
        session.fail_phase(interruption, interrupted=True)
        session.close(error=interruption)

    events = read_execution_events(
        spec.run_dir / "logs" / "executions" / f"{action}-run-attempt" / "rank-0.jsonl"
    )
    assert events[-2].event == expected_phase_event
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == expected_exit_code
    if action == "interrupt":
        assert events[-1].signal == 2


def test_neutral_session_cleanup_is_ordered_idempotent_and_preserves_failure(
    tmp_path: Path,
) -> None:
    """Pipelines, observers, logging, and lease finalizers close once in order."""
    spec = spec_for(tmp_path, "cleanup-run")
    session = ExecutionSession.create(spec)
    order: list[str] = []
    original_logging_close = session.execution_logging.close
    original_lease_close = session._close_callbacks[0][1]

    def close_logging() -> None:
        order.append("logging")
        original_logging_close()

    def close_lease() -> None:
        order.append("lease")
        original_lease_close()

    session.execution_logging.close = close_logging
    session._close_callbacks = (("logical-run lease", close_lease),)
    session.create_observer((RecordingSink(order),))
    pipeline = session.create_background_pipeline(lambda value: order.append(value) or value)
    pipeline.submit("pipeline")
    session.start_phase("train")
    session.complete_phase()
    session.close()
    session.close()

    assert order == ["pipeline", "observer", "logging", "lease"]


def test_neutral_create_logging_failure_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logging startup failure cannot retain direct producer ownership."""
    spec = spec_for(tmp_path, "logging-failure")

    def fail_logging(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("cannot open rank log")

    monkeypatch.setattr(execution_module, "create_execution_logging", fail_logging)
    with pytest.raises(OSError, match="cannot open rank log"):
        ExecutionSession.create(spec)
    with claim_logical_run_lease(spec.run_dir):
        pass


def test_neutral_cleanup_failure_records_terminal_outcome(tmp_path: Path) -> None:
    """A pipeline cleanup error wins only when no workload exception is active."""
    spec = spec_for(tmp_path, "cleanup-failure")
    session = ExecutionSession.create(spec)
    session.start_phase("train")

    def fail_pipeline(_: str) -> str:
        raise OSError("flush failed")

    pipeline = session.create_background_pipeline(fail_pipeline)
    pipeline.submit("artifact")

    with pytest.raises(BackgroundPipelineError, match="flush failed"):
        session.close()

    events = read_execution_events(
        spec.run_dir / "logs" / "executions" / "cleanup-failure-attempt" / "rank-0.jsonl"
    )
    assert events[-2].event == "phase_failed"
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == 1


def test_null_execution_observer_never_creates_files_or_events(tmp_path: Path) -> None:
    """The detached no-op observer performs no filesystem or event side effects."""
    before = sorted(tmp_path.rglob("*"))

    NULL_EXECUTION_OBSERVER.start_phase("prepare")
    with NULL_EXECUTION_OBSERVER.task("unit"):
        NULL_EXECUTION_OBSERVER.progress("unit", completed=1, total=2)
    NULL_EXECUTION_OBSERVER.record_count("unit", completed=3)
    with NULL_EXECUTION_OBSERVER.heartbeats(message="idle"):
        pass
    NULL_EXECUTION_OBSERVER.complete_phase()
    NULL_EXECUTION_OBSERVER.skip_phase("not requested")

    assert sorted(tmp_path.rglob("*")) == before
    assert isinstance(NULL_EXECUTION_OBSERVER, ExecutionObserver)


def test_session_execution_observer_forwards_onto_a_live_session(tmp_path: Path) -> None:
    """The session-backed observer records the same events as direct session calls."""
    spec = spec_for(tmp_path, "observer-run")
    with ExecutionSession.create(spec) as session:
        observer = SessionExecutionObserver(session)
        observer.start_phase("train")
        with observer.task("unit-a"):
            observer.progress("unit-a", completed=1, total=2)
            observer.progress("unit-a", completed=2, total=2, final=True)
        observer.record_count("unit-b", completed=5)
        with observer.heartbeats(message="working"):
            pass
        observer.complete_phase()

    events = read_execution_events(
        spec.run_dir / "logs" / "executions" / "observer-run-attempt" / "rank-0.jsonl"
    )
    assert [event.event for event in events] == [
        "process_started",
        "phase_started",
        "task_started",
        "progress",
        "progress",
        "task_completed",
        "task_started",
        "progress",
        "task_completed",
        "phase_completed",
        "process_completed",
    ]


def test_session_execution_observer_requires_an_active_phase(tmp_path: Path) -> None:
    """Forwarding a task/progress/heartbeat call before any phase starts fails clearly."""
    spec = spec_for(tmp_path, "observer-no-phase")
    with ExecutionSession.create(spec) as session:
        observer = SessionExecutionObserver(session)
        with pytest.raises(RuntimeError, match="active phase"):
            observer.progress("unit", completed=1)
