"""Exercise the public programmatic workflow API and subprocess supervision."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import UserList
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import mammoth.workflow.launch as workflow_launch
import mammoth.workflow.runner as workflow_runner
from mammoth.core import (
    GroupLayout,
    claim_logical_run_lease,
    join_execution_context,
    load_group_manifest,
    read_execution_events,
    read_group_events,
)
from mammoth.workflow import (
    Execution,
    ProcessResult,
    Run,
    Step,
    SupervisedProcess,
    Workflow,
    WorkflowResult,
    run_captured_process,
)
from mammoth.workflow.launch import launch_process


def successful_step(name: str, *, marker: Path | None = None) -> Step:
    """Return one small successful child command for workflow tests."""
    source = (
        "pass" if marker is None else f"from pathlib import Path; Path({str(marker)!r}).touch()"
    )
    return Step(name, (sys.executable, "-c", source))


def execution_events(root: Path, run_name: str, execution_id: str):
    """Read one workflow runner stream by stable identity."""
    return read_execution_events(
        root / run_name / "logs" / "executions" / execution_id / "runner.jsonl"
    )


def test_models_freeze_caller_inputs_and_step_has_only_final_argv_policy(
    tmp_path: Path,
) -> None:
    command = UserList(["python", "job.py"])
    environment = {"MODE": "train"}
    runtime = {"dataset": {"names": UserList(["sample"])}}
    step = Step("validation fold 1", command, environment=environment)
    execution = Execution(runtime=runtime)
    command.append("--changed")
    environment["MODE"] = "changed"
    runtime["dataset"]["names"].append("changed")

    assert step.command == ("python", "job.py")
    assert step.environment == {"MODE": "train"}
    assert execution.runtime["dataset"]["names"] == ("sample",)
    with pytest.raises(TypeError):
        step.environment["NEW"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        execution.runtime["dataset"]["names"] += ("other",)  # type: ignore[index,operator]
    assert {item.name for item in fields(Step)} == {
        "name",
        "command",
        "cwd",
        "timeout_seconds",
        "environment",
    }
    assert Step("default-environment", ("true",)).environment == {}
    assert not tmp_path.exists() or tuple(tmp_path.iterdir()) == ()


def test_workflow_freezes_caller_collections_and_canonical_dispatch(tmp_path: Path) -> None:
    """Workflow construction detaches its ordering inputs before planning or execution."""
    root = tmp_path / "runs"
    runs = UserList([Run("run", (Step("prepare", ("prepare",)), Step("train", ("train",))))])
    step_order = UserList(["prepare", "train"])

    workflow = Workflow(
        root=root,
        runs=runs,
        order="step-major",
        step_order=step_order,
    )
    runs.clear()
    step_order.reverse()

    assert workflow.runs[0].name == "run"
    assert workflow.step_order == ("prepare", "train")
    assert [(plan.run_name, plan.step_name) for plan in workflow.plan()] == [
        ("run", "prepare"),
        ("run", "train"),
    ]
    assert not root.exists()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"starting_epoch": 1},
        {"resume_checkpoint_sha256": "a" * 64},
        {"starting_global_step": 2},
    ),
)
def test_execution_rejects_resume_facts_without_checkpoint(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="require resume_checkpoint"):
        Execution(**kwargs)


def test_plan_derives_mixed_roots_and_run_major_order_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_root = tmp_path / "default"
    override_root = tmp_path / "override"
    workflow = Workflow(
        root=default_root,
        runs=(
            Run("alpha", (Step("prepare", ("prepare-a",)), Step("train", ("train-a",)))),
            Run(
                "beta",
                (Step("prepare", ("prepare-b",)), Step("evaluate", ("eval-b",))),
                root=override_root,
            ),
        ),
    )

    monkeypatch.setattr(
        workflow_runner.RunLayout,
        "prepare",
        lambda self: pytest.fail("planning must not prepare layouts"),
    )
    plans = workflow.plan()

    assert [(plan.run_name, plan.step_name) for plan in plans] == [
        ("alpha", "prepare"),
        ("alpha", "train"),
        ("beta", "prepare"),
        ("beta", "evaluate"),
    ]
    assert plans[0].run_dir == default_root / "alpha"
    assert plans[2].run_dir == override_root / "beta"
    assert not default_root.exists()
    assert not override_root.exists()


def test_step_major_order_uses_canonical_names_and_allows_missing_steps(tmp_path: Path) -> None:
    workflow = Workflow(
        root=tmp_path,
        order="step-major",
        step_order=("prepare", "train", "evaluate"),
        runs=(
            Run("alpha", (Step("prepare", ("a-p",)), Step("train", ("a-t",)))),
            Run("beta", (Step("prepare", ("b-p",)), Step("evaluate", ("b-e",)))),
        ),
    )

    assert [(plan.run_name, plan.step_name) for plan in workflow.plan()] == [
        ("alpha", "prepare"),
        ("beta", "prepare"),
        ("alpha", "train"),
        ("beta", "evaluate"),
    ]


@pytest.mark.parametrize(
    ("build", "message"),
    (
        (
            lambda: Workflow(
                runs=(Run("run", (Step("train", ("x",)), Step("prepare", ("y",)))),),
                order="step-major",
                step_order=("prepare", "train"),
            ),
            "subsequence",
        ),
        (
            lambda: Workflow(
                runs=(Run("run", (Step("prepare", ("x",)), Step("train", ("y",)))),),
                order="step-major",
                step_order=("prepare",),
            ),
            "absent from step_order",
        ),
        (
            lambda: Workflow(
                runs=(Run("run", (Step("prepare", ("x",)),)),),
                order="step-major",
                step_order=("prepare", "prepare"),
            ),
            "must not contain duplicates",
        ),
        (
            lambda: Workflow(
                runs=(Run("run", (Step("prepare", ("x",)),)),),
                step_order=("prepare",),
            ),
            "must omit step_order",
        ),
        (
            lambda: Workflow(
                runs=(Run("run", (Step("prepare", ("x",)),)),),
                order="step-major",
            ),
            "require a non-empty step_order",
        ),
    ),
)
def test_construction_rejects_invalid_order_contract(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_construction_rejects_duplicate_runs_and_run_rejects_duplicate_steps() -> None:
    first = Run("duplicate", (Step("step", ("one",)),))
    second = Run("duplicate", (Step("other", ("two",)),))
    with pytest.raises(ValueError, match="run names must be unique"):
        Workflow(runs=(first, second))
    with pytest.raises(ValueError, match="step names must be unique"):
        Run("run", (Step("same", ("one",)), Step("same", ("two",))))


def test_plan_and_run_reuse_the_construction_derived_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning and execution consume one immutable dispatch without re-derivation."""
    workflow = Workflow(
        root=tmp_path,
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run("alpha", (Step("prepare", ("a-p",)), Step("train", ("a-t",)))),
            Run("beta", (Step("prepare", ("b-p",)),)),
        ),
    )
    expected = (("alpha", "prepare"), ("beta", "prepare"), ("alpha", "train"))

    monkeypatch.setattr(
        workflow_runner,
        "_derive_dispatch",
        lambda _workflow: pytest.fail("dispatch must only be derived during construction"),
    )
    assert tuple((plan.run_name, plan.step_name) for plan in workflow.plan()) == expected
    monkeypatch.setattr(
        Workflow,
        "plan",
        lambda _workflow: pytest.fail("run must not re-plan the workflow"),
    )
    monkeypatch.setattr(
        workflow_runner,
        "launch_process",
        lambda *args, **kwargs: ProcessResult(0, 0.0),
    )

    result = workflow.run(base_environment={})

    assert tuple((item.run_name, item.step.name) for item in result.dispatch) == expected


def test_resolver_and_hook_observe_lease_bound_lifecycle_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    order: list[str] = []
    original_create = workflow_runner.create_execution_context

    def resolve(context: Any) -> Execution:
        order.append("resolve")
        assert context.layout.run_dir.is_dir()
        with pytest.raises(RuntimeError, match="already active"):
            claim_logical_run_lease(context.layout.run_dir)
        assert list(context.layout.executions_dir.iterdir()) == []
        return Execution(invocation_kind="resolved-workflow", runtime={"source": "resolver"})

    def record_create(*args: Any, **kwargs: Any) -> Any:
        order.append("metadata")
        return original_create(*args, **kwargs)

    def before_first_step(context: Any) -> None:
        order.append("hook")
        events = execution_events(root, "ordered", context.execution.metadata.execution_id)
        assert [event.event for event in events] == ["execution_started"]

    monkeypatch.setattr(workflow_runner, "create_execution_context", record_create)
    workflow = Workflow(
        root=root,
        runs=(
            Run(
                "ordered",
                (successful_step("first"), successful_step("second")),
                resolve_execution=resolve,
                before_first_step=before_first_step,
            ),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.exit_code == 0
    assert order == ["resolve", "metadata", "hook"]
    events = execution_events(root, "ordered", result.run("ordered").execution_id or "")
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_completed",
        "phase_completed",
        "phase_started",
        "task_started",
        "task_completed",
        "phase_completed",
        "execution_completed",
    ]


def test_workflow_records_sys_argv_and_chains_previous_execution_under_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["python", "workflow.py", "--token=secret"])
    workflow = Workflow(root=root, runs=(Run("chain", (successful_step("train"),)),))

    first = workflow.run(base_environment={}).run("chain")
    second = workflow.run(base_environment={}).run("chain")
    first_metadata = join_execution_context(root / "chain", first.execution_id or "").metadata
    second_metadata = join_execution_context(root / "chain", second.execution_id or "").metadata

    assert first_metadata.command == ("python", "workflow.py", "--token=<redacted>")
    assert first_metadata.previous_execution_id is None
    assert second_metadata.previous_execution_id == first.execution_id
    with claim_logical_run_lease(root / "chain"):
        pass


def test_before_first_step_failure_returns_result_without_phase_events(tmp_path: Path) -> None:
    def fail_before_step(_context: Any) -> None:
        raise OSError("hook unavailable")

    workflow = Workflow(
        root=tmp_path,
        runs=(
            Run(
                "hook-failure",
                (successful_step("train"),),
                before_first_step=fail_before_step,
            ),
        ),
    )

    result = workflow.run(base_environment={})
    run_result = result.run("hook-failure")
    events = execution_events(tmp_path, "hook-failure", run_result.execution_id or "")

    assert result.exit_code == 1
    assert run_result.outcome == "failed"
    assert run_result.steps == ()
    assert "hook unavailable" in (run_result.reason or "")
    assert [event.event for event in events] == ["execution_started", "execution_failed"]


def test_first_failure_stops_dispatch_and_keeps_unstarted_runs_artifact_free(
    tmp_path: Path,
) -> None:
    alpha_train = tmp_path / "alpha-train"
    gamma_prepare = tmp_path / "gamma-prepare"
    workflow = Workflow(
        root=tmp_path / "runs",
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run(
                "alpha",
                (successful_step("prepare"), successful_step("train", marker=alpha_train)),
            ),
            Run(
                "beta",
                (
                    Step("prepare", (sys.executable, "-c", "raise SystemExit(7)")),
                    successful_step("train"),
                ),
            ),
            Run(
                "gamma",
                (
                    successful_step("prepare", marker=gamma_prepare),
                    successful_step("train"),
                ),
            ),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.exit_code == 7
    assert result.run("alpha").outcome == "blocked"
    assert result.run("beta").outcome == "failed"
    assert result.run("gamma").outcome == "blocked"
    assert result.run("gamma").execution_id is None
    assert not alpha_train.exists()
    assert not gamma_prepare.exists()
    assert not (tmp_path / "runs" / "gamma").exists()
    assert [(item.run_name, item.step.name) for item in result.dispatch] == [
        ("alpha", "prepare"),
        ("beta", "prepare"),
    ]


@pytest.mark.parametrize(
    ("process", "expected_exit", "expected_outcome"),
    (
        (ProcessResult(-15, 0.1, timed_out=True, signal=15), 1, "failed"),
        (ProcessResult(-9, 0.1, signal=9), 137, "interrupted"),
        (ProcessResult(12, 0.1), 12, "failed"),
    ),
)
def test_process_failure_exit_code_precedence(
    process: ProcessResult,
    expected_exit: int,
    expected_outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_runner, "launch_process", lambda *args, **kwargs: process)
    workflow = Workflow(root=tmp_path, runs=(Run("run", (Step("step", ("ignored",)),)),))

    result = workflow.run(base_environment={})

    assert result.exit_code == expected_exit
    assert result.step("run", "step").outcome == expected_outcome


def test_injected_launcher_replaces_process_creation_without_spawning_a_child(
    tmp_path: Path,
) -> None:
    """A public ``launcher`` parameter substitutes process creation for a fake."""
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_launcher(
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> ProcessResult:
        del cwd, timeout_seconds
        calls.append((command, dict(environment)))
        return ProcessResult(return_code=0, duration_seconds=0.0)

    workflow = Workflow(
        root=tmp_path,
        runs=(Run("run", (Step("step", ("this-binary-does-not-exist",)),)),),
        launcher=fake_launcher,
    )

    result = workflow.run(base_environment={})

    assert result.successful
    assert result.step("run", "step").outcome == "completed"
    assert [command for command, _ in calls] == [("this-binary-does-not-exist",)]


def test_construction_rejects_a_non_callable_launcher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="launcher must be callable"):
        Workflow(
            root=tmp_path,
            runs=(Run("run", (Step("step", ("ignored",)),)),),
            launcher="not callable",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("order", "alpha_outcome", "expected_dispatch"),
    (
        (
            "step-major",
            "blocked",
            (("alpha", "prepare"), ("beta", "prepare")),
        ),
        (
            "run-major",
            "completed",
            (("alpha", "prepare"), ("alpha", "train"), ("beta", "prepare")),
        ),
    ),
)
def test_child_signal_interrupts_only_its_run_and_blocks_undispatched_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    alpha_outcome: str,
    expected_dispatch: tuple[tuple[str, str], ...],
) -> None:
    """A child signal is run-local rather than a workflow-level interruption."""

    def launch_in_order(*args: Any, **kwargs: Any) -> ProcessResult:
        del args
        environment = kwargs["environment"]
        if environment["MAMMOTH_RUN_NAME"] == "beta" and environment["MAMMOTH_PHASE"] == "prepare":
            return ProcessResult(-signal.SIGKILL, 0.1, signal=signal.SIGKILL)
        return ProcessResult(0, 0.1)

    monkeypatch.setattr(workflow_runner, "launch_process", launch_in_order)
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        order=order,  # type: ignore[arg-type]
        step_order=None if order == "run-major" else ("prepare", "train"),
        runs=(
            Run(
                "alpha",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
            Run(
                "beta",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
            Run(
                "gamma",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.exit_code == 128 + signal.SIGKILL
    assert result.run("alpha").outcome == alpha_outcome
    assert result.run("beta").outcome == "interrupted"
    assert result.run("gamma").outcome == "blocked"
    assert result.run("gamma").execution_id is None
    assert not (root / "gamma").exists()
    assert tuple((item.run_name, item.step.name) for item in result.dispatch) == expected_dispatch


def test_parent_signal_during_launch_interrupts_active_runs_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal caught by the workflow process remains workflow-scoped."""

    def launch_with_parent_signal(*args: Any, **kwargs: Any) -> ProcessResult:
        del args
        environment = kwargs["environment"]
        if environment["MAMMOTH_RUN_NAME"] == "beta" and environment["MAMMOTH_PHASE"] == "prepare":
            return ProcessResult(
                -signal.SIGTERM,
                0.1,
                interrupted=True,
                signal=signal.SIGTERM,
            )
        return ProcessResult(0, 0.1)

    monkeypatch.setattr(workflow_runner, "launch_process", launch_with_parent_signal)
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run(
                "alpha",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
            Run(
                "beta",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
            Run(
                "gamma",
                (Step("prepare", ("ignored",)), Step("train", ("ignored",))),
            ),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.exit_code == 128 + signal.SIGTERM
    assert result.run("alpha").outcome == "interrupted"
    assert result.run("beta").outcome == "interrupted"
    assert result.run("gamma").outcome == "blocked"
    assert result.run("gamma").execution_id is None
    assert not (root / "gamma").exists()


def test_launch_failure_returns_structured_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_launch(*args: Any, **kwargs: Any) -> ProcessResult:
        del args, kwargs
        raise FileNotFoundError("launcher missing")

    monkeypatch.setattr(workflow_runner, "launch_process", fail_launch)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (Step("step", ("missing",)),)),),
    ).run(base_environment={})

    assert result.exit_code == 1
    assert result.run("run").outcome == "failed"
    assert "launcher missing" in (result.step("run", "step").reason or "")


@pytest.mark.parametrize("signal_number", (signal.SIGINT, signal.SIGTERM))
def test_pre_activation_signal_returns_blocked_artifact_free_result_and_restores_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_number: signal.Signals,
) -> None:
    """Signals during run-owned setup return before layouts, leases, or children exist."""
    root = tmp_path / "runs"
    handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    launched = False

    def interrupt_base_environment(_environment: Any) -> Mapping[str, str]:
        signal.raise_signal(signal_number)
        raise AssertionError("workflow interruption must leave base setup immediately")

    def record_launch(*args: Any, **kwargs: Any) -> ProcessResult:
        nonlocal launched
        del args, kwargs
        launched = True
        return ProcessResult(0, 0.0)

    monkeypatch.setattr(workflow_runner, "_base_environment", interrupt_base_environment)
    monkeypatch.setattr(workflow_runner, "launch_process", record_launch)
    workflow = Workflow(
        root=root,
        runs=(
            Run("alpha", (Step("step", ("ignored",)),)),
            Run("beta", (Step("step", ("ignored",)),)),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.exit_code == 128 + signal_number
    assert all(run.outcome == "blocked" for run in result.runs)
    assert all(run.execution_id is None and run.steps == () for run in result.runs)
    assert result.dispatch == ()
    assert not launched
    assert not root.exists()
    assert {number: signal.getsignal(number) for number in handlers} == handlers


def test_invalid_base_environment_raises_and_restores_signal_handlers(tmp_path: Path) -> None:
    """Ordinary setup errors remain exceptions despite the full run signal boundary."""
    root = tmp_path / "runs"
    handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    workflow = Workflow(root=root, runs=(Run("run", (Step("step", ("ignored",)),)),))

    with pytest.raises(ValueError, match="environment values must be strings"):
        workflow.run(base_environment={"INVALID": 1})  # type: ignore[dict-item]

    assert {number: signal.getsignal(number) for number in handlers} == handlers
    assert not root.exists()


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="POSIX signal masking is unavailable",
)
def test_signal_handler_installation_masks_sigterm_until_workflow_owns_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SIGTERM between handler replacements is deferred to Mammoth's completed set."""
    original_signal = signal.signal
    injected = False

    class Marker(BaseException):
        pass

    def old_sigterm_handler(signal_number: int, _frame: object) -> None:
        raise Marker(f"old SIGTERM handler received {signal_number}")

    def signal_with_install_window(signal_number: signal.Signals, handler: Any) -> Any:
        nonlocal injected
        if signal_number == signal.SIGTERM and not injected:
            injected = True
            signal.raise_signal(signal.SIGTERM)
        return original_signal(signal_number, handler)

    previous_sigterm = original_signal(signal.SIGTERM, old_sigterm_handler)
    monkeypatch.setattr(workflow_runner.signal, "signal", signal_with_install_window)
    try:
        result = Workflow(
            root=tmp_path / "runs",
            runs=(Run("run", (Step("step", ("ignored",)),)),),
        ).run(base_environment={})
        assert signal.getsignal(signal.SIGTERM) is old_sigterm_handler
    finally:
        original_signal(signal.SIGTERM, previous_sigterm)

    assert injected
    assert result.exit_code == 128 + signal.SIGTERM
    assert result.run("run").outcome == "blocked"
    assert not (tmp_path / "runs").exists()


def test_workflow_run_rejects_worker_thread_before_artifacts_or_children(tmp_path: Path) -> None:
    """Workflow execution cannot silently continue without main-thread signal ownership."""
    root = tmp_path / "runs"
    workflow = Workflow(root=root, runs=(Run("run", (Step("step", ("ignored",)),)),))
    results: list[WorkflowResult] = []
    errors: list[BaseException] = []

    def run_workflow() -> None:
        try:
            results.append(workflow.run(base_environment={}))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run_workflow)
    worker.start()
    worker.join()

    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "Workflow.run() requires the main thread to own process signals"
    assert not root.exists()


def test_workflow_run_rejects_missing_signal_masking_before_artifacts_or_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow execution does not weaken the signal contract without POSIX masking."""
    root = tmp_path / "runs"
    workflow = Workflow(root=root, runs=(Run("run", (Step("step", ("ignored",)),)),))
    monkeypatch.delattr(workflow_runner.signal, "pthread_sigmask", raising=False)

    with pytest.raises(RuntimeError, match="requires POSIX pthread_sigmask signal masking"):
        workflow.run(base_environment={})

    assert not root.exists()


def test_workflow_interruption_returns_signal_exit_and_closes_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> ProcessResult:
        del args, kwargs
        raise workflow_runner._WorkflowInterrupted(signal.SIGTERM)

    monkeypatch.setattr(workflow_runner, "launch_process", interrupt)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (Step("step", ("ignored",)),)),),
    ).run(base_environment={})
    run_result = result.run("run")
    events = execution_events(tmp_path, "run", run_result.execution_id or "")

    assert result.exit_code == 128 + signal.SIGTERM
    assert run_result.outcome == "interrupted"
    assert result.step("run", "step").signal == signal.SIGTERM
    assert [event.event for event in events][-3:] == [
        "task_failed",
        "phase_failed",
        "execution_interrupted",
    ]


@pytest.mark.parametrize(
    ("process", "expected_exit", "expected_step", "expected_execution"),
    (
        (ProcessResult(0, 0.1), 128 + signal.SIGTERM, "completed", "execution_interrupted"),
        (
            ProcessResult(-15, 0.1, timed_out=True, signal=15),
            1,
            "failed",
            "execution_failed",
        ),
    ),
)
def test_step_terminal_signal_preserves_one_recorded_result_and_timeout_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process: ProcessResult,
    expected_exit: int,
    expected_step: str,
    expected_execution: str,
) -> None:
    """A deferred signal cannot duplicate terminal events or discard a timeout."""
    original_terminal = workflow_runner._emit_step_terminal

    def emit_then_signal(active: Any, step: Step, result: Any) -> None:
        original_terminal(active, step, result)
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(workflow_runner, "launch_process", lambda *args, **kwargs: process)
    monkeypatch.setattr(workflow_runner, "_emit_step_terminal", emit_then_signal)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (Step("step", ("ignored",)),)),),
    ).run(base_environment={})
    run_result = result.run("run")
    step_result = result.step("run", "step")
    events = execution_events(tmp_path, "run", run_result.execution_id or "")

    assert result.exit_code == expected_exit
    assert step_result.outcome == expected_step
    assert run_result.steps == (step_result,)
    assert len(result.dispatch) == 1
    assert [event.event for event in events].count("task_completed") <= 1
    assert [event.event for event in events].count("task_failed") <= 1
    assert [event.event for event in events].count("phase_completed") <= 1
    assert [event.event for event in events].count("phase_failed") <= 1
    assert events[-1].event == expected_execution


def test_signal_before_execution_terminal_retries_finalization_as_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first finalization-window signal is returned and resources still close."""
    original_emit = workflow_runner.RunObserver.emit
    signaled = False

    def emit_with_signal(observer: Any, event: str, **fields: Any) -> Any:
        nonlocal signaled
        if event == "execution_completed" and not signaled:
            signaled = True
            signal.raise_signal(signal.SIGTERM)
        return original_emit(observer, event, **fields)

    monkeypatch.setattr(workflow_runner.RunObserver, "emit", emit_with_signal)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),)),),
    ).run(base_environment={})
    run_result = result.run("run")
    events = execution_events(tmp_path, "run", run_result.execution_id or "")

    assert result.exit_code == 128 + signal.SIGTERM
    assert run_result.outcome == "interrupted"
    assert [event.event for event in events][-1] == "execution_interrupted"
    assert "execution_completed" not in [event.event for event in events]
    with claim_logical_run_lease(tmp_path / "run"):
        pass


def test_signal_after_durable_execution_terminal_does_not_emit_another_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal after the barrier preserves the accepted terminal transition."""
    original_emit = workflow_runner.RunObserver.emit
    signaled = False

    def emit_then_signal(observer: Any, event: str, **fields: Any) -> Any:
        nonlocal signaled
        observation = original_emit(observer, event, **fields)
        if event == "execution_completed" and not signaled:
            signaled = True
            signal.raise_signal(signal.SIGTERM)
        return observation

    monkeypatch.setattr(workflow_runner.RunObserver, "emit", emit_then_signal)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),)),),
    ).run(base_environment={})
    run_result = result.run("run")
    events = execution_events(tmp_path, "run", run_result.execution_id or "")
    terminal_events = [
        event.event
        for event in events
        if event.event.startswith("execution_") and event.event != "execution_started"
    ]

    assert result.exit_code == 128 + signal.SIGTERM
    assert run_result.outcome == "completed"
    assert terminal_events == ["execution_completed"]
    with claim_logical_run_lease(tmp_path / "run"):
        pass


def test_signal_during_observer_close_is_not_discarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-terminal cleanup signal still controls the workflow process code."""
    original_close = workflow_runner.RunObserver.close
    signaled = False

    def close_with_signal(observer: Any) -> None:
        nonlocal signaled
        if not signaled:
            signaled = True
            signal.raise_signal(signal.SIGTERM)
        original_close(observer)

    monkeypatch.setattr(workflow_runner.RunObserver, "close", close_with_signal)
    result = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),)),),
    ).run(base_environment={})

    assert result.exit_code == 128 + signal.SIGTERM
    assert result.run("run").outcome == "completed"
    with claim_logical_run_lease(tmp_path / "run"):
        pass


def test_cleanup_error_survives_later_signal_and_failed_close_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal retries retain earlier cleanup errors and retry a pre-close failure."""
    original_emit = workflow_runner.RunObserver.emit
    original_close = workflow_runner.RunObserver.close
    close_error = OSError("observer close blocked before release")
    completed_emits = 0
    close_attempts = 0

    def emit_second_terminal_signal(observer: Any, event: str, **fields: Any) -> Any:
        nonlocal completed_emits
        if event == "execution_completed":
            completed_emits += 1
            if completed_emits == 2:
                signal.raise_signal(signal.SIGTERM)
        return original_emit(observer, event, **fields)

    def fail_first_close(observer: Any) -> None:
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise close_error
        original_close(observer)

    monkeypatch.setattr(workflow_runner.RunObserver, "emit", emit_second_terminal_signal)
    monkeypatch.setattr(workflow_runner.RunObserver, "close", fail_first_close)
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        runs=(
            Run("alpha", (successful_step("step"),)),
            Run("beta", (successful_step("step"),)),
        ),
    )

    with pytest.raises(OSError, match="observer close blocked") as raised:
        workflow.run(base_environment={})

    assert raised.value is close_error
    assert close_attempts == 3
    for run_name in ("alpha", "beta"):
        with claim_logical_run_lease(root / run_name):
            pass


def test_terminal_emit_failure_retries_before_observer_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient pre-emit failure is retried without duplicating the terminal."""
    original_emit = workflow_runner.RunObserver.emit
    terminal_error = OSError("terminal dispatch unavailable")
    terminal_attempts = 0

    def fail_first_terminal(observer: Any, event: str, **fields: Any) -> Any:
        nonlocal terminal_attempts
        if event == "execution_completed":
            terminal_attempts += 1
            if terminal_attempts == 1:
                raise terminal_error
        return original_emit(observer, event, **fields)

    monkeypatch.setattr(workflow_runner.RunObserver, "emit", fail_first_terminal)
    workflow = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),)),),
    )

    with pytest.raises(OSError, match="terminal dispatch unavailable") as raised:
        workflow.run(base_environment={})

    assert raised.value is terminal_error
    assert terminal_attempts == 2
    execution_dir = next((tmp_path / "run" / "logs" / "executions").iterdir())
    events = execution_events(tmp_path, "run", execution_dir.name)
    assert [event.event for event in events].count("execution_completed") == 1
    with claim_logical_run_lease(tmp_path / "run"):
        pass


def test_lease_close_failure_retries_before_reporting_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient pre-close lease error gets a second bounded cleanup attempt."""
    original_claim = workflow_runner.claim_logical_run_lease
    lease_error = OSError("lease close interrupted before release")
    close_attempts = 0

    class RetryLease:
        def __init__(self, run_dir: Path) -> None:
            self.inner = original_claim(run_dir)

        def close(self) -> None:
            nonlocal close_attempts
            close_attempts += 1
            if close_attempts == 1:
                raise lease_error
            self.inner.close()

        def retire(self) -> None:
            self.close()

    monkeypatch.setattr(
        workflow_runner,
        "claim_logical_run_lease",
        lambda run_dir: RetryLease(run_dir),
    )
    workflow = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),)),),
    )

    with pytest.raises(OSError, match="lease close interrupted") as raised:
        workflow.run(base_environment={})

    assert raised.value is lease_error
    assert close_attempts == 2
    with original_claim(tmp_path / "run"):
        pass


def test_later_resolver_failure_finalizes_active_runs_and_preserves_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    original_error = PermissionError("checkpoint resolution denied")
    original_close = workflow_runner.RunObserver.close
    close_calls = 0

    def fail_resolver(_context: Any) -> Execution:
        raise original_error

    def close_then_fail_once(observer: Any) -> None:
        nonlocal close_calls
        original_close(observer)
        close_calls += 1
        if close_calls == 1:
            raise OSError("observer close failed")

    monkeypatch.setattr(workflow_runner.RunObserver, "close", close_then_fail_once)
    workflow = Workflow(
        root=root,
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run("alpha", (successful_step("prepare"), successful_step("train"))),
            Run("beta", (successful_step("prepare"), successful_step("train"))),
            Run(
                "gamma",
                (successful_step("prepare"), successful_step("train")),
                resolve_execution=fail_resolver,
            ),
        ),
    )

    with pytest.raises(PermissionError, match="checkpoint resolution denied") as raised:
        workflow.run(base_environment={})

    assert raised.value is original_error
    assert any("observer close failed" in note for note in original_error.__notes__)
    for run_name in ("alpha", "beta"):
        run_dir = root / run_name
        with claim_logical_run_lease(run_dir):
            pass
        execution_dirs = tuple((run_dir / "logs" / "executions").iterdir())
        events = execution_events(root, run_name, execution_dirs[0].name)
        assert events[-1].event == "execution_failed"
    with claim_logical_run_lease(root / "gamma"):
        pass
    assert tuple((root / "gamma" / "logs" / "executions").iterdir()) == ()


def test_resolver_error_survives_untransferred_lease_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver exception remains primary when its newly claimed lease cleanup fails."""
    original_claim = workflow_runner.claim_logical_run_lease
    resolver_error = LookupError("resume source unavailable")

    class FailingLease:
        def __init__(self, run_dir: Path) -> None:
            self.lease = original_claim(run_dir)

        def close(self) -> None:
            self.lease.close()
            raise OSError("lease close report failed")

    def fail_resolver(_context: Any) -> Execution:
        raise resolver_error

    monkeypatch.setattr(
        workflow_runner,
        "claim_logical_run_lease",
        lambda run_dir: FailingLease(run_dir),
    )
    workflow = Workflow(
        root=tmp_path,
        runs=(Run("run", (successful_step("step"),), resolve_execution=fail_resolver),),
    )

    with pytest.raises(LookupError, match="resume source unavailable") as raised:
        workflow.run(base_environment={})

    assert raised.value is resolver_error
    assert any("lease close report failed" in note for note in resolver_error.__notes__)
    with original_claim(tmp_path / "run"):
        pass


def test_child_environment_uses_only_canonical_mammoth_variables(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    source = (
        "import json, os; from pathlib import Path; "
        f"Path({str(output)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')"
    )
    workflow = Workflow(
        root=tmp_path / "runs",
        runs=(
            Run(
                "environment",
                (
                    Step(
                        "phase with space",
                        (sys.executable, "-c", source),
                        environment={"X": "step"},
                    ),
                ),
                environment={"X": "run", "RUN_ONLY": "yes"},
            ),
        ),
    )

    result = workflow.run(
        base_environment={
            "X": "base",
            "MAMMOTH_EXECUTION_ID": "wrong",
            "MAMMOTH_OLD_ALIAS": "remove-me",
        }
    )
    child = json.loads(output.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert child["X"] == "step"
    assert child["RUN_ONLY"] == "yes"
    assert {name for name in child if name.startswith("MAMMOTH_")} == {
        "MAMMOTH_EXECUTION_ID",
        "MAMMOTH_RUN_NAME",
        "MAMMOTH_INVOCATION_KIND",
        "MAMMOTH_PHASE",
    }
    assert child["MAMMOTH_RUN_NAME"] == "environment"
    assert child["MAMMOTH_PHASE"] == "phase with space"


def test_workflow_success_closes_observer_and_releases_lease(tmp_path: Path) -> None:
    workflow = Workflow(root=tmp_path, runs=(Run("run", (successful_step("step"),)),))

    result = workflow.run(base_environment={})

    assert result.successful
    with claim_logical_run_lease(tmp_path / "run"):
        pass


def test_launch_timeout_includes_started_callback_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []
    clock = [0.0]

    class FakeSupervisedProcess:
        pid = 123
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> FakeSupervisedProcess:
            return self

        def wait(self, timeout: float | None = None) -> int:
            timeouts.append(timeout)
            return 0

    def slow_started(_pid: int) -> None:
        clock[0] = 0.1

    monkeypatch.setattr("mammoth.workflow.launch.SupervisedProcess", FakeSupervisedProcess)
    monkeypatch.setattr("mammoth.workflow.launch.time.monotonic", lambda: clock[0])
    result = launch_process(
        ("ignored",),
        cwd=None,
        environment={},
        timeout_seconds=0.05,
        on_started=slow_started,
    )

    assert timeouts == [0.0]
    assert result.timed_out
    assert result.duration_seconds == 0.1


def test_launch_defers_signal_until_new_child_is_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A startup-window signal produces a result only after the child is reaped."""
    stopped = False

    class FakeSupervisedProcess:
        pid = 123
        returncode = -signal.SIGTERM

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> FakeSupervisedProcess:
            signal.raise_signal(signal.SIGTERM)
            return self

        def stop(self) -> int:
            nonlocal stopped
            stopped = True
            return self.returncode

    monkeypatch.setattr("mammoth.workflow.launch.SupervisedProcess", FakeSupervisedProcess)

    result = launch_process(
        ("ignored",),
        cwd=None,
        environment={},
        timeout_seconds=None,
    )

    assert stopped
    assert result.interrupted
    assert result.signal == signal.SIGTERM
    assert result.return_code == -signal.SIGTERM


def test_launch_start_failure_preserves_deferred_signal_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded startup signal wins even when child creation also raises."""
    stopped = False

    class FakeSupervisedProcess:
        returncode = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> FakeSupervisedProcess:
            signal.raise_signal(signal.SIGTERM)
            raise FileNotFoundError("launcher unavailable")

        def stop(self) -> int | None:
            nonlocal stopped
            stopped = True
            return self.returncode

    monkeypatch.setattr(workflow_launch, "SupervisedProcess", FakeSupervisedProcess)

    result = launch_process(
        ("missing-launcher",),
        cwd=None,
        environment={},
        timeout_seconds=None,
    )

    assert not stopped
    assert result.interrupted
    assert result.signal == signal.SIGTERM
    assert result.return_code == -signal.SIGTERM


def test_launch_handler_restore_exception_still_reaps_started_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer handler delivery during ownership handoff cannot strand a child."""
    stopped = False
    injected = False
    original_signal = signal.signal

    class OuterSignal(BaseException):
        pass

    def outer_handler(_signal_number: int, _frame: object) -> None:
        raise OuterSignal

    previous_sigterm = original_signal(signal.SIGTERM, outer_handler)

    class FakeSupervisedProcess:
        pid = 123
        returncode = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> FakeSupervisedProcess:
            return self

        def stop(self) -> int:
            nonlocal stopped
            stopped = True
            self.returncode = -signal.SIGTERM
            return self.returncode

    def signal_with_restore_injection(
        signal_number: signal.Signals,
        handler: Any,
    ) -> Any:
        nonlocal injected
        previous = original_signal(signal_number, handler)
        if signal_number == signal.SIGTERM and handler is outer_handler and not injected:
            injected = True
            signal.raise_signal(signal.SIGTERM)
        return previous

    monkeypatch.setattr(workflow_launch, "SupervisedProcess", FakeSupervisedProcess)
    monkeypatch.setattr(workflow_launch.signal, "signal", signal_with_restore_injection)
    try:
        with pytest.raises(OuterSignal):
            launch_process(
                ("ignored",),
                cwd=None,
                environment={},
                timeout_seconds=None,
            )
    finally:
        original_signal(signal.SIGTERM, previous_sigterm)

    assert injected
    assert stopped


def test_supervisor_stops_descendant_in_an_independent_session(tmp_path: Path) -> None:
    worker_pid_path = tmp_path / "worker.pid"
    launcher = "\n".join(
        (
            "import subprocess",
            "import sys",
            "import time",
            "from pathlib import Path",
            "worker = subprocess.Popen(",
            "    (sys.executable, '-c', 'import time; time.sleep(60)'),",
            "    start_new_session=True,",
            ")",
            f"Path({str(worker_pid_path)!r}).write_text(str(worker.pid), encoding='utf-8')",
            "while True:",
            "    time.sleep(1)",
        )
    )
    supervisor = SupervisedProcess(
        (sys.executable, "-c", launcher),
        cwd=None,
        environment={},
        terminate_grace_seconds=0.1,
        descendant_grace_seconds=0.1,
    ).start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not worker_pid_path.is_file():
        time.sleep(0.01)
    worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))

    supervisor.stop()

    with pytest.raises(ProcessLookupError):
        os.kill(supervisor.pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_captured_process_separates_stdout_stderr_and_nonzero_exit() -> None:
    result = run_captured_process(
        (
            sys.executable,
            "-c",
            "import sys; print('standard'); print('diagnostic', file=sys.stderr); "
            "raise SystemExit(7)",
        ),
        cwd=None,
        environment={},
        timeout_seconds=None,
    )

    assert result.stdout == "standard\n"
    assert result.stderr == "diagnostic\n"
    assert result.return_code == 7
    assert result.duration_seconds >= 0
    assert not result.timed_out
    assert result.signal is None


def test_captured_process_timeout_returns_partial_output_and_reaps_launcher() -> None:
    result = run_captured_process(
        (
            sys.executable,
            "-c",
            "import sys, time; print('before timeout', flush=True); "
            "print('before diagnostic', file=sys.stderr, flush=True); time.sleep(60)",
        ),
        cwd=None,
        environment={},
        timeout_seconds=0.05,
        terminate_grace_seconds=0.1,
    )

    assert result.timed_out
    assert result.return_code < 0
    assert result.signal is not None
    assert result.stdout == "before timeout\n"
    assert result.stderr == "before diagnostic\n"


def test_captured_process_cleans_up_when_communication_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "interrupted-child.pid"

    def interrupt_communication(
        process: subprocess.Popen[str],
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        del process, input, timeout
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not child_pid_path.is_file():
            time.sleep(0.01)
        raise RuntimeError("injected communication interruption")

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_communication)
    with pytest.raises(RuntimeError, match="injected communication interruption"):
        run_captured_process(
            (
                sys.executable,
                "-c",
                "import os, time; from pathlib import Path; "
                f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                "time.sleep(60)",
            ),
            cwd=None,
            environment={},
            timeout_seconds=None,
            terminate_grace_seconds=0.1,
        )
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_captured_process_cleans_up_when_start_raises_after_child_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child assigned just before a start exception is still owned and stopped."""
    stopped = False

    class FakeProcess:
        returncode = None

    def start_then_interrupt(
        supervisor: SupervisedProcess,
        *,
        capture_output: bool,
    ) -> SupervisedProcess:
        assert capture_output
        supervisor._process = FakeProcess()  # type: ignore[assignment]
        raise KeyboardInterrupt

    def record_stop(supervisor: SupervisedProcess) -> int:
        nonlocal stopped
        stopped = True
        assert supervisor._process is not None
        supervisor._process.returncode = -signal.SIGINT
        return -signal.SIGINT

    monkeypatch.setattr(SupervisedProcess, "_start", start_then_interrupt)
    monkeypatch.setattr(SupervisedProcess, "stop", record_stop)

    with pytest.raises(KeyboardInterrupt):
        run_captured_process(
            ("ignored",),
            cwd=None,
            environment={},
            timeout_seconds=None,
        )

    assert stopped


def test_captured_start_signal_is_redelivered_to_workflow_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured-process cleanup preserves the outer workflow signal handler."""
    stopped = False

    class FakeProcess:
        returncode = None

    class FakeSupervisedProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self._process: FakeProcess | None = None

        @property
        def process(self) -> FakeProcess:
            assert self._process is not None
            return self._process

        def _start(self, *, capture_output: bool) -> FakeSupervisedProcess:
            assert capture_output
            self._process = FakeProcess()
            signal.raise_signal(signal.SIGTERM)
            return self

        def stop(self) -> int:
            nonlocal stopped
            stopped = True
            assert self._process is not None
            self._process.returncode = -signal.SIGTERM
            return self._process.returncode

    def start_captured_process(_context: Any) -> None:
        run_captured_process(
            ("ignored",),
            cwd=None,
            environment={},
            timeout_seconds=None,
        )

    monkeypatch.setattr(workflow_launch, "SupervisedProcess", FakeSupervisedProcess)
    result = Workflow(
        root=tmp_path,
        runs=(
            Run(
                "run",
                (Step("step", ("ignored",)),),
                before_first_step=start_captured_process,
            ),
        ),
    ).run(base_environment={})

    assert stopped
    assert result.exit_code == 128 + signal.SIGTERM
    assert result.run("run").outcome == "interrupted"
    assert result.run("run").steps == ()


def group_events(root: Path, group_id: str):
    """Read one workflow's group event stream by its published group ID."""
    return read_group_events(GroupLayout(root, group_id).events_path)


def test_group_manifest_is_published_before_the_first_step_in_run_major_order(
    tmp_path: Path,
) -> None:
    """The manifest records the full declared plan; events mirror run-major dispatch."""
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        runs=(
            Run("alpha", (successful_step("prepare"), successful_step("train"))),
            Run("beta", (successful_step("prepare"),)),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.group_id is not None
    layout = GroupLayout(root, result.group_id)
    manifest = load_group_manifest(layout.manifest_path)
    assert manifest.order == "run-major"
    assert [(member.run_name, member.steps) for member in manifest.members] == [
        ("alpha", ("prepare", "train")),
        ("beta", ("prepare",)),
    ]
    events = group_events(root, result.group_id)
    # Run terminals are only known once the whole dispatch finishes (workflow
    # finalization runs once, after every step), so they batch at the very
    # end in declaration order rather than interleaving with step events.
    assert [(event.event, event.run_name, event.step_name) for event in events] == [
        ("group_started", None, None),
        ("run_started", "alpha", None),
        ("step_started", "alpha", "prepare"),
        ("step_completed", "alpha", "prepare"),
        ("step_started", "alpha", "train"),
        ("step_completed", "alpha", "train"),
        ("run_started", "beta", None),
        ("step_started", "beta", "prepare"),
        ("step_completed", "beta", "prepare"),
        ("run_completed", "alpha", None),
        ("run_completed", "beta", None),
        ("group_completed", None, None),
    ]


def test_group_events_mirror_step_major_interleaving(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run("alpha", (successful_step("prepare"), successful_step("train"))),
            Run("beta", (successful_step("prepare"), successful_step("train"))),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.group_id is not None
    manifest = load_group_manifest(GroupLayout(root, result.group_id).manifest_path)
    assert manifest.order == "step-major"
    events = group_events(root, result.group_id)
    step_events = [
        (event.event, event.run_name, event.step_name)
        for event in events
        if event.event in {"step_started", "step_completed"}
    ]
    assert step_events == [
        ("step_started", "alpha", "prepare"),
        ("step_completed", "alpha", "prepare"),
        ("step_started", "beta", "prepare"),
        ("step_completed", "beta", "prepare"),
        ("step_started", "alpha", "train"),
        ("step_completed", "alpha", "train"),
        ("step_started", "beta", "train"),
        ("step_completed", "beta", "train"),
    ]
    assert events[-1].event == "group_completed"


def test_group_metadata_round_trips_and_detaches_caller_collections(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    metadata = {"campaign": "demo", "tags": UserList(["a", "b"])}
    workflow = Workflow(
        root=root,
        runs=(Run("run", (successful_step("step"),)),),
        group_metadata=metadata,
    )
    metadata["tags"].append("changed-after-construction")

    result = workflow.run(base_environment={})

    assert result.group_id is not None
    manifest = load_group_manifest(GroupLayout(root, result.group_id).manifest_path)
    assert manifest.to_dict()["metadata"] == {"campaign": "demo", "tags": ["a", "b"]}


def test_group_terminal_status_is_failed_and_never_names_blocked_members(
    tmp_path: Path,
) -> None:
    """A stopped dispatch records failure and omits never-activated members."""
    root = tmp_path / "runs"
    workflow = Workflow(
        root=root,
        order="step-major",
        step_order=("prepare", "train"),
        runs=(
            Run("alpha", (successful_step("prepare"), successful_step("train"))),
            Run(
                "beta",
                (
                    Step("prepare", (sys.executable, "-c", "raise SystemExit(3)")),
                    successful_step("train"),
                ),
            ),
            Run("gamma", (successful_step("prepare"), successful_step("train"))),
        ),
    )

    result = workflow.run(base_environment={})

    assert result.group_id is not None
    events = group_events(root, result.group_id)
    assert [event.run_name for event in events if event.event == "run_started"] == ["alpha", "beta"]
    assert ("run_blocked", "alpha") in [(event.event, event.run_name) for event in events]
    assert ("run_failed", "beta") in [(event.event, event.run_name) for event in events]
    assert "gamma" not in [event.run_name for event in events if event.run_name is not None]
    assert events[-1].event == "group_failed"


def test_group_terminal_status_is_interrupted_on_workflow_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> ProcessResult:
        del args, kwargs
        raise workflow_runner._WorkflowInterrupted(signal.SIGTERM)

    monkeypatch.setattr(workflow_runner, "launch_process", interrupt)
    root = tmp_path / "runs"
    result = Workflow(
        root=root,
        runs=(Run("run", (Step("step", ("ignored",)),)),),
    ).run(base_environment={})

    assert result.group_id is not None
    events = group_events(root, result.group_id)
    assert events[-1].event == "group_interrupted"
    assert events[-1].signal == signal.SIGTERM


def test_group_artifacts_are_absent_when_no_run_ever_activates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal caught during setup leaves no group manifest or event stream."""
    root = tmp_path / "runs"

    def interrupt_base_environment(_environment: Any) -> Mapping[str, str]:
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("workflow interruption must leave base setup immediately")

    monkeypatch.setattr(workflow_runner, "_base_environment", interrupt_base_environment)
    workflow = Workflow(root=root, runs=(Run("run", (Step("step", ("ignored",)),)),))

    result = workflow.run(base_environment={})

    assert result.group_id is None
    assert not root.exists()


def test_loose_runs_outside_a_workflow_remain_valid_without_group_artifacts(
    tmp_path: Path,
) -> None:
    """A run created directly through core APIs never requires ``.mammoth/``."""
    from mammoth.core import RunLayout, create_execution_context

    entry = tmp_path / "runs"
    layout = RunLayout(entry, "loose-run").prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name="loose-run",
        invocation_kind="direct",
        intended_phases=("only",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
    )

    assert context.metadata.run_name == "loose-run"
    assert not (entry / ".mammoth").exists()
