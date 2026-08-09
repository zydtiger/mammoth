from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import (
    RunLayout,
    claim_logical_run_lease,
    create_execution_context,
    latest_execution_id,
    read_execution_events,
)
from mammoth.logging import RunObserver
from mammoth.workflow import (
    DispatchEntry,
    ExecutionInputs,
    LifecycleEventContext,
    PreDispatchContext,
    ProgrammaticRun,
    ProgrammaticWorkflow,
    RunResult,
    SupervisedProcess,
    load_workflow,
    plan_programmatic_workflow,
    plan_workflow,
    run_captured_process,
    run_programmatic_workflow,
    run_workflow,
)
from mammoth.workflow.config import StepConfig
from mammoth.workflow.launch import launch_process, terminate_process_tree
from mammoth.workflow.runner import _complete_run_results, _execute_step


def write_workflow(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_strict_yaml_defaults_and_dependency_order(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "defaults": {"on_failure": "continue", "timeout_seconds": 10},
            "runs": {
                "alpha": {
                    "steps": {
                        "final": {"command": ["final"], "needs": ["prepare"]},
                        "prepare": {"command": ["prepare"]},
                    }
                }
            },
        },
    )

    workflow = load_workflow(path)
    ordered = workflow.runs[0].ordered_steps()

    assert [step.name for step in ordered] == ["prepare", "final"]
    assert ordered[1].on_failure == "continue"
    assert ordered[1].timeout_seconds == 10.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"schema_version": 1, "surprise": True, "runs": {}},
            "Unknown keys in workflow",
        ),
        (
            {
                "schema_version": 1,
                "runs": {
                    "run": {
                        "steps": {
                            "one": {"command": ["one"], "needs": ["missing"]}
                        }
                    }
                },
            },
            "missing dependencies",
        ),
        (
            {
                "schema_version": 1,
                "runs": {
                    "run": {
                        "steps": {
                            "one": {"command": ["one"], "needs": ["two"]},
                            "two": {"command": ["two"], "needs": ["one"]},
                        }
                    }
                },
            },
            "contains a cycle",
        ),
    ],
)
def test_workflow_validation_rejects_unknown_missing_and_cyclic_config(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_workflow(write_workflow(tmp_path, payload))


def test_selected_step_includes_transitive_dependencies(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "one": {"command": ["one"]},
                        "two": {"command": ["two"], "needs": ["one"]},
                        "three": {"command": ["three"], "needs": ["two"]},
                        "unrelated": {"command": ["other"]},
                    }
                }
            },
        },
    )

    plans = plan_workflow(load_workflow(path), selected_steps=["three"])

    assert [plan.step_name for plan in plans] == ["one", "two", "three"]


def test_schema_planning_does_not_inspect_current_artifact_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow(
        write_workflow(
            tmp_path,
            {
                "schema_version": 1,
                "runs": {"alpha": {"steps": {"one": {"command": ["one"]}}}},
            },
        )
    )
    conflicting_run_path = tmp_path / "alpha"
    conflicting_run_path.write_text("not an artifact directory", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    plans = plan_workflow(workflow)

    assert [(plan.run_name, plan.step_name) for plan in plans] == [("alpha", "one")]


def test_direct_step_config_rejects_string_commands_and_freezes_tokens() -> None:
    with pytest.raises(ValueError, match="sequence of arguments"):
        StepConfig("one", "echo hello")

    command = ["echo", "hello"]
    step = StepConfig("one", command)
    command.append("later")

    assert step.command == ("echo", "hello")


def test_direct_step_config_rejects_non_path_cwd_before_planning() -> None:
    with pytest.raises(ValueError, match="cwd must be a path-like value"):
        StepConfig("one", ("echo", "hello"), cwd=42)  # type: ignore[arg-type]


def test_empty_schema_selectors_retain_side_effect_free_planning_and_dry_run(
    tmp_path: Path,
) -> None:
    workflow = load_workflow(
        write_workflow(
            tmp_path,
            {
                "schema_version": 1,
                "runs": {"alpha": {"steps": {"one": {"command": ["one"]}}}},
            },
        )
    )

    assert plan_workflow(workflow, selected_runs=()) == ()
    assert plan_workflow(workflow, selected_steps=()) == ()
    no_runs = run_workflow(workflow, entry=tmp_path / "no-runs", selected_runs=())
    no_steps = run_workflow(
        workflow,
        entry=tmp_path / "no-steps",
        selected_steps=(),
        dry_run=True,
    )

    assert no_runs.successful and no_runs.runs == ()
    assert no_steps.successful
    assert [(run.run_name, run.outcome) for run in no_steps.runs] == [("alpha", "dry-run")]
    assert not (tmp_path / "no-runs").exists()
    assert not (tmp_path / "no-steps").exists()


def test_dry_run_builds_torchrun_command_without_artifacts(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "distributed": {
                    "steps": {
                        "train": {
                            "command": ["project-train", "--opaque"],
                            "launcher": "torchrun",
                            "processes": 3,
                        }
                    }
                }
            },
        },
    )
    entry = tmp_path / "never-created"

    result = run_workflow(load_workflow(path), entry=entry, dry_run=True)

    assert result.successful
    assert not entry.exists()
    assert result.plans[0].command[:6] == (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "3",
    )


def test_torchrun_workflow_children_join_runner_execution_with_rank_streams(
    tmp_path: Path,
) -> None:
    child = tmp_path / "distributed_child.py"
    child.write_text(
        """from __future__ import annotations

import os
from pathlib import Path

from mammoth.torch import (
    ExecutionRequest,
    RuntimeConfig,
    initialize_runtime,
)

phase = os.environ["MAMMOTH_PHASE"]
with initialize_runtime(
    RuntimeConfig(strategy="ddp", device="cpu", backend="gloo")
) as runtime:
    bundle = runtime.start_execution(
        ExecutionRequest(
            run_dir=Path(os.environ["RUN_DIR"]),
            run_name=os.environ["MAMMOTH_RUN_NAME"],
            invocation_kind="workflow-child",
            intended_phases=(phase,),
            command=("python", "distributed_child.py"),
        )
    )
    bundle.observer.emit("process_started", phase=phase)
    bundle.observer.emit("process_completed", phase=phase, exit_code=0)
""",
        encoding="utf-8",
    )
    entry = tmp_path / "runs"
    run_dir = entry / "distributed"
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "distributed": {
                    "environment": {"RUN_DIR": str(run_dir)},
                    "steps": {
                        "train": {
                            "command": [str(child)],
                            "launcher": "torchrun",
                            "processes": 2,
                        }
                    },
                }
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=entry)

    assert result.successful
    execution_id = result.runs[0].execution_id
    assert execution_id is not None
    execution_dir = run_dir / "logs" / "executions" / execution_id
    for rank in range(2):
        assert (execution_dir / f"rank-{rank}.log").is_file()
        events = read_execution_events(execution_dir / f"rank-{rank}.jsonl")
        assert [event.event for event in events] == [
            "process_started",
            "process_completed",
        ]
        assert all(event.world_size == 2 for event in events)


def test_local_workflow_inherits_explicit_environment_and_writes_runner_events(
    tmp_path: Path,
) -> None:
    child_code = (
        "import os; from pathlib import Path; "
        "Path('child.json').write_text(__import__('json').dumps({"
        "'custom': os.environ['CUSTOM'], "
        "'mammoth': os.environ['MAMMOTH_EXECUTION_ID'], "
        "'caller_alias': os.environ.get('CALLER_EXECUTION_ID'), "
        "'phase': os.environ['MAMMOTH_PHASE']}))"
    )
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "project-a": {
                    "environment": {"CUSTOM": "run-value", "SECRET_TOKEN": "do-not-persist"},
                    "steps": {
                        "opaque-step": {
                            "command": [sys.executable, "-c", child_code],
                            "cwd": ".",
                            "environment": {"CUSTOM": "step-value"},
                        }
                    },
                }
            },
        },
    )
    entry = tmp_path / "runs"

    result = run_workflow(
        load_workflow(path),
        entry=entry,
        invocation_command=("mammoth", "workflow", "run", str(path)),
        base_environment={},
    )

    assert result.successful
    child = json.loads((tmp_path / "child.json").read_text())
    assert child["custom"] == "step-value"
    assert child["mammoth"] == result.runs[0].execution_id
    assert child["caller_alias"] is None
    assert child["phase"] == "opaque-step"
    layout = RunLayout(entry, "project-a")
    execution_dir = layout.execution_dir(result.runs[0].execution_id or "missing")
    events = read_execution_events(execution_dir / "runner.jsonl")
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_completed",
        "phase_completed",
        "execution_completed",
    ]
    assert "do-not-persist" not in (execution_dir / "execution.json").read_text()


def test_programmatic_workflow_interleaves_runs_with_independent_attempts(tmp_path: Path) -> None:
    order_path = tmp_path / "dispatch-order.txt"

    def command(label: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"Path({str(order_path)!r}).open('a', encoding='utf-8').write({label!r} + '\\n')",
        )

    def run(name: str, entry: Path) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(entry, name),
            steps=(
                StepConfig("one", command(f"{name}.one")),
                StepConfig("two", command(f"{name}.two"), needs=("one",)),
            ),
            execution=ExecutionInputs(
                invocation_kind="project-workflow",
                command=("project", "run"),
                config_reference="project-runset.yaml",
            ),
        )

    workflow = ProgrammaticWorkflow(
        runs=(run("alpha", tmp_path / "entry-a"), run("beta", tmp_path / "entry-b")),
        dispatch=(
            DispatchEntry("alpha", "one"),
            DispatchEntry("beta", "one"),
            DispatchEntry("alpha", "two"),
            DispatchEntry("beta", "two"),
        ),
    )

    result = run_programmatic_workflow(workflow, base_environment={})

    assert result.successful
    assert order_path.read_text(encoding="utf-8").splitlines() == [
        "alpha.one",
        "beta.one",
        "alpha.two",
        "beta.two",
    ]
    assert [(item.run_name, item.step.name) for item in result.dispatch] == [
        ("alpha", "one"),
        ("beta", "one"),
        ("alpha", "two"),
        ("beta", "two"),
    ]
    alpha = result.run("alpha")
    beta = result.run("beta")
    assert alpha.execution_id is not None and beta.execution_id is not None
    assert alpha.execution_id != beta.execution_id
    assert (tmp_path / "entry-a" / "alpha" / "logs" / "executions" / alpha.execution_id).is_dir()
    assert (tmp_path / "entry-b" / "beta" / "logs" / "executions" / beta.execution_id).is_dir()
    for run_result, entry in ((alpha, tmp_path / "entry-a"), (beta, tmp_path / "entry-b")):
        execution_dir = RunLayout(entry, run_result.run_name).execution_dir(
            run_result.execution_id or "missing"
        )
        events = read_execution_events(execution_dir / "runner.jsonl")
        assert events[0].event == "execution_started"
        assert events[-1].event == "execution_completed"
        assert [event.event for event in events].count("phase_started") == 2
        assert [event.event for event in events].count("task_started") == 2
        assert [event.event for event in events].count("task_completed") == 2


def test_programmatic_lifecycle_fields_enrich_interleaved_run_events(tmp_path: Path) -> None:
    def run(name: str) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(tmp_path / "runs", name),
            steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
            execution=ExecutionInputs("project", ("project", "run")),
            lifecycle_fields={"campaign": "summer", "runset_path": f"{name}.yaml"},
        )

    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run("alpha"), run("beta")),
            dispatch=(DispatchEntry("alpha", "one"), DispatchEntry("beta", "one")),
        ),
        base_environment={},
    )

    assert result.successful
    for name in ("alpha", "beta"):
        execution_id = result.run(name).execution_id
        assert execution_id is not None
        events = read_execution_events(
            RunLayout(tmp_path / "runs", name).execution_dir(execution_id) / "runner.jsonl"
        )
        assert [event.event for event in events] == [
            "execution_started",
            "phase_started",
            "task_started",
            "task_completed",
            "phase_completed",
            "execution_completed",
        ]
        assert all(event.extensions["campaign"] == "summer" for event in events)
        assert all(event.extensions["runset_path"] == f"{name}.yaml" for event in events)


def test_programmatic_lifecycle_provider_receives_immutable_context_and_real_pid(
    tmp_path: Path,
) -> None:
    contexts: list[LifecycleEventContext] = []

    def enrich(context: LifecycleEventContext) -> dict[str, object]:
        contexts.append(context)
        assert not hasattr(context, "observer")
        assert not hasattr(context, "lease")
        assert not hasattr(context, "process")
        return {"caller_event": context.event, "provider_pid": context.child_pid}

    command = (sys.executable, "-c", "raise SystemExit(0)", "--api-token", "secret")
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", command),),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            lifecycle_field_provider=enrich,
        ),
        base_environment={},
    )

    assert result.successful
    task_context = next(context for context in contexts if context.event == "task_started")
    assert task_context.step == run.steps[0]
    assert task_context.command == (*command[:-1], "<redacted>")
    assert task_context.child_pid is not None
    with pytest.raises(AttributeError):
        cast(Any, task_context).child_pid = 0

    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    task_started = next(event for event in events if event.event == "task_started")
    assert task_started.extensions["child_pid"] == task_context.child_pid
    assert task_started.extensions["provider_pid"] == task_context.child_pid
    assert all(event.extensions["caller_event"] == event.event for event in events)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"phase": "caller-owned"}, "Mammoth-owned"),
        ({"metrics": {"score": 1.0}}, "Mammoth-owned"),
        ({"media": {"preview": "image"}}, "Mammoth-owned"),
        ({"logical_step": 1}, "Mammoth-owned"),
        ({"context": object()}, "JSON-compatible"),
        ({"api_token": "secret"}, "credentials or unsafe"),
    ],
)
def test_programmatic_lifecycle_fields_reject_unsafe_values_before_side_effects(
    tmp_path: Path,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProgrammaticRun(
            name="alpha",
            layout=RunLayout(tmp_path / "runs", "alpha"),
            steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
            execution=ExecutionInputs("project", ("project", "run")),
            lifecycle_fields=fields,
        )

    assert not (tmp_path / "runs").exists()


def test_programmatic_run_retains_existing_positional_constructor_order(tmp_path: Path) -> None:
    run = ProgrammaticRun(
        "alpha",
        RunLayout(tmp_path / "runs", "alpha"),
        (StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        ExecutionInputs("project", ("project", "run")),
        {},
        True,
    )

    assert run.resolve_previous_execution is True
    assert run.lifecycle_fields == {}


def test_programmatic_invalid_dynamic_lifecycle_fields_reap_started_child(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    child = (
        "import time; from pathlib import Path; "
        "time.sleep(1); "
        f"Path({str(marker)!r}).touch()"
    )
    launched_pids: list[int] = []

    def invalid_task_start_fields(context: LifecycleEventContext) -> dict[str, object]:
        if context.event == "task_started":
            assert context.child_pid is not None
            launched_pids.append(context.child_pid)
            return {"metrics": {"score": 1.0}}
        return {}

    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", child)),),
        execution=ExecutionInputs("project", ("project", "run")),
        lifecycle_fields={"campaign": "summer"},
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            lifecycle_field_provider=invalid_task_start_fields,
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "failed"
    assert "lifecycle event field provider failed" in (result.step("alpha", "one").reason or "")
    assert launched_pids
    with pytest.raises(ProcessLookupError):
        os.kill(launched_pids[0], 0)
    assert not marker.exists()
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
        "execution_failed",
    ]
    assert all(event.extensions["campaign"] == "summer" for event in events)
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_enriched_task_start_signal_keeps_lifecycle_paired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "import time; time.sleep(60)")),),
        execution=ExecutionInputs("project", ("project", "run")),
        lifecycle_fields={"campaign": "summer"},
    )
    original_emit = RunObserver.emit
    sent = False

    def terminate_after_task_start(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        original_emit(observer, event, **fields)
        if event == "task_started" and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_after_task_start,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
        "execution_interrupted",
    ]
    task_started = next(event for event in events if event.event == "task_started")
    child_pid = task_started.extensions["child_pid"]
    assert isinstance(child_pid, int)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert all(event.extensions["campaign"] == "summer" for event in events)


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt(), SystemExit("provider exited"), BaseException("provider escaped")],
)
def test_programmatic_provider_base_exception_is_structured_failure(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    marker = tmp_path / "must-not-run"

    def fail_at_phase_start(context: LifecycleEventContext) -> None:
        if context.event == "phase_started":
            raise failure
        return None

    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig(
                "one",
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
            ),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            lifecycle_field_provider=fail_at_phase_start,
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "failed"
    assert "lifecycle event field provider failed" in (result.step("alpha", "one").reason or "")
    assert not marker.exists()
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "phase_failed",
        "execution_failed",
    ]
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_execution_start_provider_failure_is_a_structured_result(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-run"

    def fail_at_execution_start(context: LifecycleEventContext) -> None:
        if context.event == "execution_started":
            raise RuntimeError("startup provider exploded")
        return None

    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig(
                "one",
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
            ),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            lifecycle_field_provider=fail_at_execution_start,
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "failed"
    assert "lifecycle event field provider failed" in (result.step("alpha", "one").reason or "")
    assert not marker.exists()
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == ["execution_started", "execution_failed"]
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_execution_terminal_provider_failure_keeps_its_reason(
    tmp_path: Path,
) -> None:
    def fail_at_execution_terminal(context: LifecycleEventContext) -> dict[str, object]:
        if context.event == "execution_completed":
            raise RuntimeError("terminal provider exploded")
        return {}

    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            lifecycle_field_provider=fail_at_execution_terminal,
        ),
        base_environment={},
    )

    run_result = result.run("alpha")
    assert run_result.outcome == "failed"
    assert run_result.reason == "lifecycle event field provider failed: terminal provider exploded"
    assert result.step("alpha", "one").outcome == "completed"
    execution_id = run_result.execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert events[-1].event == "execution_failed"
    assert events[-1].message == run_result.reason


def test_schema_v1_workflow_events_remain_unenriched(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "alpha": {
                    "steps": {
                        "one": {"command": [sys.executable, "-c", "raise SystemExit(0)"]}
                    }
                }
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs", base_environment={})

    assert result.successful
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert all(event.extensions == {} for event in events)


@pytest.mark.parametrize(
    ("dispatch", "message"),
    [
        ((DispatchEntry("unknown", "one"),), "unknown run"),
        ((DispatchEntry("alpha", "one"), DispatchEntry("alpha", "one")), "duplicates"),
        ((DispatchEntry("alpha", "two"), DispatchEntry("alpha", "one")), "before dependencies"),
        ((DispatchEntry("alpha", "one"),), "omits selected steps"),
    ],
)
def test_programmatic_plan_validation_rejects_before_artifacts(
    tmp_path: Path,
    dispatch: tuple[DispatchEntry, ...],
    message: str,
) -> None:
    entry = tmp_path / "must-not-exist"
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(entry, "alpha"),
        steps=(
            StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),
            StepConfig("two", (sys.executable, "-c", "raise SystemExit(0)"), needs=("one",)),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    workflow = ProgrammaticWorkflow(runs=(run,), dispatch=dispatch)

    with pytest.raises(ValueError, match=message):
        plan_programmatic_workflow(workflow)

    assert not entry.exists()


def test_programmatic_dry_run_returns_global_plan_without_creating_entry(tmp_path: Path) -> None:
    entry = tmp_path / "must-not-exist"
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(entry, "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    workflow = ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),))

    result = run_programmatic_workflow(workflow, dry_run=True, base_environment={})

    assert result.successful
    assert [(plan.run_name, plan.step_name) for plan in result.plans] == [("alpha", "one")]
    assert not entry.exists()


def test_programmatic_plan_rejects_existing_file_as_layout_entry(tmp_path: Path) -> None:
    entry = tmp_path / "not-a-directory"
    entry.write_text("file", encoding="utf-8")
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(entry, "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    with pytest.raises(ValueError, match="layout path must be a directory"):
        plan_programmatic_workflow(
            ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),))
        )


def test_programmatic_first_failure_blocks_later_cross_run_dispatch(tmp_path: Path) -> None:
    def run(name: str, first: tuple[str, ...]) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(tmp_path / "runs", name),
            steps=(
                StepConfig("one", first),
                StepConfig("two", (sys.executable, "-c", "raise SystemExit(0)"), needs=("one",)),
            ),
            execution=ExecutionInputs("project", ("project", "run")),
        )

    workflow = ProgrammaticWorkflow(
        runs=(
            run("alpha", (sys.executable, "-c", "raise SystemExit(4)")),
            run("beta", (sys.executable, "-c", "raise SystemExit(0)")),
        ),
        dispatch=(
            DispatchEntry("alpha", "one"),
            DispatchEntry("beta", "one"),
            DispatchEntry("alpha", "two"),
            DispatchEntry("beta", "two"),
        ),
    )

    result = run_programmatic_workflow(workflow, base_environment={})

    assert not result.successful
    assert result.run("alpha").outcome == "failed"
    assert result.run("beta").outcome == "blocked"
    assert [(item.run_name, item.step.outcome) for item in result.dispatch] == [
        ("alpha", "failed"),
        ("beta", "skipped"),
        ("alpha", "skipped"),
        ("beta", "skipped"),
    ]
    assert result.step("beta", "one").reason == "blocked after workflow failure alpha/one"
    beta_execution = result.run("beta").execution_id
    assert beta_execution is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "beta").execution_dir(beta_execution) / "runner.jsonl"
    )
    assert "task_skipped" in [event.event for event in events]
    assert "phase_skipped" in [event.event for event in events]
    assert events[-1].event == "execution_failed"


def test_programmatic_run_major_dispatch_completes_before_later_setup_failure(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    marker = tmp_path / "alpha-completed"

    def run(name: str) -> ProgrammaticRun:
        command = (
            sys.executable,
            "-c",
            (
                f"from pathlib import Path; Path({str(marker)!r}).touch()"
                if name == "alpha"
                else "raise SystemExit(0)"
            ),
        )
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(entry, name),
            steps=(StepConfig("one", command),),
            execution=ExecutionInputs("project", ("project", "run")),
        )

    workflow = ProgrammaticWorkflow(
        runs=(run("alpha"), run("beta")),
        dispatch=(DispatchEntry("alpha", "one"), DispatchEntry("beta", "one")),
    )
    beta_layout = RunLayout(entry, "beta")
    with claim_logical_run_lease(beta_layout.run_dir), pytest.raises(
        RuntimeError,
        match="already active",
    ):
        run_programmatic_workflow(workflow, base_environment={})

    execution_dir = next((entry / "alpha" / "logs" / "executions").iterdir())
    events = read_execution_events(execution_dir / "runner.jsonl")
    assert marker.is_file()
    assert events[-1].event == "execution_completed"


def test_programmatic_startup_sigterm_finalizes_registered_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_context = create_execution_context

    def terminate_after_context(*args: Any, **kwargs: Any) -> Any:
        context = original_context(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return context

    monkeypatch.setattr(
        "mammoth.workflow.runner.create_execution_context",
        terminate_after_context,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert events[-1].event == "execution_interrupted"
    assert events[-1].signal == signal.SIGTERM
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_sigterm_between_start_events_closes_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_emit = RunObserver.emit
    sent = False

    def terminate_after_phase_start(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        original_emit(observer, event, **fields)
        if event == "phase_started" and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_after_phase_start,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    assert result.step("alpha", "one").outcome == "interrupted"
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
        "execution_interrupted",
    ]


@pytest.mark.parametrize(
    ("trigger_event", "expected_events"),
    [
        ("execution_started", ["execution_started", "execution_interrupted"]),
        (
            "phase_started",
            [
                "execution_started",
                "phase_started",
                "task_started",
                "task_failed",
                "phase_failed",
                "execution_interrupted",
            ],
        ),
    ],
)
def test_programmatic_sigterm_before_start_event_keeps_lifecycle_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger_event: str,
    expected_events: list[str],
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_emit = RunObserver.emit
    sent = False

    def terminate_before_start_event(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        if event == trigger_event and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        original_emit(observer, event, **fields)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_before_start_event,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == expected_events


def test_programmatic_finalization_sigterm_returns_results_and_releases_all_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(name: str) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(tmp_path / "runs", name),
            steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
            execution=ExecutionInputs("project", ("project", "run")),
        )

    original_close = RunObserver.close
    sent = False

    def terminate_during_close(observer: RunObserver) -> None:
        nonlocal sent
        original_close(observer)
        if not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.close",
        terminate_during_close,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run("alpha"), run("beta")),
            dispatch=(DispatchEntry("alpha", "one"), DispatchEntry("beta", "one")),
        ),
        base_environment={},
    )

    assert result.successful
    assert [run.outcome for run in result.runs] == ["completed", "completed"]
    for name in ("alpha", "beta"):
        with claim_logical_run_lease(RunLayout(tmp_path / "runs", name).run_dir):
            pass


def test_programmatic_final_result_completion_defers_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    def terminate_before_result_completion(*args: Any, **kwargs: Any) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        return _complete_run_results(*args, **kwargs)

    monkeypatch.setattr(
        "mammoth.workflow.runner._complete_run_results",
        terminate_before_result_completion,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.successful
    assert result.run("alpha").outcome == "completed"


def test_programmatic_interruption_reports_unstarted_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(name: str) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(tmp_path / "runs", name),
            steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
            execution=ExecutionInputs("project", ("project", "run")),
        )

    original_emit = RunObserver.emit
    sent = False

    def terminate_after_task_completion(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        original_emit(observer, event, **fields)
        if event == "task_completed" and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_after_task_completion,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run("alpha"), run("beta")),
            dispatch=(DispatchEntry("alpha", "one"), DispatchEntry("beta", "one")),
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    assert result.step("alpha", "one").outcome == "completed"
    assert result.run("beta") == RunResult("beta", "interrupted", None, ())
    assert not (tmp_path / "runs" / "beta").exists()
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_completed",
        "phase_completed",
        "execution_interrupted",
    ]


def test_programmatic_post_step_signal_preserves_dispatch_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(name: str) -> ProgrammaticRun:
        return ProgrammaticRun(
            name=name,
            layout=RunLayout(tmp_path / "runs", name),
            steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
            execution=ExecutionInputs("project", ("project", "run")),
        )

    def terminate_after_step(*args: Any, **kwargs: Any) -> Any:
        result = _execute_step(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(
        "mammoth.workflow.runner._execute_step",
        terminate_after_step,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run("alpha"), run("beta")),
            dispatch=(DispatchEntry("alpha", "one"), DispatchEntry("beta", "one")),
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    assert result.step("alpha", "one").outcome == "completed"
    assert [(item.run_name, item.step.outcome) for item in result.dispatch] == [
        ("alpha", "completed"),
    ]
    assert result.run("beta") == RunResult("beta", "interrupted", None, ())
    assert not (tmp_path / "runs" / "beta").exists()


@pytest.mark.parametrize(
    ("trigger_event", "command", "expected_step", "task_event", "phase_event"),
    [
        (
            "phase_completed",
            (sys.executable, "-c", "raise SystemExit(0)"),
            "completed",
            "task_completed",
            "phase_completed",
        ),
        (
            "task_failed",
            (sys.executable, "-c", "raise SystemExit(3)"),
            "failed",
            "task_failed",
            "phase_failed",
        ),
        (
            "phase_failed",
            (sys.executable, "-c", "raise SystemExit(3)"),
            "failed",
            "task_failed",
            "phase_failed",
        ),
    ],
)
def test_programmatic_terminal_event_signal_keeps_paired_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger_event: str,
    command: tuple[str, ...],
    expected_step: str,
    task_event: str,
    phase_event: str,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", command),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_emit = RunObserver.emit
    sent = False

    def terminate_after_terminal_event(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        original_emit(observer, event, **fields)
        if event == trigger_event and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_after_terminal_event,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    assert result.step("alpha", "one").outcome == expected_step
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        task_event,
        phase_event,
        "execution_interrupted",
    ]


@pytest.mark.parametrize("trigger_event", ["task_skipped", "phase_skipped"])
def test_programmatic_skip_signal_keeps_paired_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger_event: str,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig("fail", (sys.executable, "-c", "raise SystemExit(3)")),
            StepConfig("later", (sys.executable, "-c", "raise SystemExit(0)")),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_emit = RunObserver.emit
    sent = False

    def terminate_after_skip_event(
        observer: RunObserver,
        event: Any,
        **fields: Any,
    ) -> None:
        nonlocal sent
        original_emit(observer, event, **fields)
        if event == trigger_event and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(
        "mammoth.workflow.runner.RunObserver.emit",
        terminate_after_skip_event,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "fail"), DispatchEntry("alpha", "later")),
        ),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    assert result.step("alpha", "later").outcome == "skipped"
    execution_id = result.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
        "task_skipped",
        "phase_skipped",
        "execution_interrupted",
    ]


def test_programmatic_pre_dispatch_hook_and_metadata_sanitization(tmp_path: Path) -> None:
    order_path = tmp_path / "order.txt"
    child_path = tmp_path / "child.txt"
    captured: list[tuple[str, str, str]] = []
    runtime_metadata: dict[str, object] = {
        "api_token": "runtime-secret",
        "nested": {"value": 1},
    }

    def hook(context: PreDispatchContext) -> None:
        captured.append(
            (
                context.run.name,
                context.step.name,
                context.execution.metadata.execution_id,
            )
        )
        order_path.write_text("hook", encoding="utf-8")

    child = (
        "import os; from pathlib import Path; "
        f"Path({str(child_path)!r}).write_text("
        "os.environ['API_TOKEN'] + ':' + os.environ['MAMMOTH_INVOCATION_KIND'], "
        "encoding='utf-8'); "
        f"Path({str(order_path)!r}).write_text(Path({str(order_path)!r}).read_text() + ',child')"
    )
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig(
                "one",
                (sys.executable, "-c", child),
                environment={"API_TOKEN": "only-for-child"},
            ),
        ),
        execution=ExecutionInputs(
            "project",
            ("project", "run"),
            config_reference="https://example.invalid/run?token=metadata-secret",
            runtime=runtime_metadata,
        ),
    )
    workflow = ProgrammaticWorkflow(
        runs=(run,),
        dispatch=(DispatchEntry("alpha", "one"),),
        pre_dispatch=hook,
    )
    nested = runtime_metadata["nested"]
    assert isinstance(nested, dict)
    nested["value"] = 2

    result = run_programmatic_workflow(workflow, base_environment={})

    assert result.successful
    assert captured == [("alpha", "one", result.run("alpha").execution_id or "")]
    assert order_path.read_text(encoding="utf-8") == "hook,child"
    assert child_path.read_text(encoding="utf-8") == "only-for-child:project"
    metadata = json.loads(
        RunLayout(tmp_path / "runs", "alpha")
        .execution_dir(result.run("alpha").execution_id or "missing")
        .joinpath("execution.json")
        .read_text(encoding="utf-8")
    )
    assert "only-for-child" not in json.dumps(metadata)
    assert "metadata-secret" not in json.dumps(metadata)
    assert "runtime-secret" not in json.dumps(metadata)
    assert metadata["runtime"]["nested"]["value"] == 1


def test_programmatic_hook_failure_and_sigterm_release_attempt_resources(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-run"
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig(
                "one",
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
            ),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    def fail_hook(_context: object) -> None:
        raise RuntimeError("hook exploded")

    failed = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            pre_dispatch=fail_hook,
        ),
        base_environment={},
    )
    assert failed.run("alpha").outcome == "failed"
    assert "pre-dispatch hook failed" in (failed.step("alpha", "one").reason or "")
    assert not marker.exists()

    def terminate_hook(_context: object) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    interrupted = run_programmatic_workflow(
        ProgrammaticWorkflow(
            runs=(run,),
            dispatch=(DispatchEntry("alpha", "one"),),
            pre_dispatch=terminate_hook,
        ),
        base_environment={},
    )
    assert interrupted.run("alpha").outcome == "interrupted"
    execution_id = interrupted.run("alpha").execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "alpha").execution_dir(execution_id) / "runner.jsonl"
    )
    assert events[-1].event == "execution_interrupted"
    assert events[-1].signal == signal.SIGTERM


def test_programmatic_sigint_reaps_active_child_and_releases_lease(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = "\n".join(
        (
            "import os",
            "import signal",
            "import time",
            "from pathlib import Path",
            f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()), encoding='utf-8')",
            "os.kill(os.getppid(), signal.SIGINT)",
            "time.sleep(60)",
        )
    )
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(
            StepConfig(
                "one",
                (
                    sys.executable,
                    "-c",
                    child_code,
                ),
            ),
        ),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    # Acquiring the lease again proves executor cleanup completed after SIGINT.
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_second_sigterm_does_not_interrupt_child_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = "\n".join(
        (
            "import os",
            "import signal",
            "import time",
            "from pathlib import Path",
            f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()), encoding='utf-8')",
            "os.kill(os.getppid(), signal.SIGTERM)",
            "time.sleep(60)",
        )
    )
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", child_code)),),
        execution=ExecutionInputs("project", ("project", "run")),
    )

    def send_second_signal(*args: Any, **kwargs: Any) -> int | None:
        os.kill(os.getpid(), signal.SIGTERM)
        return terminate_process_tree(*args, **kwargs)

    monkeypatch.setattr(
        "mammoth.workflow.launch.terminate_process_tree",
        send_second_signal,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha").outcome == "interrupted"
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    with claim_logical_run_lease(RunLayout(tmp_path / "runs", "alpha").run_dir):
        pass


def test_programmatic_signal_during_handler_installation_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ProgrammaticRun(
        name="alpha",
        layout=RunLayout(tmp_path / "runs", "alpha"),
        steps=(StepConfig("one", (sys.executable, "-c", "raise SystemExit(0)")),),
        execution=ExecutionInputs("project", ("project", "run")),
    )
    original_signal = signal.signal
    original_handler = signal.getsignal(signal.SIGTERM)
    sent = False

    def install_then_terminate(
        signal_number: signal.Signals,
        handler: Any,
    ) -> Any:
        nonlocal sent
        previous = original_signal(signal_number, handler)
        if signal_number == signal.SIGTERM and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return previous

    monkeypatch.setattr(
        "mammoth.workflow.runner.signal.signal",
        install_then_terminate,
    )
    result = run_programmatic_workflow(
        ProgrammaticWorkflow(runs=(run,), dispatch=(DispatchEntry("alpha", "one"),)),
        base_environment={},
    )

    assert result.run("alpha") == RunResult("alpha", "interrupted", None, ())
    assert signal.getsignal(signal.SIGTERM) == original_handler
    assert not (tmp_path / "runs").exists()


def test_stop_policy_skips_remaining_steps_after_failure(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "fail": {"command": [sys.executable, "-c", "raise SystemExit(3)"]},
                        "later": {"command": [sys.executable, "-c", "raise SystemExit(0)"]},
                    }
                }
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs")

    assert not result.successful
    assert result.runs[0].outcome == "failed"
    assert [(step.name, step.outcome) for step in result.runs[0].steps] == [
        ("fail", "failed"),
        ("later", "skipped"),
    ]


def test_launch_failure_becomes_a_step_result_instead_of_crashing(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "missing": {"command": ["definitely-not-a-real-command-mammoth"]}
                    }
                }
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs")

    assert result.runs[0].outcome == "failed"
    assert result.runs[0].steps[0].outcome == "failed"
    assert "could not launch" in (result.runs[0].steps[0].reason or "")
    execution_id = result.runs[0].execution_id
    assert execution_id is not None
    events = read_execution_events(
        RunLayout(tmp_path / "runs", "run").execution_dir(execution_id) / "runner.jsonl"
    )
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
        "execution_failed",
    ]
    assert all(event.extensions == {} for event in events)


def test_continue_and_run_dependency_policy_executes_after_failure(tmp_path: Path) -> None:
    marker = tmp_path / "continued"
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "fail": {
                            "command": [sys.executable, "-c", "raise SystemExit(2)"],
                            "on_failure": "continue",
                        },
                        "recover": {
                            "command": [
                                sys.executable,
                                "-c",
                                f"from pathlib import Path; Path({str(marker)!r}).touch()",
                            ],
                            "needs": ["fail"],
                            "dependency_failure": "run",
                        },
                    }
                }
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs")

    assert not result.successful
    assert marker.is_file()
    assert [step.outcome for step in result.runs[0].steps] == ["failed", "completed"]


def test_yaml_workflow_continue_policy_runs_later_run_after_interruption(tmp_path: Path) -> None:
    marker = tmp_path / "later-run-completed"
    interrupt = (
        "import os, signal, time; "
        "os.kill(os.getppid(), signal.SIGINT); "
        "time.sleep(60)"
    )
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "alpha": {"steps": {"one": {"command": [sys.executable, "-c", interrupt]}}},
                "beta": {
                    "steps": {
                        "one": {
                            "command": [
                                sys.executable,
                                "-c",
                                f"from pathlib import Path; Path({str(marker)!r}).touch()",
                            ]
                        }
                    }
                },
            },
        },
    )

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs", base_environment={})

    assert result.run("alpha").outcome == "interrupted"
    assert result.run("beta").outcome == "completed"
    assert marker.is_file()


def test_yaml_workflow_resolves_later_run_lineage_at_its_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "alpha-completed"
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "alpha": {
                    "steps": {
                        "one": {
                            "command": [
                                sys.executable,
                                "-c",
                                f"from pathlib import Path; Path({str(marker)!r}).touch()",
                            ]
                        }
                    }
                },
                "beta": {"steps": {"one": {"command": [sys.executable, "-c", "pass"]}}},
            },
        },
    )

    def latest_after_alpha(run_dir: Path) -> str | None:
        if run_dir.name == "beta":
            assert marker.is_file()
        return latest_execution_id(run_dir)

    monkeypatch.setattr(
        "mammoth.workflow.runner.latest_execution_id",
        latest_after_alpha,
    )
    result = run_workflow(load_workflow(path), entry=tmp_path / "runs", base_environment={})

    assert result.successful
    assert marker.is_file()


def test_timeout_terminates_child_process_group(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "slow": {
                            "command": [sys.executable, "-c", "import time; time.sleep(60)"],
                            "timeout_seconds": 0.05,
                        }
                    }
                }
            },
        },
    )
    started = time.monotonic()

    result = run_workflow(load_workflow(path), entry=tmp_path / "runs")

    assert time.monotonic() - started < 2
    process = result.runs[0].steps[0].process
    assert process is not None and process.timed_out
    assert result.runs[0].outcome == "failed"


def test_launch_timeout_includes_started_callback_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []
    clock = [0.0]

    class FakeSupervisedProcess:
        pid = 123
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

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


def test_captured_process_drains_large_both_streams_without_deadlock() -> None:
    result = run_captured_process(
        (
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('out' * 300_000); sys.stderr.write('err' * 300_000)",
        ),
        cwd=None,
        environment={},
        timeout_seconds=5,
    )

    assert result.return_code == 0
    assert result.stdout == "out" * 300_000
    assert result.stderr == "err" * 300_000


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


def test_captured_process_timeout_stops_independent_descendant(tmp_path: Path) -> None:
    worker_pid_path = tmp_path / "captured-worker.pid"
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
            "print('started', flush=True)",
            "while True:",
            "    time.sleep(1)",
        )
    )

    result = run_captured_process(
        (sys.executable, "-c", launcher),
        cwd=None,
        environment={},
        timeout_seconds=0.1,
        terminate_grace_seconds=0.1,
        descendant_grace_seconds=0.1,
    )
    worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))

    assert result.timed_out
    assert result.stdout == "started\n"
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


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


def test_workflow_cli_dry_run_is_side_effect_free(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {"run": {"steps": {"step": {"command": ["echo", "hello"]}}}},
        },
    )
    entry = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        ["workflow", "run", str(path), "--entry", str(entry), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "run/step: echo hello" in result.output
    assert not entry.exists()


def test_workflow_cli_failure_returns_nonzero_without_traceback(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "run": {
                    "steps": {
                        "step": {"command": [sys.executable, "-c", "raise SystemExit(7)"]}
                    }
                }
            },
        },
    )

    result = CliRunner().invoke(
        app,
        ["workflow", "run", str(path), "--entry", str(tmp_path / "runs")],
    )

    assert result.exit_code == 1
    assert "run: failed execution=" in result.output
    assert "Traceback" not in result.output
