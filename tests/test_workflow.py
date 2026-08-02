from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import RunLayout, read_execution_events
from mammoth.workflow import SupervisedProcess, load_workflow, plan_workflow, run_workflow


def write_workflow(tmp_path: Path, payload: dict) -> Path:
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
    payload: dict,
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


def test_local_workflow_inherits_explicit_environment_and_writes_runner_events(
    tmp_path: Path,
) -> None:
    child_code = (
        "import os; from pathlib import Path; "
        "Path('child.json').write_text(__import__('json').dumps({"
        "'custom': os.environ['CUSTOM'], "
        "'mammoth': os.environ['MAMMOTH_EXECUTION_ID'], "
        "'legacy': os.environ['TISAM_EXECUTION_ID'], "
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
    assert child["mammoth"] == child["legacy"] == result.runs[0].execution_id
    assert child["phase"] == "opaque-step"
    layout = RunLayout(entry, "project-a")
    execution_dir = layout.execution_dir(result.runs[0].execution_id or "missing")
    events = read_execution_events(execution_dir / "runner.jsonl")
    assert [event.event for event in events] == [
        "execution_started",
        "phase_started",
        "phase_completed",
        "execution_completed",
    ]
    assert "do-not-persist" not in (execution_dir / "execution.json").read_text()


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
