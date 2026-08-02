"""Exercise the public Typer application and both supported entry-point paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mammoth import __version__
from mammoth.cli import app

runner = CliRunner()


def write_workflow(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"{__version__}\n"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["monitor", "--help"],
        ["workflow", "--help"],
        ["workflow", "run", "--help"],
    ],
)
def test_cli_help_for_root_and_commands(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Traceback" not in result.output


def test_monitor_rejects_missing_entry_without_traceback() -> None:
    result = runner.invoke(app, ["monitor", "run"])

    assert result.exit_code == 2
    assert "Missing option '--entry'" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("interval", ["0", "-0.5", "nan", "inf", "+inf", "-inf"])
def test_monitor_rejects_non_positive_interval(interval: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "run", "--entry", str(tmp_path), "--interval", interval],
    )

    assert result.exit_code == 2
    assert "value must be finite and greater than zero" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_monitor_rejects_malformed_interval(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "run", "--entry", str(tmp_path), "--interval", "abc"],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--interval'" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_monitor_reports_missing_run_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["monitor", "missing", "--entry", str(tmp_path)])

    assert result.exit_code == 2
    assert "No valid executions found" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_workflow_rejects_missing_file_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            str(tmp_path / "missing.yaml"),
            "--entry",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    ("source", "selectors", "message"),
    [
        ("runs: [\n", [], "while parsing a flow node"),
        ("schema_version: 2\nruns: {}\n", [], "Unsupported workflow schema version"),
        (
            "schema_version: 1\nruns:\n  run:\n    steps:\n      step:\n"
            "        command: [echo]\n",
            ["--run", "missing"],
            "Workflow has no runs: missing",
        ),
        (
            "schema_version: 1\nruns:\n  run:\n    steps:\n      step:\n"
            "        command: [echo]\n",
            ["--step", "missing"],
            "Run 'run' has no steps: missing",
        ),
    ],
)
def test_workflow_rejects_invalid_definition_or_selector_without_traceback(
    source: str,
    selectors: list[str],
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(source)

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            str(path),
            "--entry",
            str(tmp_path / "runs"),
            *selectors,
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert all(fragment in result.output for fragment in message.split())
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_python_module_entrypoint_hides_expected_workflow_tracebacks(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text("schema_version: 2\nruns: {}\n")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mammoth",
            "workflow",
            "run",
            str(path),
            "--entry",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 2
    assert all(fragment in output for fragment in ["Unsupported", "workflow", "schema", "version"])
    assert "Traceback" not in output


def test_workflow_cli_preserves_repeated_run_and_step_selectors(tmp_path: Path) -> None:
    path = write_workflow(
        tmp_path,
        {
            "schema_version": 1,
            "runs": {
                "alpha": {
                    "steps": {
                        "prepare": {"command": ["prepare-alpha"]},
                        "train": {"command": ["train-alpha"], "needs": ["prepare"]},
                        "unused": {"command": ["unused-alpha"]},
                    }
                },
                "beta": {
                    "steps": {
                        "prepare": {"command": ["prepare-beta"]},
                        "train": {"command": ["train-beta"], "needs": ["prepare"]},
                        "unused": {"command": ["unused-beta"]},
                    }
                },
                "unselected": {
                    "steps": {
                        "prepare": {"command": ["prepare-unselected"]},
                        "train": {"command": ["train-unselected"], "needs": ["prepare"]},
                    }
                },
            },
        },
    )
    entry = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            str(path),
            "--entry",
            str(entry),
            "--run",
            "alpha",
            "--run",
            "beta",
            "--step",
            "prepare",
            "--step",
            "train",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "alpha/prepare: prepare-alpha",
        "alpha/train: train-alpha",
        "beta/prepare: prepare-beta",
        "beta/train: train-beta",
    ]
    assert not entry.exists()
