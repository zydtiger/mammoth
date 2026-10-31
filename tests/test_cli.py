"""Exercise the public Typer monitor application."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from mammoth import __version__
from mammoth.cli import app
from mammoth.core import (
    GroupEventWriter,
    GroupLayout,
    GroupMember,
    RunLayout,
    create_execution_context,
    publish_group_manifest,
)
from mammoth.core.events import ExecutionEventWriter

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"{__version__}\n"


@pytest.mark.parametrize("arguments", [["--help"], ["monitor", "--help"]])
def test_cli_help_for_root_and_monitor(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Traceback" not in result.output


def test_cli_has_no_workflow_command() -> None:
    result = runner.invoke(app, ["workflow", "--help"])

    assert result.exit_code == 2
    assert "No such command" in result.output


@pytest.mark.parametrize("interval", ["0", "-0.5", "nan", "inf", "+inf", "-inf"])
@pytest.mark.parametrize("option", ["--interval", "--stale-after"])
def test_monitor_rejects_non_positive_duration(
    interval: str,
    option: str,
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        ["monitor", "show", "run", "--entry", str(tmp_path), option, interval],
    )

    assert result.exit_code == 2
    assert "value must be finite and greater than" in result.output
    assert "zero" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_monitor_rejects_malformed_interval(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "show", "run", "--entry", str(tmp_path), "--interval", "abc"],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--interval'" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_monitor_reports_missing_run_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["monitor", "show", "missing", "--entry", str(tmp_path)])

    assert result.exit_code == 2
    assert "No valid executions found" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def create_completed_monitor_run(tmp_path: Path) -> None:
    """Create one terminal execution for monitor CLI behavior tests."""
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name="run",
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("train",),
        execution_id="attempt",
        created_at="2026-01-01T00:00:00Z",
    )
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")


def test_monitor_defaults_entry_to_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_completed_monitor_run(tmp_path / "runs")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["monitor", "show", "run"])

    assert result.exit_code == 0
    assert "Execution: attempt" in result.output


def test_monitor_tui_defaults_to_watch_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_completed_monitor_run(tmp_path)
    run_textual = Mock()
    monkeypatch.setattr(
        "mammoth.cli.load_textual_ui",
        lambda: SimpleNamespace(run_textual=run_textual),
    )

    result = runner.invoke(app, ["monitor", "tui", "run", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert run_textual.call_count == 1
    assert run_textual.call_args.kwargs == {
        "watch": True,
        "telemetry": True,
        "interval_seconds": 2.0,
        "stale_after_seconds": 90.0,
    }


def test_monitor_show_never_loads_the_terminal_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_completed_monitor_run(tmp_path)
    load_textual = Mock()
    monkeypatch.setattr("mammoth.cli.load_textual_ui", load_textual)

    result = runner.invoke(app, ["monitor", "show", "run", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert "Execution: attempt" in result.output
    load_textual.assert_not_called()


def test_monitor_explicit_tui_opt_outs_reach_textual_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_completed_monitor_run(tmp_path)
    run_textual = Mock()
    monkeypatch.setattr(
        "mammoth.cli.load_textual_ui",
        lambda: SimpleNamespace(run_textual=run_textual),
    )

    result = runner.invoke(
        app,
        [
            "monitor",
            "tui",
            "run",
            "--entry",
            str(tmp_path),
            "--no-watch",
            "--no-telemetry",
        ],
    )

    assert result.exit_code == 0
    assert run_textual.call_args.kwargs == {
        "watch": False,
        "telemetry": False,
        "interval_seconds": 2.0,
        "stale_after_seconds": 90.0,
    }


def test_monitor_missing_interactive_dependencies_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_completed_monitor_run(tmp_path)

    def raise_missing() -> None:
        raise ModuleNotFoundError("textual")

    monkeypatch.setattr("mammoth.cli.load_textual_ui", raise_missing)

    result = runner.invoke(app, ["monitor", "tui", "run", "--entry", str(tmp_path)])

    assert result.exit_code == 2
    assert "uv sync --extra monitor" in result.output
    assert "mammoth monitor show" in result.output


def test_monitor_show_fleet_view_lists_groups_and_loose_runs(tmp_path: Path) -> None:
    manifest = publish_group_manifest(
        tmp_path,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
    )
    group_layout = GroupLayout(tmp_path, manifest.group_id)
    writer = GroupEventWriter(group_layout.events_path, group_id=manifest.group_id)
    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("run_completed", run_name="alpha")
    writer.emit("group_completed")
    writer.close()
    RunLayout(tmp_path, "loose").prepare()

    result = runner.invoke(app, ["monitor", "show", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert f"Entry: {tmp_path}" in result.output
    assert "members=1/0/1" in result.output
    assert "terminal=group_completed" in result.output
    assert "loose: pending" in result.output
    assert "Traceback" not in result.output


def test_monitor_fleet_view_reports_useful_output_with_nothing_published(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["monitor", "show", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert "(none observed)" in result.output
    assert "Traceback" not in result.output


def test_monitor_group_view_plain_renders_selected_group(tmp_path: Path) -> None:
    manifest = publish_group_manifest(
        tmp_path,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
    )

    result = runner.invoke(
        app,
        ["monitor", "show", "--entry", str(tmp_path), "--group", manifest.group_id],
    )

    assert result.exit_code == 0
    assert f"Group: {manifest.group_id}" in result.output
    assert "alpha: pending" in result.output


def test_monitor_group_view_rejects_an_unknown_group_id(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "show", "--entry", str(tmp_path), "--group", "does-not-exist"],
    )

    assert result.exit_code == 2
    assert "No group" in result.output
    assert "Traceback" not in result.output


def test_monitor_match_groups_loose_runs_ad_hoc_in_plain_mode(tmp_path: Path) -> None:
    RunLayout(tmp_path, "sweep-a").prepare()
    RunLayout(tmp_path, "sweep-b").prepare()
    RunLayout(tmp_path, "other").prepare()

    result = runner.invoke(
        app,
        ["monitor", "show", "--entry", str(tmp_path), "--match", "sweep-*"],
    )

    assert result.exit_code == 0
    assert "Match: sweep-*" in result.output
    assert "match:sweep-*" in result.output
    assert "Loose runs:" not in result.output


def test_monitor_rejects_group_or_match_combined_with_a_run_name(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "show", "run", "--entry", str(tmp_path), "--group", "some-group"],
    )

    assert result.exit_code == 2
    assert "--group and --match apply only when RUN is" in result.output
    assert "omitted" in result.output
    assert "Traceback" not in result.output


def test_monitor_rejects_execution_without_a_run_name(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["monitor", "show", "--entry", str(tmp_path), "--execution", "bogus"],
    )

    assert result.exit_code == 2
    assert "--execution applies only to a single-run" in result.output
    assert "RUN given" in result.output
    assert "Traceback" not in result.output


def test_monitor_rejects_group_and_match_combined(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "monitor",
            "show",
            "--entry",
            str(tmp_path),
            "--group",
            "some-group",
            "--match",
            "sweep-*",
        ],
    )

    assert result.exit_code == 2
    assert "--group and --match are mutually exclusive" in result.output
    assert "Traceback" not in result.output


def test_monitor_tui_fleet_view_defaults_to_watch_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RunLayout(tmp_path, "loose").prepare()
    run_fleet_textual = Mock()
    monkeypatch.setattr(
        "mammoth.cli.load_textual_ui",
        lambda: SimpleNamespace(run_fleet_textual=run_fleet_textual),
    )

    result = runner.invoke(app, ["monitor", "tui", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert run_fleet_textual.call_count == 1
    kwargs = run_fleet_textual.call_args.kwargs
    assert kwargs["watch"] is True
    assert kwargs["telemetry"] is True
    assert kwargs["interval_seconds"] == 2.0
    assert kwargs["stale_after_seconds"] == 90.0
    assert kwargs["open_group_id"] is None


def test_monitor_single_run_invocation_is_unaffected_by_the_fleet_view(
    tmp_path: Path,
) -> None:
    """Regression: `mammoth monitor show <run_name>` keeps the single-run snapshot."""
    layout = RunLayout(tmp_path, "run").prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name="run",
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("train",),
        execution_id="attempt",
        created_at="2026-01-01T00:00:00Z",
    )
    with ExecutionEventWriter.for_runner(context) as writer:
        writer.emit("execution_started")
        writer.emit("execution_completed")

    result = runner.invoke(app, ["monitor", "show", "run", "--entry", str(tmp_path)])

    assert result.exit_code == 0
    assert "Run: run" in result.output
    assert "Execution: attempt" in result.output
    assert "Entry:" not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["monitor", "run", "--entry", "runs"],
        ["monitor", "--entry", "runs"],
        ["monitor", "show", "run", "--plain"],
        ["monitor", "tui", "run", "--rich"],
    ],
)
def test_monitor_removed_invocation_forms_fail(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2


def test_monitor_without_subcommand_prints_help() -> None:
    result = runner.invoke(app, ["monitor"])

    assert "tui" in result.output
    assert "web" in result.output
    assert "show" in result.output


def test_monitor_show_json_documents_run_and_fleet(tmp_path: Path) -> None:
    create_completed_monitor_run(tmp_path)

    run_result = runner.invoke(app, ["monitor", "show", "run", "--entry", str(tmp_path), "--json"])
    fleet_result = runner.invoke(app, ["monitor", "show", "--entry", str(tmp_path), "--json"])

    assert run_result.exit_code == 0
    run_document = json.loads(run_result.output)
    assert run_document["execution_id"] == "attempt"
    assert run_document["status"] == "completed"
    assert fleet_result.exit_code == 0
    assert json.loads(fleet_result.output)["loose_runs"][0]["run_name"] == "run"


def test_monitor_web_rejects_a_missing_root(tmp_path: Path) -> None:
    result = runner.invoke(app, ["monitor", "web", "--root", str(tmp_path / "missing")])

    assert result.exit_code == 2
    assert "Not a directory" in result.output


def test_monitor_web_serves_the_requested_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = Mock(url="http://127.0.0.1:9/")
    create = Mock(return_value=server)
    monkeypatch.setattr("mammoth.cli.create_web_server", create)

    result = runner.invoke(app, ["monitor", "web", "--root", str(tmp_path), "--port", "9"])

    assert result.exit_code == 0
    assert create.call_args.kwargs["host"] == "127.0.0.1"
    assert create.call_args.kwargs["port"] == 9
    server.serve_forever.assert_called_once()
    server.shutdown_monitor.assert_called_once()
