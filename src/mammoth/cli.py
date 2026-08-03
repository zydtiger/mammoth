"""Typer command-line entry point for Mammoth infrastructure.

This module exposes the public monitor and workflow command tree while keeping
optional monitor UI imports out of base package startup. The console script and
``python -m mammoth`` both enter through :func:`main`.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from yaml import YAMLError

from mammoth import __version__
from mammoth.core import RunLayout
from mammoth.monitor import ExecutionMonitor, render_snapshot, sample_viewer_telemetry
from mammoth.workflow import load_workflow, plan_workflow, run_workflow

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
app = typer.Typer(
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
    help="Project-independent infrastructure for AI workloads.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
workflow_app = typer.Typer(
    context_settings=CONTEXT_SETTINGS,
    help="Run a declarative command workflow.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(workflow_app, name="workflow")


def version_callback(value: bool) -> None:
    """Print the package version for the eager root option."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Show the Mammoth version and exit.",
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Route the public Mammoth command tree."""


def positive_float(value: float) -> float:
    """Validate one finite positive CLI duration for the monitor command."""
    if not math.isfinite(value) or value <= 0:
        raise typer.BadParameter("value must be finite and greater than zero")
    return value


@app.command("monitor")
def run_monitor(
    run_name: Annotated[str, typer.Argument(help="Logical run name to inspect.")],
    entry: Annotated[Path, typer.Option("--entry", help="Run-directory entry path.")],
    execution: Annotated[
        str | None,
        typer.Option("--execution", help="Exact immutable execution to inspect."),
    ] = None,
    watch: Annotated[bool, typer.Option("--watch", help="Poll until the run finishes.")] = False,
    rich: Annotated[bool, typer.Option("--rich", help="Use the optional Rich live view.")] = False,
    telemetry: Annotated[
        bool,
        typer.Option("--telemetry", help="Include viewer-host telemetry."),
    ] = False,
    interval: Annotated[
        float,
        typer.Option(
            "--interval",
            callback=positive_float,
            help="Polling interval in seconds.",
        ),
    ] = 1.0,
) -> None:
    """Render once or watch one selected execution."""
    try:
        monitor = ExecutionMonitor(RunLayout(entry, run_name), execution)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="RUN_NAME/--execution") from None
    if rich:
        rich_module = importlib.import_module("mammoth.monitor.rich_ui")
        rich_module.watch_rich(monitor, interval_seconds=interval)
        return
    while True:
        snapshot = monitor.poll()
        sys.stdout.write(render_snapshot(snapshot))
        if telemetry:
            sys.stdout.write(json.dumps(asdict(sample_viewer_telemetry()), sort_keys=True) + "\n")
        sys.stdout.flush()
        if not watch or snapshot.status in {"completed", "failed", "interrupted"}:
            return
        time.sleep(interval)


@workflow_app.command("run")
def run_workflow_command(
    workflow_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Strict YAML workflow file.",
        ),
    ],
    entry: Annotated[Path, typer.Option("--entry", help="Run-directory entry path.")],
    selected_runs: Annotated[
        list[str] | None,
        typer.Option("--run", help="Select one run exactly; repeat to select more."),
    ] = None,
    selected_steps: Annotated[
        list[str] | None,
        typer.Option("--step", help="Select one step exactly; repeat to select more."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print commands without artifacts."),
    ] = False,
) -> None:
    """Plan or execute a strict YAML workflow from the public CLI."""
    try:
        workflow = load_workflow(workflow_file)
        plan_workflow(
            workflow,
            selected_runs=selected_runs,
            selected_steps=selected_steps,
        )
    except (OSError, ValueError, KeyError, YAMLError) as error:
        message = error.args[0] if isinstance(error, KeyError) else str(error)
        raise typer.BadParameter(
            str(message),
            param_hint="WORKFLOW_FILE/--run/--step",
        ) from None
    result = run_workflow(
        workflow,
        entry=entry,
        selected_runs=selected_runs,
        selected_steps=selected_steps,
        dry_run=dry_run,
        invocation_command=tuple(sys.argv),
    )
    if dry_run:
        for plan in result.plans:
            sys.stdout.write(f"{plan.run_name}/{plan.step_name}: {' '.join(plan.command)}\n")
    else:
        for run_result in result.runs:
            sys.stdout.write(
                f"{run_result.run_name}: {run_result.outcome} "
                f"execution={run_result.execution_id}\n"
            )
    if not result.successful:
        raise typer.Exit(code=1)


def run(argv: Sequence[str] | None = None) -> int:
    """Execute one Typer command and return its process exit code."""
    try:
        app(args=list(argv) if argv is not None else None, prog_name="mammoth")
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    return 0


def main() -> NoReturn:
    """Terminate the console process with :func:`run`'s status."""
    raise SystemExit(run())
