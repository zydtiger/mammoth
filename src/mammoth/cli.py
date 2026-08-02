"""Standard-library command-line entry point for Mammoth infrastructure.

This module routes monitor commands now and is extended by the workflow layer.
It keeps optional UI imports out of base package startup.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from mammoth import __version__
from mammoth.core import RunLayout
from mammoth.monitor import ExecutionMonitor, render_snapshot, sample_viewer_telemetry
from mammoth.workflow import load_workflow, run_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the public Mammoth command parser."""
    parser = argparse.ArgumentParser(prog="mammoth")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("monitor", help="passively inspect one logical run")
    monitor.add_argument("run_name")
    monitor.add_argument("--entry", type=Path, required=True)
    monitor.add_argument("--execution")
    monitor.add_argument("--watch", action="store_true")
    monitor.add_argument("--rich", action="store_true")
    monitor.add_argument("--telemetry", action="store_true")
    monitor.add_argument("--interval", type=positive_float, default=1.0)
    workflow = commands.add_parser("workflow", help="run a declarative command workflow")
    workflow_commands = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_run = workflow_commands.add_parser("run", help="plan or execute a workflow")
    workflow_run.add_argument("workflow_file", type=Path)
    workflow_run.add_argument("--entry", type=Path, required=True)
    workflow_run.add_argument("--run", dest="selected_runs", action="append")
    workflow_run.add_argument("--step", dest="selected_steps", action="append")
    workflow_run.add_argument("--dry-run", action="store_true")
    return parser


def positive_float(value: str) -> float:
    """Parse one positive CLI duration."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def run(argv: Sequence[str] | None = None) -> int:
    """Execute one Mammoth command and return its process exit code."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "monitor":
        return run_monitor(arguments)
    if arguments.command == "workflow" and arguments.workflow_command == "run":
        return run_workflow_command(arguments)
    raise AssertionError(f"Unhandled Mammoth command: {arguments.command}")


def run_monitor(arguments: argparse.Namespace) -> int:
    """Render once or watch one selected execution."""
    monitor = ExecutionMonitor(
        RunLayout(arguments.entry, arguments.run_name),
        arguments.execution,
    )
    if arguments.rich:
        rich_module = importlib.import_module("mammoth.monitor.rich_ui")
        rich_module.watch_rich(monitor, interval_seconds=arguments.interval)
        return 0
    while True:
        snapshot = monitor.poll()
        sys.stdout.write(render_snapshot(snapshot))
        if arguments.telemetry:
            sys.stdout.write(json.dumps(asdict(sample_viewer_telemetry()), sort_keys=True) + "\n")
        sys.stdout.flush()
        if not arguments.watch or snapshot.status in {"completed", "failed", "interrupted"}:
            return 0
        time.sleep(arguments.interval)


def run_workflow_command(arguments: argparse.Namespace) -> int:
    """Plan or execute a strict YAML workflow from the public CLI."""
    result = run_workflow(
        load_workflow(arguments.workflow_file),
        entry=arguments.entry,
        selected_runs=arguments.selected_runs,
        selected_steps=arguments.selected_steps,
        dry_run=arguments.dry_run,
        invocation_command=tuple(sys.argv),
    )
    if arguments.dry_run:
        for plan in result.plans:
            sys.stdout.write(
                f"{plan.run_name}/{plan.step_name}: "
                f"{' '.join(plan.command)}\n"
            )
    else:
        for run_result in result.runs:
            sys.stdout.write(
                f"{run_result.run_name}: {run_result.outcome} "
                f"execution={run_result.execution_id}\n"
            )
    return 0 if result.successful else 1


def main() -> NoReturn:
    """Console-script wrapper that terminates with :func:`run`'s status."""
    raise SystemExit(run())
