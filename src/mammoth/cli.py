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


def main() -> NoReturn:
    """Console-script wrapper that terminates with :func:`run`'s status."""
    raise SystemExit(run())
