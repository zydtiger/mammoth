"""Compatibility entry point for the former optional Rich live display.

New callers use :mod:`mammoth.monitor.textual_ui`; this module preserves the
existing internal ``watch_rich`` function while routing it to Textual.
"""

from __future__ import annotations

from mammoth.monitor.model import ExecutionMonitor, RunMonitor
from mammoth.monitor.textual_ui import run_textual


def watch_rich(
    monitor: ExecutionMonitor,
    *,
    interval_seconds: float = 1.0,
    stale_after_seconds: float = 90.0,
) -> None:
    """Launch the Textual dashboard through the legacy helper name."""
    del stale_after_seconds
    run_monitor = RunMonitor(monitor.layout, monitor.context.metadata.execution_id)
    run_textual(
        run_monitor,
        run_monitor.poll(),
        watch=True,
        telemetry=True,
        interval_seconds=interval_seconds,
    )
