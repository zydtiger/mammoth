"""Optional Rich live display for the passive Mammoth monitor.

The CLI loads this module only when the ``monitor`` extra is installed and the
user requests an interactive view.
"""

from __future__ import annotations

import time

from rich.live import Live
from rich.panel import Panel

from mammoth.monitor.model import ExecutionMonitor
from mammoth.monitor.render import render_snapshot


def watch_rich(monitor: ExecutionMonitor, *, interval_seconds: float = 1.0) -> None:
    """Refresh an interactive Rich panel until interrupted."""
    with Live(refresh_per_second=max(1, round(1 / interval_seconds))) as live:
        while True:
            snapshot = monitor.poll()
            live.update(Panel(render_snapshot(snapshot), title="Mammoth"), refresh=True)
            if snapshot.status in {"completed", "failed", "interrupted"}:
                return
            time.sleep(interval_seconds)
