"""Textual application for Mammoth's passive live execution dashboard.

The public CLI imports this optional module only for an interactive terminal.
It polls :class:`mammoth.monitor.model.RunMonitor` and samples viewer-host
telemetry in an exclusive worker so the Textual event loop stays responsive.
"""

from __future__ import annotations

from dataclasses import replace

from rich.console import Group
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Static

from mammoth.monitor.dashboard import dashboard_layout
from mammoth.monitor.model import RunMonitor, RunSnapshot
from mammoth.monitor.psutil_telemetry import (
    PsutilViewerTelemetry,
    sample_psutil_viewer_telemetry,
)


class MonitorApp(App[None]):
    """Refresh one logical run and expose execution-history navigation."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    Footer {
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j,down", "next_execution", "Next"),
        Binding("k,up", "previous_execution", "Previous"),
        Binding("enter", "toggle_detail", "Overview / detail"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        monitor: RunMonitor,
        initial_snapshot: RunSnapshot,
        *,
        watch: bool = True,
        telemetry: bool = True,
        interval_seconds: float = 1.0,
    ) -> None:
        """Bind the passive monitor and interactive default policies."""
        super().__init__()
        self.monitor = monitor
        self.snapshot = initial_snapshot
        self.watch_enabled = watch
        self.telemetry_enabled = telemetry
        self.interval_seconds = interval_seconds
        self.host: PsutilViewerTelemetry | None = None
        self.detail = False
        self.error: str | None = None

    def compose(self) -> ComposeResult:
        """Create one scrollable dashboard surface and standard footer."""
        yield Static(id="body")
        yield Footer()

    def on_mount(self) -> None:
        """Render immediately and schedule nonblocking refreshes when enabled."""
        self._render()
        self.action_refresh()
        if self.watch_enabled:
            self.set_interval(self.interval_seconds, self.action_refresh)

    def on_resize(self) -> None:
        """Switch layouts when the terminal crosses the compact breakpoint."""
        self._render()

    def action_refresh(self) -> None:
        """Start one exclusive background poll and telemetry sample."""
        self._refresh_state()

    @work(name="monitor-refresh", group="monitor-refresh", exclusive=True, thread=True)
    def _refresh_state(self) -> None:
        """Poll files and local telemetry outside the Textual event loop."""
        try:
            snapshot = self.monitor.poll(self.snapshot.selected_execution_id)
            host = sample_psutil_viewer_telemetry() if self.telemetry_enabled else None
        except (OSError, RuntimeError, ValueError) as error:
            self.call_from_thread(self._accept_error, str(error))
            return
        self.call_from_thread(self._accept_refresh, snapshot, host)

    def _accept_refresh(
        self,
        snapshot: RunSnapshot,
        host: PsutilViewerTelemetry | None,
    ) -> None:
        """Publish one completed worker refresh on the Textual event loop."""
        self.snapshot = snapshot
        self.host = host
        self.error = None
        self._render()

    def _accept_error(self, message: str) -> None:
        """Retain the last valid state and expose one refresh error."""
        self.error = message
        self._render()

    def action_next_execution(self) -> None:
        """Select the next execution unless an exact execution was requested."""
        self._move_selection(1)

    def action_previous_execution(self) -> None:
        """Select the previous execution unless an exact execution was requested."""
        self._move_selection(-1)

    def action_toggle_detail(self) -> None:
        """Toggle logical overview and exact selected-execution details."""
        self.detail = not self.detail
        self._render()

    def _move_selection(self, offset: int) -> None:
        """Move within immutable execution history and render immediately."""
        if self.monitor.execution_id is not None:
            return
        ids = [execution.execution_id for execution in self.snapshot.executions]
        index = ids.index(self.snapshot.selected_execution_id)
        next_index = min(len(ids) - 1, max(0, index + offset))
        self.snapshot = replace(self.snapshot, selected_execution_id=ids[next_index])
        self._render()

    def _render(self) -> None:
        """Update the Rich dashboard using the current terminal width."""
        body = self.query_one("#body", Static)
        compact = (body.size.width or self.size.width) < 80
        renderable = dashboard_layout(
            self.snapshot,
            host=self.host,
            detail=self.detail,
            compact=compact,
        )
        if self.error is not None:
            renderable = Group(
                renderable,
                Text(f"Refresh warning: {self.error}", style="yellow"),
            )
        body.update(renderable)


def run_textual(
    monitor: RunMonitor,
    initial_snapshot: RunSnapshot,
    *,
    watch: bool,
    telemetry: bool,
    interval_seconds: float,
) -> None:
    """Run the Textual monitor application until the viewer quits."""
    MonitorApp(
        monitor,
        initial_snapshot,
        watch=watch,
        telemetry=telemetry,
        interval_seconds=interval_seconds,
    ).run()
