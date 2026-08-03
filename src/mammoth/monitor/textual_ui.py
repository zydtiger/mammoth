"""Textual application for Mammoth's passive live execution dashboard.

The public CLI imports this optional module only for an interactive terminal.
It polls :class:`mammoth.monitor.model.RunMonitor` and samples viewer-host
telemetry in an exclusive worker so the Textual event loop stays responsive.
"""

from __future__ import annotations

from dataclasses import replace
from threading import Lock

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
    PsutilViewerTelemetrySampler,
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
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
        initial_host: PsutilViewerTelemetry | None = None,
        telemetry_sampler: PsutilViewerTelemetrySampler | None = None,
    ) -> None:
        """Bind the passive monitor and interactive default policies."""
        super().__init__()
        self.monitor = monitor
        self.snapshot = initial_snapshot
        self.watch_enabled = watch
        self.telemetry_enabled = telemetry
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.host = initial_host
        self.telemetry_sampler = telemetry_sampler or (
            PsutilViewerTelemetrySampler() if telemetry else None
        )
        self.detail = monitor.execution_id is not None
        self.error: str | None = None
        self._refresh_generation = 0
        self._refresh_lock = Lock()

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
        self._refresh_generation += 1
        self._refresh_state(self._refresh_generation)

    @work(name="monitor-refresh", group="monitor-refresh", exclusive=True, thread=True)
    def _refresh_state(self, generation: int) -> None:
        """Poll files and local telemetry outside the Textual event loop."""
        try:
            with self._refresh_lock:
                snapshot = self.monitor.poll(self.snapshot.selected_execution_id)
                host = (
                    self.telemetry_sampler.sample()
                    if self.telemetry_sampler is not None
                    else None
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.call_from_thread(self._accept_error, generation, str(error))
            return
        self.call_from_thread(self._accept_refresh, generation, snapshot, host)

    def _accept_refresh(
        self,
        generation: int,
        snapshot: RunSnapshot,
        host: PsutilViewerTelemetry | None,
    ) -> None:
        """Publish one completed worker refresh on the Textual event loop."""
        if generation != self._refresh_generation:
            return
        self.snapshot = snapshot
        self.host = host
        self.error = None
        self._render()

    def _accept_error(self, generation: int, message: str) -> None:
        """Retain the last valid state and expose one refresh error."""
        if generation != self._refresh_generation:
            return
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
        if self.monitor.execution_id is None:
            self.detail = not self.detail
        self._render()

    def _move_selection(self, offset: int) -> None:
        """Move within immutable execution history and render immediately."""
        if self.monitor.execution_id is not None:
            return
        ids = [execution.execution_id for execution in self.snapshot.executions]
        index = ids.index(self.snapshot.selected_execution_id)
        next_index = min(len(ids) - 1, max(0, index + offset))
        self._refresh_generation += 1
        self.snapshot = replace(self.snapshot, selected_execution_id=ids[next_index])
        self._render()

    def _render(self) -> None:
        """Update the Rich dashboard using the current terminal width."""
        body = self.query_one("#body", Static)
        available_width = body.size.width or max(0, self.size.width - 2)
        compact = available_width < 80
        renderable = dashboard_layout(
            self.snapshot,
            host=self.host,
            detail=self.detail,
            compact=compact,
            pinned=self.monitor.execution_id is not None,
            stale_after_seconds=self.stale_after_seconds,
            refresh_seconds=self.interval_seconds,
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
    stale_after_seconds: float,
) -> None:
    """Run the Textual monitor application until the viewer quits."""
    telemetry_sampler = (
        PsutilViewerTelemetrySampler(allow_sudo_password_prompt=True)
        if telemetry
        else None
    )
    initial_host = telemetry_sampler.sample() if telemetry_sampler is not None else None
    MonitorApp(
        monitor,
        initial_snapshot,
        watch=watch,
        telemetry=telemetry,
        interval_seconds=interval_seconds,
        stale_after_seconds=stale_after_seconds,
        initial_host=initial_host,
        telemetry_sampler=telemetry_sampler,
    ).run()
