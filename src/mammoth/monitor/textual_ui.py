"""Textual application for Mammoth's passive live execution dashboard.

The public CLI imports this optional module only for an interactive terminal.
It polls :class:`mammoth.monitor.model.RunMonitor` and samples viewer-host
telemetry in an exclusive worker so the Textual event loop stays responsive.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock

from rich.console import Group
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Static

from mammoth.core.layout import RunLayout
from mammoth.monitor.dashboard import (
    dashboard_layout,
    fleet_dashboard_layout,
    fleet_rows,
    group_dashboard_layout,
)
from mammoth.monitor.fleet import FleetMonitor, FleetSnapshot, GroupSnapshot
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


class RunScreen(Screen[None]):
    """Push the existing single-run dashboard onto the fleet navigation stack.

    Mirrors :class:`MonitorApp`'s refresh, selection, and detail-toggle
    behavior as a pushed screen instead of a standalone application, so
    drilling into a run from the fleet or a group view reaches the same
    dashboard the standalone ``mammoth monitor <run_name>`` invocation opens.
    Back navigation pops this screen and returns to the originating view.
    """

    CSS = MonitorApp.CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j,down", "next_execution", "Next"),
        Binding("k,up", "previous_execution", "Previous"),
        Binding("enter", "toggle_detail", "Overview / detail"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape,backspace", "app.pop_screen", "Back"),
    ]

    def __init__(
        self,
        monitor: RunMonitor,
        initial_snapshot: RunSnapshot,
        *,
        watch: bool = True,
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
        host: PsutilViewerTelemetry | None = None,
        telemetry_sampler: PsutilViewerTelemetrySampler | None = None,
    ) -> None:
        """Bind the passive run monitor drilled into from a fleet or group view."""
        super().__init__()
        self.monitor = monitor
        self.snapshot = initial_snapshot
        self.watch_enabled = watch
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.host = host
        self.telemetry_sampler = telemetry_sampler
        self.detail = False
        self.error: str | None = None
        self._refresh_generation = 0
        self._refresh_lock = Lock()

    def compose(self) -> ComposeResult:
        """Create one scrollable dashboard surface and standard footer."""
        yield Static(id="body")
        yield Footer()

    def on_mount(self) -> None:
        """Render immediately and schedule nonblocking refreshes when enabled."""
        self._update_body()
        self.action_refresh()
        if self.watch_enabled:
            self.set_interval(self.interval_seconds, self.action_refresh)

    def on_resize(self) -> None:
        """Switch layouts when the terminal crosses the compact breakpoint."""
        self._update_body()

    def action_refresh(self) -> None:
        """Start one exclusive background poll and telemetry sample."""
        self._refresh_generation += 1
        self._refresh_state(self._refresh_generation)

    @work(name="run-screen-refresh", group="run-screen-refresh", exclusive=True, thread=True)
    def _refresh_state(self, generation: int) -> None:
        """Poll files and local telemetry outside the Textual event loop."""
        try:
            with self._refresh_lock:
                snapshot = self.monitor.poll(self.snapshot.selected_execution_id)
                host = (
                    self.telemetry_sampler.sample() if self.telemetry_sampler is not None else None
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.app.call_from_thread(self._accept_error, generation, str(error))
            return
        self.app.call_from_thread(self._accept_refresh, generation, snapshot, host)

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
        self._update_body()

    def _accept_error(self, generation: int, message: str) -> None:
        """Retain the last valid state and expose one refresh error."""
        if generation != self._refresh_generation:
            return
        self.error = message
        self._update_body()

    def action_next_execution(self) -> None:
        """Select the next execution."""
        self._move_selection(1)

    def action_previous_execution(self) -> None:
        """Select the previous execution."""
        self._move_selection(-1)

    def action_toggle_detail(self) -> None:
        """Toggle logical overview and exact selected-execution details."""
        self.detail = not self.detail
        self._update_body()

    def _move_selection(self, offset: int) -> None:
        """Move within immutable execution history and render immediately."""
        ids = [execution.execution_id for execution in self.snapshot.executions]
        index = ids.index(self.snapshot.selected_execution_id)
        next_index = min(len(ids) - 1, max(0, index + offset))
        self._refresh_generation += 1
        self.snapshot = replace(self.snapshot, selected_execution_id=ids[next_index])
        self._update_body()

    def _update_body(self) -> None:
        """Update the Rich dashboard using the current terminal width."""
        body = self.query_one("#body", Static)
        available_width = body.size.width or max(0, self.size.width - 2)
        compact = available_width < 80
        renderable = dashboard_layout(
            self.snapshot,
            host=self.host,
            detail=self.detail,
            compact=compact,
            pinned=False,
            stale_after_seconds=self.stale_after_seconds,
            refresh_seconds=self.interval_seconds,
        )
        if self.error is not None:
            renderable = Group(
                renderable,
                Text(f"Refresh warning: {self.error}", style="yellow"),
            )
        body.update(renderable)


class GroupScreen(Screen[None]):
    """Show one group's members in schedule order, folded from a shared fleet poll.

    Each refresh re-polls the same :class:`~mammoth.monitor.fleet.FleetMonitor`
    the fleet screen uses (incremental group manifest, group event, and run
    tail folding) and looks this group up by ID in the fresh snapshot, so no
    separate monitor is constructed for the group view.
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j,down", "next_row", "Next"),
        Binding("k,up", "previous_row", "Previous"),
        Binding("enter", "open_selected", "Open run"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape,backspace", "app.pop_screen", "Back"),
    ]

    def __init__(
        self,
        fleet_app: FleetApp,
        fleet_monitor: FleetMonitor,
        group_id: str,
        initial_group: GroupSnapshot,
        *,
        watch: bool = True,
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
    ) -> None:
        """Bind the fleet monitor and the group this screen displays."""
        super().__init__()
        self.fleet_app = fleet_app
        self.fleet_monitor = fleet_monitor
        self.group_id = group_id
        self.group = initial_group
        self.watch_enabled = watch
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.selected_index = 0
        self.error: str | None = None
        self._refresh_generation = 0
        self._refresh_lock = Lock()

    def compose(self) -> ComposeResult:
        """Create one scrollable dashboard surface and standard footer."""
        yield Static(id="body")
        yield Footer()

    def on_mount(self) -> None:
        """Render immediately and schedule nonblocking refreshes when enabled."""
        self._update_body()
        self.action_refresh()
        if self.watch_enabled:
            self.set_interval(self.interval_seconds, self.action_refresh)

    def on_resize(self) -> None:
        """Switch layouts when the terminal crosses the compact breakpoint."""
        self._update_body()

    def action_refresh(self) -> None:
        """Start one exclusive background fleet poll."""
        self._refresh_generation += 1
        self._refresh_state(self._refresh_generation)

    @work(name="group-refresh", group="group-refresh", exclusive=True, thread=True)
    def _refresh_state(self, generation: int) -> None:
        """Poll the shared fleet monitor outside the Textual event loop."""
        try:
            with self._refresh_lock:
                fleet_snapshot = self.fleet_monitor.poll(
                    stale_after_seconds=self.stale_after_seconds
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.app.call_from_thread(self._accept_error, generation, str(error))
            return
        group = next(
            (item for item in fleet_snapshot.groups if item.group_id == self.group_id),
            None,
        )
        self.app.call_from_thread(self._accept_refresh, generation, group)

    def _accept_refresh(self, generation: int, group: GroupSnapshot | None) -> None:
        """Publish one completed worker refresh on the Textual event loop."""
        if generation != self._refresh_generation:
            return
        if group is not None:
            self.group = group
            if self.group.members:
                self.selected_index = min(self.selected_index, len(self.group.members) - 1)
        self.error = None
        self._update_body()

    def _accept_error(self, generation: int, message: str) -> None:
        """Retain the last valid state and expose one refresh error."""
        if generation != self._refresh_generation:
            return
        self.error = message
        self._update_body()

    def action_next_row(self) -> None:
        """Select the next member row."""
        self._move_selection(1)

    def action_previous_row(self) -> None:
        """Select the previous member row."""
        self._move_selection(-1)

    def _move_selection(self, offset: int) -> None:
        if not self.group.members:
            return
        self.selected_index = min(len(self.group.members) - 1, max(0, self.selected_index + offset))
        self._update_body()

    def action_open_selected(self) -> None:
        """Drill into the selected member's existing single-run dashboard."""
        if not self.group.members or self.selected_index >= len(self.group.members):
            return
        run_name = self.group.members[self.selected_index].run_name
        self.fleet_app.push_run_screen(run_name)

    def _update_body(self) -> None:
        """Update the Rich dashboard using the current terminal width."""
        body = self.query_one("#body", Static)
        available_width = body.size.width or max(0, self.size.width - 2)
        compact = available_width < 80
        renderable = group_dashboard_layout(
            self.group,
            selected_index=self.selected_index,
            compact=compact,
            stale_after_seconds=self.stale_after_seconds,
            refresh_seconds=self.interval_seconds,
        )
        if self.error is not None:
            renderable = Group(
                renderable,
                Text(f"Refresh warning: {self.error}", style="yellow"),
            )
        body.update(renderable)


class FleetScreen(Screen[None]):
    """Show every group and loose run beneath one entry, the fleet root view."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j,down", "next_row", "Next"),
        Binding("k,up", "previous_row", "Previous"),
        Binding("enter", "open_selected", "Open"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        fleet_app: FleetApp,
        fleet_monitor: FleetMonitor,
        initial_snapshot: FleetSnapshot,
        *,
        watch: bool = True,
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
    ) -> None:
        """Bind the fleet monitor this root screen refreshes and navigates."""
        super().__init__()
        self.fleet_app = fleet_app
        self.fleet_monitor = fleet_monitor
        self.snapshot = initial_snapshot
        self.watch_enabled = watch
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.selected_index = 0
        self.error: str | None = None
        self._refresh_generation = 0
        self._refresh_lock = Lock()

    def compose(self) -> ComposeResult:
        """Create one scrollable dashboard surface and standard footer."""
        yield Static(id="body")
        yield Footer()

    def on_mount(self) -> None:
        """Render immediately and schedule nonblocking refreshes when enabled."""
        self._update_body()
        self.action_refresh()
        if self.watch_enabled:
            self.set_interval(self.interval_seconds, self.action_refresh)

    def on_resize(self) -> None:
        """Switch layouts when the terminal crosses the compact breakpoint."""
        self._update_body()

    def action_refresh(self) -> None:
        """Start one exclusive background fleet poll."""
        self._refresh_generation += 1
        self._refresh_state(self._refresh_generation)

    @work(name="fleet-refresh", group="fleet-refresh", exclusive=True, thread=True)
    def _refresh_state(self, generation: int) -> None:
        """Poll every group and loose run outside the Textual event loop."""
        try:
            with self._refresh_lock:
                snapshot = self.fleet_monitor.poll(stale_after_seconds=self.stale_after_seconds)
        except (OSError, RuntimeError, ValueError) as error:
            self.app.call_from_thread(self._accept_error, generation, str(error))
            return
        self.app.call_from_thread(self._accept_refresh, generation, snapshot)

    def _accept_refresh(self, generation: int, snapshot: FleetSnapshot) -> None:
        """Publish one completed worker refresh on the Textual event loop."""
        if generation != self._refresh_generation:
            return
        self.snapshot = snapshot
        rows = fleet_rows(snapshot)
        self.selected_index = min(self.selected_index, len(rows) - 1) if rows else 0
        self.error = None
        self._update_body()

    def _accept_error(self, generation: int, message: str) -> None:
        """Retain the last valid state and expose one refresh error."""
        if generation != self._refresh_generation:
            return
        self.error = message
        self._update_body()

    def action_next_row(self) -> None:
        """Select the next group or loose-run row."""
        self._move_selection(1)

    def action_previous_row(self) -> None:
        """Select the previous group or loose-run row."""
        self._move_selection(-1)

    def _move_selection(self, offset: int) -> None:
        rows = fleet_rows(self.snapshot)
        if not rows:
            return
        self.selected_index = min(len(rows) - 1, max(0, self.selected_index + offset))
        self._update_body()

    def action_open_selected(self) -> None:
        """Drill into the selected group's members or straight into a loose run."""
        rows = fleet_rows(self.snapshot)
        if not rows or self.selected_index >= len(rows):
            return
        row = rows[self.selected_index]
        if row.kind == "group":
            group = next(
                (item for item in self.snapshot.groups if item.group_id == row.key),
                None,
            )
            if group is not None:
                self.fleet_app.push_group_screen(row.key, group)
        else:
            self.fleet_app.push_run_screen(row.key)

    def _update_body(self) -> None:
        """Update the Rich dashboard using the current terminal width."""
        body = self.query_one("#body", Static)
        available_width = body.size.width or max(0, self.size.width - 2)
        compact = available_width < 80
        renderable = fleet_dashboard_layout(
            self.snapshot,
            selected_index=self.selected_index,
            compact=compact,
            stale_after_seconds=self.stale_after_seconds,
            refresh_seconds=self.interval_seconds,
        )
        if self.error is not None:
            renderable = Group(
                renderable,
                Text(f"Refresh warning: {self.error}", style="yellow"),
            )
        body.update(renderable)


class FleetApp(App[None]):
    """Root the Fleet -> Group -> Run Textual screen stack for one entry.

    Only the fleet and (when selected) group screens are active by default;
    a full :class:`~mammoth.monitor.model.RunMonitor` is constructed only in
    :meth:`push_run_screen`, when an operator actually drills into one run.
    """

    CSS = MonitorApp.CSS

    def __init__(
        self,
        entry: Path,
        fleet_monitor: FleetMonitor,
        initial_snapshot: FleetSnapshot,
        *,
        watch: bool = True,
        telemetry: bool = True,
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 90.0,
        open_group_id: str | None = None,
        telemetry_sampler: PsutilViewerTelemetrySampler | None = None,
        initial_host: PsutilViewerTelemetry | None = None,
    ) -> None:
        """Bind the entry-wide fleet monitor and optional direct group target.

        ``telemetry_sampler`` and ``initial_host`` are created by
        :func:`run_fleet_textual` before this app is constructed, sampling
        DIMM identity (and any sudo password prompt it needs) on the plain
        terminal instead of from inside the running Textual app. This app
        never constructs a sampler itself. The pre-UI ``initial_host`` sample
        is reused for the first run screen drilled into rather than
        discarded; see :meth:`push_run_screen`.
        """
        super().__init__()
        self.entry = entry
        self.fleet_monitor = fleet_monitor
        self.initial_snapshot = initial_snapshot
        self.watch_enabled = watch
        self.telemetry_enabled = telemetry
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.open_group_id = open_group_id
        self._telemetry_sampler = telemetry_sampler
        self._initial_host = initial_host

    def on_mount(self) -> None:
        """Root the screen stack at the fleet view, opening a requested group."""
        self.push_screen(
            FleetScreen(
                self,
                self.fleet_monitor,
                self.initial_snapshot,
                watch=self.watch_enabled,
                interval_seconds=self.interval_seconds,
                stale_after_seconds=self.stale_after_seconds,
            )
        )
        if self.open_group_id is not None:
            group = next(
                (
                    item
                    for item in self.initial_snapshot.groups
                    if item.group_id == self.open_group_id
                ),
                None,
            )
            if group is not None:
                self.push_group_screen(self.open_group_id, group)

    def push_group_screen(self, group_id: str, group: GroupSnapshot) -> None:
        """Push the group screen for ``group_id`` using an already-resolved snapshot."""
        self.push_screen(
            GroupScreen(
                self,
                self.fleet_monitor,
                group_id,
                group,
                watch=self.watch_enabled,
                interval_seconds=self.interval_seconds,
                stale_after_seconds=self.stale_after_seconds,
            )
        )

    def push_run_screen(self, run_name: str) -> None:
        """Lazily construct one full RunMonitor and push its dashboard screen."""
        run_monitor = RunMonitor(RunLayout(self.entry, run_name))
        try:
            initial_snapshot = run_monitor.poll()
        except FileNotFoundError:
            return
        self.push_screen(
            RunScreen(
                run_monitor,
                initial_snapshot,
                watch=self.watch_enabled,
                interval_seconds=self.interval_seconds,
                stale_after_seconds=self.stale_after_seconds,
                host=self._next_run_screen_host(),
                telemetry_sampler=self._telemetry_sampler,
            )
        )

    def _next_run_screen_host(self) -> PsutilViewerTelemetry | None:
        """Reuse the pre-UI telemetry sample once, then sample fresh after.

        The first run drilled into after startup gets the sample already
        taken before this app ran (see :func:`run_fleet_textual`) instead of
        a redundant duplicate; every later drill-in samples again through the
        same sampler, which never prompts twice since it tries cached sudo
        credentials first.
        """
        if self._initial_host is not None:
            host, self._initial_host = self._initial_host, None
            return host
        return self._telemetry_sampler.sample() if self._telemetry_sampler is not None else None


def run_fleet_textual(
    entry: Path,
    fleet_monitor: FleetMonitor,
    initial_snapshot: FleetSnapshot,
    *,
    watch: bool,
    telemetry: bool,
    interval_seconds: float,
    stale_after_seconds: float,
    open_group_id: str | None = None,
) -> None:
    """Run the Textual fleet application until the viewer quits.

    Mirrors :func:`run_textual`: samples viewer-host telemetry once on the
    plain terminal, before the app is constructed or run, so a DIMM-identity
    sudo password prompt happens outside the Textual UI instead of erupting
    from inside it the first time an operator drills into a run.
    """
    telemetry_sampler = (
        PsutilViewerTelemetrySampler(allow_sudo_password_prompt=True) if telemetry else None
    )
    initial_host = telemetry_sampler.sample() if telemetry_sampler is not None else None
    FleetApp(
        entry,
        fleet_monitor,
        initial_snapshot,
        watch=watch,
        telemetry=telemetry,
        interval_seconds=interval_seconds,
        stale_after_seconds=stale_after_seconds,
        telemetry_sampler=telemetry_sampler,
        initial_host=initial_host,
        open_group_id=open_group_id,
    ).run()
