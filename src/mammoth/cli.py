"""Typer command-line entry point for Mammoth monitoring infrastructure.

This module keeps optional monitor UI imports out of base package startup. The
console script and ``python -m mammoth`` both enter through :func:`main`.
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
from types import ModuleType
from typing import Annotated, Any, NoReturn

import typer

from mammoth import __version__
from mammoth.core import RunLayout
from mammoth.monitor import (
    ExecutionMonitor,
    FleetMonitor,
    FleetSnapshot,
    GroupSnapshot,
    RunMonitor,
    render_fleet_snapshot,
    render_group_snapshot,
    render_snapshot,
    sample_viewer_telemetry,
)
from mammoth.queue import (
    DeviceLeaseConflictError,
    Job,
    JobAlreadyClaimedError,
    JobNotFoundError,
    JobOutcome,
    QueueError,
    QueueSnapshot,
    cancel_job,
    list_jobs,
    run_serve_loop,
    submit_job,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
app = typer.Typer(
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
    help="Project-independent infrastructure for AI workloads.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
queue_app = typer.Typer(
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
    help="Local device-aware job queue: submit, list, cancel, and serve opaque jobs.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(queue_app, name="queue")


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


def stdout_is_interactive() -> bool:
    """Return whether the monitor output supports an interactive dashboard."""
    return sys.stdout.isatty()


def load_textual_ui() -> ModuleType:
    """Load the optional Textual module only for an interactive invocation."""
    return importlib.import_module("mammoth.monitor.textual_ui")


@app.command("monitor")
def run_monitor(
    run_name: Annotated[
        str | None,
        typer.Argument(help="Logical run name to inspect; omit for the entry-level fleet view."),
    ] = None,
    entry: Annotated[
        Path,
        typer.Option("--entry", help="Run-directory entry path."),
    ] = Path("./runs"),
    group: Annotated[
        str | None,
        typer.Option(
            "--group",
            help="Exact group ID to open directly, without RUN_NAME.",
        ),
    ] = None,
    match: Annotated[
        str | None,
        typer.Option(
            "--match",
            help=(
                "Glob pattern grouping loose runs ad hoc in the fleet view, "
                "for entries without group manifests."
            ),
        ),
    ] = None,
    execution: Annotated[
        str | None,
        typer.Option("--execution", help="Exact immutable execution to inspect."),
    ] = None,
    watch: Annotated[
        bool | None,
        typer.Option("--watch/--no-watch", help="Enable or disable continuous polling."),
    ] = None,
    rich: Annotated[
        bool | None,
        typer.Option(
            "--rich/--plain",
            help="Use the Textual dashboard or stable plain output.",
        ),
    ] = None,
    telemetry: Annotated[
        bool | None,
        typer.Option(
            "--telemetry/--no-telemetry",
            help="Enable or disable explicitly viewer-host telemetry.",
        ),
    ] = None,
    interval: Annotated[
        float,
        typer.Option(
            "--interval",
            callback=positive_float,
            help="Polling interval in seconds.",
        ),
    ] = 2.0,
    stale_after: Annotated[
        float,
        typer.Option(
            "--stale-after",
            callback=positive_float,
            help="Seconds without an observation before a running producer is stale.",
        ),
    ] = 90.0,
) -> None:
    """Open the live dashboard on a TTY or render a stable plain snapshot.

    With ``RUN_NAME`` this is exactly the original single-run invocation.
    Without it, ``--entry`` opens the fleet view listing every group and
    loose run; ``--group`` opens one group directly; ``--match`` groups loose
    runs by name pattern ad hoc for entries without group manifests.
    """
    if run_name is not None and (group is not None or match is not None):
        raise typer.BadParameter(
            "--group and --match apply only when RUN_NAME is omitted",
            param_hint="RUN_NAME",
        )
    if run_name is None and execution is not None:
        raise typer.BadParameter(
            "--execution applies only to a single-run invocation (RUN_NAME given)",
            param_hint="--execution",
        )
    if group is not None and match is not None:
        raise typer.BadParameter(
            "--group and --match are mutually exclusive",
            param_hint="--group",
        )
    interactive = stdout_is_interactive() if rich is None else rich
    should_watch = interactive if watch is None else watch
    include_telemetry = interactive if telemetry is None else telemetry

    if run_name is None:
        _run_fleet_monitor(
            entry,
            group=group,
            match=match,
            interactive=interactive,
            should_watch=should_watch,
            include_telemetry=include_telemetry,
            interval=interval,
            stale_after=stale_after,
        )
        return

    try:
        layout = RunLayout(entry, run_name)
        monitor = ExecutionMonitor(layout, execution)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="RUN_NAME/--execution") from None
    if interactive:
        try:
            textual_module = load_textual_ui()
        except ModuleNotFoundError as error:
            typer.echo(
                "Interactive monitoring requires the monitor extra; "
                "install it with `uv sync --extra monitor` or pass `--plain`.",
                err=True,
            )
            raise typer.Exit(code=2) from error
        run_monitor_state = RunMonitor(layout, execution)
        initial_snapshot = run_monitor_state.poll()
        textual_module.run_textual(
            run_monitor_state,
            initial_snapshot,
            watch=should_watch,
            telemetry=include_telemetry,
            interval_seconds=interval,
            stale_after_seconds=stale_after,
        )
        return
    while True:
        snapshot = monitor.poll()
        sys.stdout.write(render_snapshot(snapshot, stale_after_seconds=stale_after))
        if include_telemetry:
            sys.stdout.write(json.dumps(asdict(sample_viewer_telemetry()), sort_keys=True) + "\n")
        sys.stdout.flush()
        if not should_watch or snapshot.status in {"completed", "failed", "interrupted"}:
            return
        time.sleep(interval)


@queue_app.command("submit")
def queue_submit(
    command: Annotated[
        list[str],
        typer.Argument(help="Opaque job argv; separate options from it with '--'."),
    ],
    device: Annotated[str, typer.Option("--device", help="Device spec this job requires.")],
    run_name: Annotated[
        list[str],
        typer.Option("--run-name", help="Run name this job expects to produce; repeatable."),
    ],
    entry: Annotated[Path, typer.Option("--entry", help="Queue entry root.")] = Path("./runs"),
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for the launched job."),
    ] = None,
    metadata: Annotated[
        str | None,
        typer.Option("--metadata", help="Opaque JSON object recorded with the job."),
    ] = None,
) -> None:
    """Durably enqueue one opaque job under the entry's pending spool."""
    parsed_metadata = _parse_metadata_option(metadata)
    try:
        job = submit_job(
            entry,
            argv=command,
            expected_run_names=run_name,
            device=device,
            cwd=cwd,
            metadata=parsed_metadata,
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    except QueueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"submitted {job.job_id} (sequence={job.sequence}, device={job.device})")


@queue_app.command("list")
def queue_list(
    entry: Annotated[Path, typer.Option("--entry", help="Queue entry root.")] = Path("./runs"),
) -> None:
    """Render pending, claimed, completed, failed, and interrupted queue state.

    A malformed pending or claimed job file is skipped and reported by
    filename rather than aborting the whole listing. A malformed completion
    journal, by contrast, is reported as a clean error with a nonzero exit
    after the healthy pending/claimed sections are still printed: Mammoth
    cannot safely determine completed/failed/interrupted state from a
    partially trusted journal, so it reports that explicitly instead of
    silently showing an empty (and misleadingly "nothing yet") history.
    """
    try:
        snapshot = list_jobs(entry)
    except QueueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(_render_queue_snapshot(snapshot))
    if snapshot.journal_error is not None:
        typer.echo(f"queue journal error: {snapshot.journal_error}", err=True)
        raise typer.Exit(code=1) from None


@queue_app.command("cancel")
def queue_cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID to cancel.")],
    entry: Annotated[Path, typer.Option("--entry", help="Queue entry root.")] = Path("./runs"),
) -> None:
    """Remove one unclaimed job; refuse a job already claimed or terminal."""
    try:
        job = cancel_job(entry, job_id)
    except JobNotFoundError as error:
        raise typer.BadParameter(str(error)) from None
    except JobAlreadyClaimedError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    except QueueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"cancelled {job.job_id}")


@queue_app.command("serve")
def queue_serve(
    device: Annotated[
        str,
        typer.Option("--device", help="Device spec this lane exclusively serves."),
    ],
    entry: Annotated[Path, typer.Option("--entry", help="Queue entry root.")] = Path("./runs"),
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            callback=positive_float,
            help="Idle poll interval in seconds.",
        ),
    ] = 1.0,
) -> None:
    """Run the foreground FIFO device-lane runner until interrupted.

    A malformed completion journal is refused with a clean error instead of
    starting this lane: silently skipping journal corruption risks
    double-running a job the journal had already durably recorded.
    """
    try:
        run_serve_loop(entry, device, poll_interval_seconds=poll_interval)
    except DeviceLeaseConflictError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    except QueueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None


def _parse_metadata_option(metadata: str | None) -> dict[str, Any] | None:
    """Parse the ``--metadata`` CLI option into an opaque JSON object."""
    if metadata is None:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"--metadata must be a JSON object: {error}") from None
    if not isinstance(parsed, dict):
        raise typer.BadParameter("--metadata must be a JSON object.")
    return parsed


def _render_queue_snapshot(snapshot: QueueSnapshot) -> str:
    """Render one plain, stable snapshot for passive queue-state readers."""
    lines: list[str] = []

    def render_job(status: str, job: Job) -> str:
        return f"{status:<12}{job.job_id} device={job.device} sequence={job.sequence}"

    def render_outcome(status: str, outcome: JobOutcome) -> str:
        return (
            f"{status:<12}{outcome.job_id} device={outcome.device} "
            f"return_code={outcome.return_code} signal={outcome.signal}"
        )

    lines.append(f"pending: {len(snapshot.pending)}")
    lines.extend(render_job("PENDING", job) for job in snapshot.pending)
    lines.append(f"claimed: {len(snapshot.claimed)}")
    lines.extend(render_job("CLAIMED", job) for job in snapshot.claimed)
    if snapshot.journal_error is not None:
        lines.append(f"journal: unreadable ({snapshot.journal_error})")
    else:
        lines.append(f"completed: {len(snapshot.completed)}")
        lines.extend(render_outcome("COMPLETED", outcome) for outcome in snapshot.completed)
        lines.append(f"failed: {len(snapshot.failed)}")
        lines.extend(render_outcome("FAILED", outcome) for outcome in snapshot.failed)
        lines.append(f"interrupted: {len(snapshot.interrupted)}")
        lines.extend(render_outcome("INTERRUPTED", outcome) for outcome in snapshot.interrupted)
    if snapshot.malformed_job_files:
        lines.append(f"malformed job files: {len(snapshot.malformed_job_files)}")
        lines.extend(f"  {path}" for path in snapshot.malformed_job_files)
    return "\n".join(lines)


def _run_fleet_monitor(
    entry: Path,
    *,
    group: str | None,
    match: str | None,
    interactive: bool,
    should_watch: bool,
    include_telemetry: bool,
    interval: float,
    stale_after: float,
) -> None:
    """Open the fleet view, or one directly requested group within it."""
    fleet_monitor = FleetMonitor(entry, match=match)
    fleet_snapshot = fleet_monitor.poll(stale_after_seconds=stale_after)
    selected_group: GroupSnapshot | None = None
    if group is not None:
        selected_group = _find_group(fleet_snapshot, group)
        if selected_group is None:
            raise typer.BadParameter(
                f"No group {group!r} found under {entry}", param_hint="--group"
            )

    if interactive:
        try:
            textual_module = load_textual_ui()
        except ModuleNotFoundError as error:
            typer.echo(
                "Interactive monitoring requires the monitor extra; "
                "install it with `uv sync --extra monitor` or pass `--plain`.",
                err=True,
            )
            raise typer.Exit(code=2) from error
        textual_module.run_fleet_textual(
            entry,
            fleet_monitor,
            fleet_snapshot,
            watch=should_watch,
            telemetry=include_telemetry,
            interval_seconds=interval,
            stale_after_seconds=stale_after,
            open_group_id=group,
        )
        return

    while True:
        if selected_group is not None:
            sys.stdout.write(render_group_snapshot(selected_group, stale_after_seconds=stale_after))
        else:
            sys.stdout.write(render_fleet_snapshot(fleet_snapshot, stale_after_seconds=stale_after))
        if include_telemetry:
            sys.stdout.write(json.dumps(asdict(sample_viewer_telemetry()), sort_keys=True) + "\n")
        sys.stdout.flush()
        if not should_watch:
            return
        time.sleep(interval)
        fleet_snapshot = fleet_monitor.poll(stale_after_seconds=stale_after)
        if group is not None:
            selected_group = _find_group(fleet_snapshot, group) or selected_group


def _find_group(fleet_snapshot: FleetSnapshot, group_id: str) -> GroupSnapshot | None:
    """Return one group snapshot by ID, or ``None`` when it is not present."""
    return next((item for item in fleet_snapshot.groups if item.group_id == group_id), None)


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
