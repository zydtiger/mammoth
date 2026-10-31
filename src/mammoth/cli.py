"""Typer command-line entry point for Mammoth monitoring infrastructure.

This module keeps optional monitor UI imports out of base package startup. The
console script and ``python -m mammoth`` both enter through :func:`main`.
"""

import importlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, NoReturn, Union

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
)
from mammoth.monitor.documents import fleet_document, group_document, run_document
from mammoth.monitor.web import create_web_server
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
monitor_app = typer.Typer(
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
    help="Passive monitors: terminal dashboard, web server, and stable snapshots.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(monitor_app, name="monitor")


def version_callback(value: bool) -> None:
    """Print the package version for the eager root option."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def root(
    version: Annotated[
        Union[bool, None],
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


def load_textual_ui() -> ModuleType:
    """Load the optional Textual module only for an interactive invocation."""
    return importlib.import_module("mammoth.monitor.textual_ui")


EntryOption = Annotated[Path, typer.Option("--entry", help="Run-directory entry path.")]
GroupOption = Annotated[
    Union[str, None],
    typer.Option("--group", help="Exact group ID to open directly, without RUN."),
]
MatchOption = Annotated[
    Union[str, None],
    typer.Option(
        "--match",
        help="Glob pattern grouping loose runs ad hoc, for entries without group manifests.",
    ),
]
ExecutionOption = Annotated[
    Union[str, None], typer.Option("--execution", help="Exact immutable execution to inspect.")
]
RunArgument = Annotated[
    Union[str, None],
    typer.Argument(metavar="[RUN]", help="Logical run to inspect; omit for the fleet view."),
]
IntervalOption = Annotated[
    float,
    typer.Option("--interval", callback=positive_float, help="Polling interval in seconds."),
]
StaleAfterOption = Annotated[
    float,
    typer.Option(
        "--stale-after",
        callback=positive_float,
        help="Seconds without an observation before a running producer is stale.",
    ),
]


def _validate_monitor_target(
    run_name: Union[str, None],
    *,
    group: Union[str, None],
    match: Union[str, None],
    execution: Union[str, None],
) -> None:
    """Reject option combinations that do not name one monitor view."""
    if run_name is not None and (group is not None or match is not None):
        raise typer.BadParameter(
            "--group and --match apply only when RUN is omitted",
            param_hint="RUN",
        )
    if run_name is None and execution is not None:
        raise typer.BadParameter(
            "--execution applies only to a single-run invocation (RUN given)",
            param_hint="--execution",
        )
    if group is not None and match is not None:
        raise typer.BadParameter(
            "--group and --match are mutually exclusive",
            param_hint="--group",
        )


def _open_fleet(
    entry: Path,
    *,
    group: Union[str, None],
    match: Union[str, None],
    stale_after: float,
) -> tuple[FleetMonitor, FleetSnapshot]:
    """Take the first fleet snapshot and confirm a requested group exists."""
    fleet_monitor = FleetMonitor(entry, match=match)
    fleet_snapshot = fleet_monitor.poll(stale_after_seconds=stale_after)
    if group is not None and _find_group(fleet_snapshot, group) is None:
        raise typer.BadParameter(f"No group {group!r} found under {entry}", param_hint="--group")
    return fleet_monitor, fleet_snapshot


@monitor_app.command("tui")
def monitor_tui(
    run_name: RunArgument = None,
    entry: EntryOption = Path("./runs"),
    group: GroupOption = None,
    match: MatchOption = None,
    execution: ExecutionOption = None,
    watch: Annotated[
        bool,
        typer.Option("--watch/--no-watch", help="Enable or disable continuous polling."),
    ] = True,
    telemetry: Annotated[
        bool,
        typer.Option("--telemetry/--no-telemetry", help="Show viewer-host telemetry."),
    ] = True,
    interval: IntervalOption = 2.0,
    stale_after: StaleAfterOption = 90.0,
) -> None:
    """Open the interactive terminal dashboard."""
    _validate_monitor_target(run_name, group=group, match=match, execution=execution)
    run_monitor: Union[RunMonitor, None] = None
    if run_name is not None:
        try:
            run_monitor = RunMonitor(RunLayout(entry, run_name), execution)
            initial_snapshot = run_monitor.poll()
        except (OSError, ValueError) as error:
            raise typer.BadParameter(str(error), param_hint="RUN/--execution") from None
    else:
        fleet_monitor, fleet_snapshot = _open_fleet(
            entry, group=group, match=match, stale_after=stale_after
        )
    try:
        textual_module = load_textual_ui()
    except ModuleNotFoundError as error:
        typer.echo(
            "The terminal dashboard requires the monitor extra; "
            "install it with `uv sync --extra monitor`, or use `mammoth monitor show`.",
            err=True,
        )
        raise typer.Exit(code=2) from error
    if run_monitor is not None:
        textual_module.run_textual(
            run_monitor,
            initial_snapshot,
            watch=watch,
            telemetry=telemetry,
            interval_seconds=interval,
            stale_after_seconds=stale_after,
        )
        return
    textual_module.run_fleet_textual(
        entry,
        fleet_monitor,
        fleet_snapshot,
        watch=watch,
        telemetry=telemetry,
        interval_seconds=interval,
        stale_after_seconds=stale_after,
        open_group_id=group,
    )


@monitor_app.command("show")
def monitor_show(
    run_name: RunArgument = None,
    entry: EntryOption = Path("./runs"),
    group: GroupOption = None,
    match: MatchOption = None,
    execution: ExecutionOption = None,
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Keep printing snapshots until the run finishes."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print one JSON object per snapshot.")] = (
        False
    ),
    interval: IntervalOption = 2.0,
    stale_after: StaleAfterOption = 90.0,
) -> None:
    """Print a stable plain-text or JSON snapshot."""
    _validate_monitor_target(run_name, group=group, match=match, execution=execution)
    if run_name is None:
        _show_fleet(
            entry,
            group=group,
            match=match,
            watch=watch,
            as_json=as_json,
            interval=interval,
            stale_after=stale_after,
        )
        return
    try:
        layout = RunLayout(entry, run_name)
        monitor = ExecutionMonitor(layout, execution)
        run_monitor = RunMonitor(layout, execution) if as_json else None
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="RUN/--execution") from None
    while True:
        snapshot = monitor.poll()
        if run_monitor is not None:
            document = run_document(run_monitor.poll(), stale_after_seconds=stale_after)
            sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
        else:
            sys.stdout.write(render_snapshot(snapshot, stale_after_seconds=stale_after))
        sys.stdout.flush()
        if not watch or snapshot.status in {"completed", "failed", "interrupted"}:
            return
        time.sleep(interval)


@monitor_app.command("web")
def monitor_web(
    root: Annotated[
        Union[list[Path], None],
        typer.Option("--root", help="Directory searched for entries; repeatable. [default: runs]"),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, max=65535, help="Port to bind.")] = 8765,
    interval: IntervalOption = 2.0,
    stale_after: StaleAfterOption = 90.0,
) -> None:
    """Serve the read-only web monitor for every entry below each root."""
    roots = tuple(root) if root else (Path("./runs"),)
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise typer.BadParameter(f"Not a directory: {', '.join(missing)}", param_hint="--root")
    try:
        server = create_web_server(
            roots,
            host=host,
            port=port,
            interval_seconds=interval,
            stale_after_seconds=stale_after,
        )
    except OSError as error:
        typer.echo(f"Could not bind {host}:{port}: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Serving {', '.join(map(str, roots))} at {server.url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown_monitor()


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
        Union[Path, None], typer.Option("--cwd", help="Working directory for the launched job.")
    ] = None,
    metadata: Annotated[
        Union[str, None],
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


def _parse_metadata_option(metadata: Union[str, None]) -> Union[dict[str, Any], None]:
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


def _show_fleet(
    entry: Path,
    *,
    group: Union[str, None],
    match: Union[str, None],
    watch: bool,
    as_json: bool,
    interval: float,
    stale_after: float,
) -> None:
    """Print the fleet view, or one directly requested group within it."""
    fleet_monitor, fleet_snapshot = _open_fleet(
        entry, group=group, match=match, stale_after=stale_after
    )
    selected: Union[GroupSnapshot, None] = None
    while True:
        if group is not None:
            selected = _find_group(fleet_snapshot, group) or selected
        if as_json:
            document = (
                group_document(selected) if selected is not None else fleet_document(fleet_snapshot)
            )
            sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
        elif selected is not None:
            sys.stdout.write(render_group_snapshot(selected, stale_after_seconds=stale_after))
        else:
            sys.stdout.write(render_fleet_snapshot(fleet_snapshot, stale_after_seconds=stale_after))
        sys.stdout.flush()
        if not watch:
            return
        time.sleep(interval)
        fleet_snapshot = fleet_monitor.poll(stale_after_seconds=stale_after)


def _find_group(fleet_snapshot: FleetSnapshot, group_id: str) -> Union[GroupSnapshot, None]:
    """Return one group snapshot by ID, or ``None`` when it is not present."""
    return next((item for item in fleet_snapshot.groups if item.group_id == group_id), None)


def run(argv: Union[Sequence[str], None] = None) -> int:
    """Execute one Typer command and return its process exit code."""
    try:
        app(args=list(argv) if argv is not None else None, prog_name="mammoth")
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    return 0


def main() -> NoReturn:
    """Terminate the console process with :func:`run`'s status."""
    raise SystemExit(run())
