"""Typer command-line entry point for Mammoth monitoring infrastructure.

This module keeps optional monitor UI imports out of base package startup. The
console script and ``python -m mammoth`` both enter through :func:`main`.
"""

import importlib
import json
import logging
import math
import sqlite3
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
from mammoth.monitor.web import create_queue_monitor_server, create_web_server
from mammoth.queue import (
    Dependency,
    JobSpec,
    JobValidationError,
    QueueClient,
    QueueError,
    ServiceConfig,
    default_socket_path,
    default_state_dir,
    format_size,
    parse_duration,
    parse_size,
    run_service,
)
from mammoth.queue.model import check_local_paths

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
    help="Host-wide GPU job queue: a scheduler service and its client commands.",
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


SocketOption = Annotated[
    Union[Path, None], typer.Option("--socket", help="Queue service API socket.")
]


@monitor_app.command("serve")
def monitor_serve(
    socket_path: SocketOption = None,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, max=65535, help="Port to bind.")] = 8765,
    allow_queue_writes: Annotated[
        bool,
        typer.Option(
            "--allow-queue-writes",
            help="Allow cancel and priority changes on this non-loopback --host address.",
        ),
    ] = False,
    interval: IntervalOption = 2.0,
    stale_after: StaleAfterOption = 90.0,
) -> None:
    """Serve the web monitor for every queued run; normally run by the mammoth-monitor service."""
    try:
        server = create_queue_monitor_server(
            QueueClient(socket_path) if socket_path is not None else None,
            host=host,
            port=port,
            interval_seconds=interval,
            stale_after_seconds=stale_after,
            allow_queue_writes=allow_queue_writes,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    except OSError as error:
        typer.echo(f"Could not bind {host}:{port}: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Serving queued runs at {server.url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown_monitor()


def _client(socket_path: Union[Path, None]) -> QueueClient:
    return QueueClient(socket_path if socket_path is not None else default_socket_path())


def _queue_failure(error: QueueError) -> NoReturn:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1) from None


@queue_app.command("submit")
def queue_submit(
    command: Annotated[
        list[str],
        typer.Argument(help="Opaque job argv; separate options from it with '--'."),
    ],
    cwd: Annotated[
        Union[Path, None],
        typer.Option("--cwd", help="Working directory; defaults to the current one."),
    ] = None,
    env: Annotated[
        Union[list[str], None],
        typer.Option("--env", help="KEY=VALUE for the job; repeatable. Shell env is not captured."),
    ] = None,
    env_file: Annotated[
        Union[list[Path], None],
        typer.Option("--env-file", help="Env file read at launch, for secrets; repeatable."),
    ] = None,
    gpus: Annotated[
        Union[int, None],
        typer.Option("--gpus", min=0, help="GPU count; defaults to the pin count, else 1."),
    ] = None,
    gpu: Annotated[
        Union[list[int], None],
        typer.Option("--gpu", help="Pin to a physical GPU index; repeatable."),
    ] = None,
    vram: Annotated[
        Union[str, None],
        typer.Option("--vram", help="VRAM reservation per GPU; unset reserves the whole GPU."),
    ] = None,
    ram: Annotated[Union[str, None], typer.Option("--ram", help="Host RAM reservation.")] = None,
    like: Annotated[
        Union[str, None],
        typer.Option("--like", help="Job ID or run whose measured peaks fill missing values."),
    ] = None,
    after: Annotated[
        Union[list[str], None],
        typer.Option("--after", help="Earlier job as ID, ID:ok, or ID:any; repeatable."),
    ] = None,
    group: Annotated[Union[str, None], typer.Option("--group", help="Group name.")] = None,
    priority: Annotated[int, typer.Option("--priority", help="Higher runs first.")] = 0,
    timeout: Annotated[
        Union[str, None],
        typer.Option("--timeout", help="Wall-clock limit such as 90m or 14d."),
    ] = None,
    run_dir: Annotated[
        Union[Path, None],
        typer.Option("--run-dir", help="Run directory, for monitor links and peak inheritance."),
    ] = None,
    socket_path: SocketOption = None,
) -> None:
    """Submit one opaque job; fails, keeping nothing, when the service is down."""
    try:
        spec = JobSpec(
            argv=tuple(command),
            cwd=str((cwd if cwd is not None else Path.cwd()).resolve()),
            env=_parse_env_options(env),
            env_files=tuple(str(path.resolve()) for path in env_file or ()),
            gpus=gpus if gpus is not None else len(gpu or ()) or 1,
            gpu=tuple(gpu or ()),
            vram=parse_size(vram) if vram is not None else None,
            ram=parse_size(ram) if ram is not None else None,
            like=like,
            after=tuple(Dependency.parse(item) for item in after or ()),
            group=group,
            priority=priority,
            timeout_seconds=parse_duration(timeout) if timeout is not None else None,
            run_dir=str(run_dir.resolve()) if run_dir is not None else None,
        )
        check_local_paths(spec)
    except JobValidationError as error:
        raise typer.BadParameter(str(error)) from None
    try:
        job = _client(socket_path).submit(spec.to_dict())
    except QueueError as error:
        _queue_failure(error)
    typer.echo(job["job_id"])


@queue_app.command("list")
def queue_list(
    as_json: Annotated[bool, typer.Option("--json", help="Print the API documents.")] = False,
    socket_path: SocketOption = None,
) -> None:
    """Show active and recent jobs, with a specific reason for every waiting job."""
    client = _client(socket_path)
    try:
        jobs, resources = client.list_jobs(), client.resources()
    except QueueError as error:
        _queue_failure(error)
    if as_json:
        typer.echo(json.dumps({"jobs": jobs, "resources": resources}, indent=2))
        return
    typer.echo(_render_queue(jobs, resources))


@queue_app.command("cancel")
def queue_cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID to cancel.")],
    socket_path: SocketOption = None,
) -> None:
    """Remove a pending job or stop a running one."""
    try:
        job = _client(socket_path).cancel(job_id)
    except QueueError as error:
        _queue_failure(error)
    typer.echo(f"{job_id} {'cancelled' if job['status'] == 'cancelled' else 'cancel requested'}")


@queue_app.command("priority")
def queue_priority(
    job_id: Annotated[str, typer.Argument(help="Pending job ID.")],
    priority: Annotated[int, typer.Argument(help="New priority; higher runs first.")],
    socket_path: SocketOption = None,
) -> None:
    """Change one pending job's priority."""
    try:
        _client(socket_path).set_priority(job_id, priority)
    except QueueError as error:
        _queue_failure(error)
    typer.echo(f"{job_id} priority {priority}")


@queue_app.command("wait")
def queue_wait(
    job_id: Annotated[Union[list[str], None], typer.Argument(help="Job IDs to wait for.")] = None,
    group: Annotated[Union[str, None], typer.Option("--group", help="Group to wait for.")] = None,
    reconnect_timeout: Annotated[
        str,
        typer.Option(
            "--reconnect-timeout",
            help="How long a down service is retried before giving up.",
        ),
    ] = "10m",
    socket_path: SocketOption = None,
) -> None:
    """Block until jobs end and print their outcomes.

    Exits 0 whatever the jobs' status, so read each job's status. Exits 1 only
    when the service stayed unreachable, which leaves job state unknown.
    """
    if not job_id and group is None:
        raise typer.BadParameter("Name job IDs or --group")
    try:
        jobs = _client(socket_path).wait(
            job_id or (), group, reconnect_seconds=parse_duration(reconnect_timeout)
        )
    except JobValidationError as error:
        raise typer.BadParameter(str(error)) from None
    except QueueError as error:
        _queue_failure(error)
    typer.echo(json.dumps({"jobs": jobs}, indent=2))


@queue_app.command("serve")
def queue_serve(
    state_dir: Annotated[
        Union[Path, None],
        typer.Option("--state-dir", help="Durable state directory."),
    ] = None,
    socket_path: SocketOption = None,
    tick: Annotated[
        float,
        typer.Option("--tick", callback=positive_float, help="Scheduler tick in seconds."),
    ] = 5.0,
    ram_headroom: Annotated[
        float,
        typer.Option("--ram-headroom", min=0.0, max=1.0, help="Host RAM fraction kept free."),
    ] = 0.1,
    ram_stable_window: Annotated[
        str,
        typer.Option("--ram-stable-window", help="Unknown-RAM job stability window."),
    ] = "5m",
    default_timeout: Annotated[
        str,
        typer.Option("--default-timeout", help="Limit for jobs that set no --timeout."),
    ] = "14d",
) -> None:
    """Run the queue service until SIGTERM; running jobs keep running after it stops."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = ServiceConfig(
            tick_seconds=tick,
            ram_headroom_fraction=ram_headroom,
            ram_stable_seconds=parse_duration(ram_stable_window),
            default_timeout_seconds=parse_duration(default_timeout),
        )
    except JobValidationError as error:
        raise typer.BadParameter(str(error)) from None
    try:
        run_service(
            state_dir if state_dir is not None else default_state_dir(),
            socket_path if socket_path is not None else default_socket_path(),
            config,
        )
    except (QueueError, OSError, sqlite3.Error) as error:
        typer.echo(f"Queue service failed: {error}", err=True)
        raise typer.Exit(code=1) from None


def _parse_env_options(values: Union[list[str], None]) -> dict[str, str]:
    env = {}
    for value in values or ():
        key, separator, content = value.partition("=")
        if not separator:
            raise JobValidationError(f"--env requires KEY=VALUE: {value!r}")
        env[key] = content
    return env


def _render_queue(jobs: Sequence[dict[str, Any]], resources: dict[str, Any]) -> str:
    """Render one plain queue listing."""
    lines = []
    for problem in ("sampling_error", "executor_error"):
        if resources.get(problem):
            lines.append(f"{problem}: {resources[problem]}")
    for gpu in resources.get("gpus", ()):
        owners = ",".join(gpu["jobs"]) or "-"
        lines.append(
            f"GPU{gpu['index']} free={format_size(gpu['free'])} total={format_size(gpu['total'])} "
            f"external={format_size(gpu['external'])} jobs={owners}"
        )
    ram = resources.get("ram")
    if ram is not None:
        lines.append(
            f"RAM admissible={format_size(max(0, ram['admissible']))} "
            f"available={format_size(ram['available'])} total={format_size(ram['total'])}"
        )
    for job in jobs:
        gpus = ",".join(map(str, job["gpu_indices"])) or "-"
        vram = format_size(job["reserved_vram"] or job["declared_vram"])
        vram = f"{vram}/{format_size(job['peak_vram'])}"
        memory = format_size(job["reserved_ram"] or job["declared_ram"])
        memory = f"{memory}/{format_size(job['peak_ram'])}"
        partial = " partial" if job["peak_partial"] else ""
        status = f"{job['status']}({job['label']})" if job["label"] else job["status"]
        detail = job["wait_reason"] or job["message"] or " ".join(job["argv"])
        lines.append(
            f"{job['job_id']} {status} pri={job['priority']} gpu={gpus} "
            f"vram={vram} ram={memory}{partial} group={job['group'] or '-'} {detail}"
        )
        lines.extend(f"  warning: {warning}" for warning in job["warnings"])
    return "\n".join(lines) if lines else "Queue is empty"


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
