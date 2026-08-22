"""Generic subprocess construction and supervision for programmatic workflows.

The workflow runner delegates each caller-complete argv here. Start-new-session
process groups ensure timeout or interruption reaches descendants as a group.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Never, Protocol

_PR_SET_PDEATHSIG = 1  # Linux <sys/prctl.h>; unavailable on other kernels.


@dataclass(frozen=True, slots=True)
class CommandPlan:
    """Fully resolved but unexecuted child command."""

    run_name: str
    step_name: str
    command: tuple[str, ...]
    cwd: Path | None
    timeout_seconds: float | None
    run_dir: Path


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Terminal outcome from one supervised process group."""

    return_code: int
    duration_seconds: float
    timed_out: bool = False
    interrupted: bool = False
    signal: int | None = None


class Launcher(Protocol):
    """Structural contract for substituting :func:`launch_process` process creation.

    A caller-supplied launcher only replaces child-process construction. Signal
    handling, lease ownership, lifecycle-JSONL emission, and cleanup remain
    owned by :mod:`mammoth.workflow.runner`.
    """

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> ProcessResult:
        """Launch one already-final argv and return its terminal outcome."""
        ...


@dataclass(frozen=True, slots=True)
class CapturedProcessResult:
    """Terminal facts and separate text streams from one supervised process."""

    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    timed_out: bool = False
    signal: int | None = None


class _DeferredProcessSignal(BaseException):
    """One termination signal deferred until child ownership is registered."""

    def __init__(self, signal_number: int, previous_handler: Any) -> None:
        self.signal_number = signal_number
        self.previous_handler = previous_handler


@dataclass(slots=True)
class SupervisedProcess:
    """Own one child launcher and bounded cleanup of its observable descendants.

    Project runners may use this object directly when they need to publish the
    child PID or surround launch and terminal writes with their own atomic state
    transitions. :func:`launch_process` provides the simpler one-call facade.
    """

    command: tuple[str, ...]
    cwd: Path | None
    environment: Mapping[str, str]
    start_new_session: bool = True
    terminate_grace_seconds: float = 5.0
    descendant_grace_seconds: float = 1.0
    parent_death_signal: int | None = None
    process_factory: Callable[..., subprocess.Popen[Any]] = field(
        default=subprocess.Popen,
        repr=False,
    )
    _process: subprocess.Popen[Any] | None = field(default=None, init=False, repr=False)

    @property
    def pid(self) -> int:
        """Return the launched process ID, rejecting access before :meth:`start`."""
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        """Return the child status, or ``None`` before launch or while running."""
        return None if self._process is None else self._process.returncode

    @property
    def process(self) -> subprocess.Popen[Any]:
        """Return the underlying process after a successful launch."""
        if self._process is None:
            raise RuntimeError("process has not been started")
        return self._process

    def start(self) -> SupervisedProcess:
        """Launch exactly once and return this supervisor for fluent use."""
        return self._start(capture_output=False)

    def _start(self, *, capture_output: bool) -> SupervisedProcess:
        """Launch once, optionally configuring the pipes for the one-call facade."""
        if self._process is not None:
            raise RuntimeError("process has already been started")
        options: dict[str, Any] = {
            "cwd": self.cwd,
            "env": dict(self.environment),
        }
        if self.start_new_session:
            options["start_new_session"] = True
        if self.parent_death_signal is not None:
            # Captured here, in the parent, before Popen() forks: this is the
            # only point at which os.getpid() reliably names the process the
            # armed death signal is actually supposed to track. Capturing it
            # instead inside the forked child's preexec_fn (as an earlier
            # revision of this mitigation did) reads os.getppid() *after*
            # the fork, which already observes any reparenting that raced
            # ahead of it, defeating the very race check it was meant to
            # perform. See _build_parent_death_signal_preexec_fn.
            options["preexec_fn"] = _build_parent_death_signal_preexec_fn(
                self.parent_death_signal,
                os.getpid(),
            )
        if capture_output:
            options.update(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self._process = self.process_factory(self.command, **options)
        return self

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the child and return its terminal status."""
        return self.process.wait(timeout=timeout)

    def stop(self) -> int | None:
        """Terminate, escalate, and reap the launcher and observable descendants."""
        return terminate_process_tree(
            self.process,
            new_session=self.start_new_session,
            grace_seconds=self.terminate_grace_seconds,
            descendant_grace_seconds=self.descendant_grace_seconds,
        )


def launch_process(
    command: tuple[str, ...],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
    terminate_grace_seconds: float = 5.0,
    on_started: Callable[[int], None] | None = None,
    parent_death_signal: int | None = None,
) -> ProcessResult:
    """Run one command and terminate its process group on timeout or interruption.

    ``on_started`` receives the real launcher PID after Mammoth has started
    the supervised child and before it waits.  A callback failure follows the
    same owned cleanup path as any other launch-time failure.

    ``parent_death_signal`` is an opt-in, Linux-only, best-effort mitigation:
    when set, the direct child is armed (via ``prctl(PR_SET_PDEATHSIG, ...)``)
    to receive that signal if this process itself terminates abruptly (for
    example ``SIGKILL``) before it can otherwise supervise the child's
    shutdown. The default, ``None``, adds no ``preexec_fn`` and leaves every
    existing caller's behavior, including :meth:`mammoth.workflow.Workflow.run`,
    byte-identical. This only reaches the *direct* child process: a
    grandchild the job itself spawns into its own session is unaffected and
    can still outlive a hard-killed launcher. See
    ``docs/ARCHITECTURE.md``'s "Runner and crash safety" section for the full
    residual-risk disclosure in the one caller that opts in,
    :func:`mammoth.queue.serve.serve_once`.
    """
    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    process = SupervisedProcess(
        command,
        cwd=cwd,
        environment=environment,
        terminate_grace_seconds=terminate_grace_seconds,
        parent_death_signal=parent_death_signal,
    )
    process_started = False
    try:
        with _defer_process_start_signals():
            process.start()
            process_started = True
        if on_started is not None:
            on_started(process.pid)
        remaining_timeout = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        return_code = process.wait(timeout=remaining_timeout)
    except _DeferredProcessSignal as interruption:
        if _supervisor_started(process, acknowledged=process_started):
            with _defer_termination_signals():
                process.stop()
        return ProcessResult(
            return_code=(
                process.returncode
                if _supervisor_started(process, acknowledged=process_started)
                and process.returncode is not None
                else -interruption.signal_number
            ),
            duration_seconds=time.monotonic() - started,
            interrupted=True,
            signal=interruption.signal_number,
        )
    except subprocess.TimeoutExpired:
        with _defer_termination_signals():
            process.stop()
        return ProcessResult(
            return_code=process.returncode if process.returncode is not None else -signal.SIGKILL,
            duration_seconds=time.monotonic() - started,
            timed_out=True,
            signal=signal.SIGTERM,
        )
    except KeyboardInterrupt:
        with _defer_termination_signals():
            process.stop()
        return ProcessResult(
            return_code=process.returncode if process.returncode is not None else -signal.SIGINT,
            duration_seconds=time.monotonic() - started,
            interrupted=True,
            signal=signal.SIGINT,
        )
    except BaseException:
        # Programmatic workflow runners translate termination signals into a
        # caller-owned interruption type.  Reap the owned child before that
        # type reaches their attempt/lease cleanup path.
        if _supervisor_started(process, acknowledged=process_started):
            with _defer_termination_signals():
                process.stop()
        raise
    timed_out = deadline is not None and time.monotonic() > deadline
    return ProcessResult(
        return_code=return_code,
        duration_seconds=time.monotonic() - started,
        timed_out=timed_out,
        signal=-return_code if return_code < 0 else None,
    )


def run_captured_process(
    command: tuple[str, ...],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
    terminate_grace_seconds: float = 5.0,
    descendant_grace_seconds: float = 1.0,
) -> CapturedProcessResult:
    """Run one command with separately captured text output and bounded cleanup.

    This one-call facade owns pipe configuration and uses :class:`SupervisedProcess`
    for the existing launcher and descendant cleanup policy. A timeout is a process
    fact, not a success policy: partial output is returned after the process tree is
    reaped. Other exceptions, including ``KeyboardInterrupt``, are propagated only
    after the owned process tree has been stopped.
    """
    started = time.monotonic()
    supervisor = SupervisedProcess(
        command,
        cwd=cwd,
        environment=environment,
        terminate_grace_seconds=terminate_grace_seconds,
        descendant_grace_seconds=descendant_grace_seconds,
    )
    process: subprocess.Popen[str] | None = None
    process_started = False
    try:
        with _defer_process_start_signals():
            process = supervisor._start(capture_output=True).process
            process_started = True
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except _DeferredProcessSignal as interruption:
        if _supervisor_started(supervisor, acknowledged=process_started):
            with _defer_termination_signals():
                supervisor.stop()
        _redeliver_deferred_process_signal(interruption)
    except subprocess.TimeoutExpired:
        with _defer_termination_signals():
            supervisor.stop()
        assert process is not None
        # A second communicate returns the complete stream contents. Do not combine
        # TimeoutExpired.output with it: subprocess may expose overlapping data.
        stdout, stderr = process.communicate()
        return_code = process.returncode if process.returncode is not None else -signal.SIGKILL
        return CapturedProcessResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            duration_seconds=time.monotonic() - started,
            timed_out=True,
            signal=-return_code if return_code < 0 else None,
        )
    except BaseException:
        if _supervisor_started(supervisor, acknowledged=process_started):
            with _defer_termination_signals():
                supervisor.stop()
        raise

    return_code = process.returncode if process.returncode is not None else -signal.SIGKILL
    return CapturedProcessResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        duration_seconds=time.monotonic() - started,
        signal=-return_code if return_code < 0 else None,
    )


def _build_parent_death_signal_preexec_fn(
    signal_number: int,
    expected_parent_pid: int,
) -> Callable[[], None]:
    """Return a preexec hook that best-effort arms a Linux parent-death signal.

    Called in the freshly forked child, before exec, from
    :meth:`SupervisedProcess._start`. Uses ``prctl(PR_SET_PDEATHSIG, ...)``
    via ``ctypes`` since the standard library exposes no wrapper. Any
    failure -- including running on a non-Linux kernel where ``libc.so.6``
    or ``prctl`` itself is unavailable -- is swallowed: this is a
    best-effort mitigation, never a launch precondition, and an exception
    escaping a ``preexec_fn`` would otherwise abort the whole launch.

    ``expected_parent_pid`` must be captured by the *caller*, in the parent
    process, with ``os.getpid()`` before ``Popen()`` forks -- not read as
    ``os.getppid()`` from inside this hook. By the time this hook runs, the
    fork has already happened, so ``os.getppid()`` here only ever observes
    the *current* parent, which already reflects any reparenting that raced
    ahead of it; comparing it against itself can never detect a mismatch.
    After arming, this hook re-checks ``os.getppid()`` against the
    pre-captured value: ``prctl(2)``'s NOTES section documents that if the
    original parent already exited in the narrow fork-to-prctl window, the
    death signal would be armed against whatever process has since become
    the parent (usually the init process) instead of the one this launch
    actually cares about, silently losing the protection. Detecting that
    and delivering ``signal_number`` to this process immediately instead
    closes that race the same way the manual page recommends.
    """

    def _preexec() -> None:
        with suppress(Exception):
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl.restype = ctypes.c_int
            libc.prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            libc.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(signal_number), 0, 0, 0)
            if os.getppid() != expected_parent_pid:
                os.kill(os.getpid(), signal_number)

    return _preexec


def _supervisor_started(
    supervisor: SupervisedProcess,
    *,
    acknowledged: bool,
) -> bool:
    """Detect a child assigned by ``_start`` before that method raised."""
    return acknowledged or getattr(supervisor, "_process", None) is not None


@contextmanager
def _defer_process_start_signals() -> Iterator[None]:
    """Record SIGINT/SIGTERM until a newly launched child is safely owned."""
    if not hasattr(signal, "SIGTERM"):
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}
    installed: list[signal.Signals] = []
    deferred_signal: int | None = None
    body_error: BaseException | None = None
    restore_error: BaseException | None = None

    def defer(signal_number: int, _frame: object) -> None:
        nonlocal deferred_signal
        previous_handler = previous.get(signal.Signals(signal_number))
        if previous_handler == signal.SIG_IGN:
            return
        if deferred_signal is None:
            deferred_signal = signal_number

    try:
        for signal_number in signals:
            previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, defer)
            installed.append(signal_number)
    except ValueError:
        pass
    try:
        yield
    except BaseException as error:
        body_error = error
    finally:
        for signal_number in reversed(installed):
            try:
                signal.signal(signal_number, previous[signal_number])
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
    if deferred_signal is not None:
        deferred = _DeferredProcessSignal(
            deferred_signal,
            previous.get(signal.Signals(deferred_signal), signal.SIG_DFL),
        )
        if body_error is not None:
            raise deferred from body_error
        if restore_error is not None:
            raise deferred from restore_error
        raise deferred
    if restore_error is not None:
        if body_error is not None:
            raise restore_error from body_error
        raise restore_error
    if body_error is not None:
        raise body_error


def _redeliver_deferred_process_signal(interruption: _DeferredProcessSignal) -> Never:
    """Deliver a captured parent signal after the owned child has been reaped."""
    handler = interruption.previous_handler
    if callable(handler):
        handler(interruption.signal_number, None)
    else:
        signal.raise_signal(interruption.signal_number)
    raise interruption


@contextmanager
def _defer_termination_signals() -> Iterator[None]:
    """Keep a second termination request from interrupting child reaping."""
    if not hasattr(signal, "SIGTERM"):
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}
    installed: list[signal.Signals] = []

    def defer(_signal_number: int, _frame: object) -> None:
        return

    try:
        try:
            for signal_number in signals:
                previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, defer)
                installed.append(signal_number)
        except ValueError:
            yield
        else:
            yield
    finally:
        for signal_number in reversed(installed):
            signal.signal(signal_number, previous[signal_number])


def terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    new_session: bool,
    grace_seconds: float,
    descendant_grace_seconds: float = 1.0,
) -> int | None:
    """Stop and reap a launcher plus descendants, including nested sessions."""
    descendant_pids = set(descendant_process_ids(process.pid))
    if process.returncode is None:
        signal_launcher(process, signal.SIGTERM, new_session=new_session)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    except BaseException:
        pass
    if process.returncode is None:
        descendant_pids.update(descendant_process_ids(process.pid))
    stop_processes(
        tuple(descendant_pids),
        grace_seconds=descendant_grace_seconds,
    )
    if process.returncode is None:
        signal_launcher(process, signal.SIGKILL, new_session=new_session)
        with suppress(BaseException):
            process.wait()
    wait_for_processes_to_exit(
        tuple(descendant_pids),
        timeout=descendant_grace_seconds,
    )
    return process.returncode


def signal_launcher(
    process: subprocess.Popen[Any],
    signal_number: int,
    *,
    new_session: bool,
) -> None:
    """Signal a launcher process or its isolated process group."""
    try:
        if new_session:
            os.killpg(process.pid, signal_number)
        elif signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def descendant_process_ids(root_pid: int) -> tuple[int, ...]:
    """Snapshot Linux procfs descendants, returning an empty set elsewhere."""
    pending = [root_pid]
    descendants: list[int] = []
    seen = {root_pid}
    while pending:
        parent_pid = pending.pop()
        task_dir = Path("/proc") / str(parent_pid) / "task"
        try:
            children_files = tuple(task_dir.glob("*/children"))
        except OSError:
            continue
        for children_file in children_files:
            try:
                child_ids = children_file.read_text(encoding="utf-8").split()
            except OSError:
                continue
            for child_id in child_ids:
                try:
                    child_pid = int(child_id)
                except ValueError:
                    continue
                if child_pid in seen:
                    continue
                seen.add(child_pid)
                descendants.append(child_pid)
                pending.append(child_pid)
    return tuple(descendants)


def running_processes(process_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return process IDs that still exist without requiring child ownership."""
    running: list[int] = []
    for process_id in process_ids:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        running.append(process_id)
    return tuple(running)


def wait_for_processes_to_exit(
    process_ids: tuple[int, ...],
    *,
    timeout: float,
) -> tuple[int, ...]:
    """Poll non-child descendants for a bounded best-effort shutdown."""
    deadline = time.monotonic() + timeout
    running = running_processes(process_ids)
    while running and time.monotonic() < deadline:
        time.sleep(0.01)
        running = running_processes(running)
    return running


def signal_processes(process_ids: tuple[int, ...], signal_number: int) -> None:
    """Signal a descendant snapshot from leaves toward its launcher."""
    for process_id in reversed(process_ids):
        with suppress(OSError, ProcessLookupError):
            os.kill(process_id, signal_number)


def stop_processes(process_ids: tuple[int, ...], *, grace_seconds: float) -> None:
    """Gracefully stop observable descendants, then force survivors."""
    running = running_processes(process_ids)
    if not running:
        return
    signal_processes(running, signal.SIGTERM)
    running = wait_for_processes_to_exit(running, timeout=grace_seconds)
    if not running:
        return
    signal_processes(running, signal.SIGKILL)
    wait_for_processes_to_exit(running, timeout=grace_seconds)
