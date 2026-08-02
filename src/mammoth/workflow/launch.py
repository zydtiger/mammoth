"""Generic local and torchrun subprocess construction and supervision.

The workflow runner delegates one command at a time here. Start-new-session
process groups ensure timeout or interruption reaches descendants as a group.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mammoth.workflow.config import StepConfig


@dataclass(frozen=True, slots=True)
class CommandPlan:
    """Fully resolved but unexecuted child command."""

    run_name: str
    step_name: str
    command: tuple[str, ...]
    cwd: Path | None
    timeout_seconds: float | None


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Terminal outcome from one supervised process group."""

    return_code: int
    duration_seconds: float
    timed_out: bool = False
    interrupted: bool = False
    signal: int | None = None


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
        if self._process is not None:
            raise RuntimeError("process has already been started")
        options: dict[str, Any] = {
            "cwd": self.cwd,
            "env": dict(self.environment),
        }
        if self.start_new_session:
            options["start_new_session"] = True
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


def command_for_step(step: StepConfig) -> tuple[str, ...]:
    """Return a local command or standard ``torch.distributed.run`` command."""
    if step.launcher == "local":
        return step.command
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(step.processes),
        *step.torchrun_args,
        *step.command,
    )


def launch_process(
    command: tuple[str, ...],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
    terminate_grace_seconds: float = 5.0,
) -> ProcessResult:
    """Run one command and terminate its process group on timeout or interruption."""
    started = time.monotonic()
    process = SupervisedProcess(
        command,
        cwd=cwd,
        environment=environment,
        terminate_grace_seconds=terminate_grace_seconds,
    ).start()
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.stop()
        return ProcessResult(
            return_code=process.returncode if process.returncode is not None else -signal.SIGKILL,
            duration_seconds=time.monotonic() - started,
            timed_out=True,
            signal=signal.SIGTERM,
        )
    except KeyboardInterrupt:
        process.stop()
        return ProcessResult(
            return_code=process.returncode if process.returncode is not None else -signal.SIGINT,
            duration_seconds=time.monotonic() - started,
            interrupted=True,
            signal=signal.SIGINT,
        )
    return ProcessResult(
        return_code=return_code,
        duration_seconds=time.monotonic() - started,
        signal=-return_code if return_code < 0 else None,
    )


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
