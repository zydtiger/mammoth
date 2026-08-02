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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_group(process, grace_seconds=terminate_grace_seconds)
        return ProcessResult(
            return_code=process.returncode if process.returncode is not None else -signal.SIGKILL,
            duration_seconds=time.monotonic() - started,
            timed_out=True,
            signal=signal.SIGTERM,
        )
    except KeyboardInterrupt:
        terminate_process_group(process, grace_seconds=terminate_grace_seconds)
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


def terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> None:
    """Terminate then kill one child process group without targeting the caller."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()
