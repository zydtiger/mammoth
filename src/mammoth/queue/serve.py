"""Exclusive device-lane leasing and the foreground FIFO job runner.

One ``serve`` invocation exclusively leases a single device spec (mirroring
:func:`mammoth.core.execution.claim_logical_run_lease`'s advisory-flock
mechanics), then repeatedly claims the oldest pending job whose device
requirement matches, launches its argv as a supervised child through
:func:`mammoth.workflow.launch.launch_process`, and appends the outcome to
the shared completion journal. See ``docs/ARCHITECTURE.md`` for the full
crash-safety contract this module implements.

Signal ownership is deliberately lighter than
:meth:`mammoth.workflow.Workflow.run`: a job's queue state is always either
untouched pending, or already durably claimed (via one atomic rename) before
its child ever launches, so no signal window can leave the spool in an
ambiguous state. ``launch_process`` already terminates a running child's
process group and reports an interrupted outcome on SIGINT; this module only
additionally stops the outer poll loop after such an outcome, and tolerates
an ordinary default-action SIGTERM (or any other abrupt process death)
because the OS releases this lane's device lease automatically and the next
``serve`` invocation for the same device reconciles any job left claimed
mid-flight as ``interrupted`` before resuming FIFO dispatch.
"""

from __future__ import annotations

import fcntl
import os
import signal
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mammoth.core.identity import validate_device_spec
from mammoth.core.layout import QueueLayout
from mammoth.queue.spool import (
    QUEUE_JOURNAL_SCHEMA_VERSION,
    Job,
    JobOutcome,
    JobStatus,
    QueueError,
    append_job_outcome,
    job_filename,
    journal_job_ids,
    load_job,
)
from mammoth.workflow.launch import Launcher, ProcessResult, launch_process

_MAMMOTH_ENVIRONMENT_PREFIX = "MAMMOTH_"


def _default_job_launcher(
    command: tuple[str, ...],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
) -> ProcessResult:
    """Launch one job with a best-effort Linux parent-death SIGTERM armed.

    This is ``serve_once``'s and ``run_serve_loop``'s default ``launcher``.
    It matches the :class:`~mammoth.workflow.launch.Launcher` protocol
    exactly (no new parameter on the protocol itself, so every existing
    launcher -- including test doubles -- keeps working unchanged) and only
    additionally passes ``parent_death_signal`` to :func:`launch_process`.
    See ``docs/ARCHITECTURE.md``'s "Runner and crash safety" section for the
    residual risk this does and does not cover.
    """
    return launch_process(
        command,
        cwd=cwd,
        environment=environment,
        timeout_seconds=timeout_seconds,
        parent_death_signal=signal.SIGTERM,
    )


class DeviceLeaseConflictError(QueueError):
    """Raised when a second ``serve`` process tries to lease an owned device."""


@dataclass
class DeviceLease:
    """Hold exclusive lane ownership of one device spec until released."""

    device: str
    path: Path
    _descriptor: int
    _closed: bool = False

    def close(self) -> None:
        """Release lane ownership while retaining the harmless lock inode."""
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> DeviceLease:
        """Return this lease for process-lifetime context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the device lease after the lane terminates."""
        self.close()


def claim_device_lease(entry: Path, device: str) -> DeviceLease:
    """Claim the exclusive advisory lease for one device lane without waiting.

    Mirrors :func:`mammoth.core.execution.claim_logical_run_lease` exactly:
    a second concurrent claim for the same device fails closed instead of
    blocking, and an OS-released flock (process crash or exit) frees the
    lane for the next ``serve`` invocation.
    """
    validated_device = validate_device_spec(device)
    layout = QueueLayout(entry).prepare()
    path = layout.lane_lease_path(validated_device)
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise QueueError(f"Device lease path must be a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeviceLeaseConflictError(
                f"Device {validated_device!r} is already served by another lane: {path}"
            ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return DeviceLease(device=validated_device, path=path, _descriptor=descriptor)


def reconcile_interrupted_jobs(entry: Path, device: str) -> tuple[JobOutcome, ...]:
    """Classify any job left claimed by a crashed prior holder of ``device``.

    Called once, immediately after a fresh :func:`claim_device_lease`
    succeeds and before FIFO dispatch resumes. Because the lease was just
    exclusively acquired, no other process can currently hold it, so any job
    file remaining in this lane's claimed directory can only be debris from
    a runner that died before recording that job's outcome, or debris from a
    runner that died *after* durably recording the outcome but before
    removing the claimed file. The completion journal is authoritative: a
    job ID already present there is stale cleanup, silently removed with no
    new record; every other leftover is durably recorded ``interrupted``
    before its claimed file is removed, so a crashed job is never silently
    re-run or silently lost.
    """
    layout = QueueLayout(entry)
    lane_dir = layout.claimed_lane_dir(device)
    if not lane_dir.is_dir():
        return ()
    already_terminal = journal_job_ids(layout.journal_path)
    outcomes: list[JobOutcome] = []
    for claimed_path in sorted(lane_dir.glob("*.json")):
        try:
            job = load_job(claimed_path)
        except QueueError:
            # Malformed debris cannot be attributed to a job ID; leave it for
            # manual inspection rather than guessing at a journal entry.
            continue
        if job.job_id in already_terminal:
            with suppress(OSError):
                claimed_path.unlink()
            continue
        outcome = JobOutcome(
            schema_version=QUEUE_JOURNAL_SCHEMA_VERSION,
            job_id=job.job_id,
            sequence=job.sequence,
            device=device,
            status="interrupted",
            time=_utc_now_iso(),
            expected_run_names=job.expected_run_names,
            message="Runner terminated before this job's outcome was recorded.",
        )
        append_job_outcome(entry, outcome)
        with suppress(OSError):
            claimed_path.unlink()
        outcomes.append(outcome)
    return tuple(outcomes)


def serve_once(
    entry: Path,
    device: str,
    *,
    launcher: Launcher = _default_job_launcher,
) -> JobOutcome | None:
    """Claim, launch, and record the outcome of one FIFO-matching pending job.

    Returns ``None`` when no pending job currently matches ``device``.
    """
    job = _claim_next_job(entry, device)
    if job is None:
        return None
    cwd = Path(job.cwd) if job.cwd else None
    result = launcher(
        job.argv,
        cwd=cwd,
        environment=_job_environment(),
        timeout_seconds=None,
    )
    status: JobStatus
    if result.interrupted:
        status = "interrupted"
    elif result.return_code == 0:
        status = "completed"
    else:
        status = "failed"
    outcome = JobOutcome(
        schema_version=QUEUE_JOURNAL_SCHEMA_VERSION,
        job_id=job.job_id,
        sequence=job.sequence,
        device=device,
        status=status,
        time=_utc_now_iso(),
        expected_run_names=job.expected_run_names,
        return_code=result.return_code,
        signal=result.signal,
        duration_seconds=result.duration_seconds,
    )
    append_job_outcome(entry, outcome)
    claimed_path = QueueLayout(entry).claimed_lane_dir(device) / job_filename(job)
    with suppress(OSError):
        claimed_path.unlink()
    return outcome


def run_serve_loop(
    entry: Path,
    device: str,
    *,
    launcher: Launcher = _default_job_launcher,
    poll_interval_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    max_jobs: int | None = None,
    stop_when_idle: bool = False,
) -> tuple[JobOutcome, ...]:
    """Run the foreground device-lane loop: lease, reconcile, then FIFO dispatch.

    ``max_jobs`` and ``stop_when_idle`` exist for deterministic testing; the
    CLI's ``mammoth queue serve`` leaves both at their defaults so the
    process serves its lane indefinitely until interrupted. The loop stops
    immediately after a job outcome classified ``interrupted`` by a live
    signal (see the module docstring), and swallows a ``KeyboardInterrupt``
    raised while idle between polls so an interactive SIGINT always exits
    this foreground command cleanly.
    """
    with claim_device_lease(entry, device) as lease:
        reconcile_interrupted_jobs(entry, lease.device)
        outcomes: list[JobOutcome] = []
        try:
            while max_jobs is None or len(outcomes) < max_jobs:
                outcome = serve_once(entry, lease.device, launcher=launcher)
                if outcome is not None:
                    outcomes.append(outcome)
                    if outcome.status == "interrupted":
                        break
                    continue
                if stop_when_idle:
                    break
                sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            pass
        return tuple(outcomes)


def _claim_next_job(entry: Path, device: str) -> Job | None:
    """Atomically claim the oldest pending job whose device spec matches.

    Uses ``os.rename`` from the pending directory directly into this lane's
    claimed directory: on a local filesystem this is atomic, and a source
    path that has already vanished (claimed by a racing lane on the same
    device spec) raises ``FileNotFoundError`` instead of silently
    overwriting anything, so exactly one lane ever wins a given job.
    """
    layout = QueueLayout(entry)
    if not layout.pending_dir.is_dir():
        return None
    lane_dir = layout.claimed_lane_dir(device)
    lane_dir.mkdir(parents=True, exist_ok=True)
    for candidate_path in sorted(layout.pending_dir.glob("*.json")):
        try:
            job = load_job(candidate_path)
        except QueueError:
            continue
        if job.device != device:
            continue
        destination = lane_dir / candidate_path.name
        try:
            os.rename(candidate_path, destination)
        except FileNotFoundError:
            continue
        return job
    return None


def _job_environment() -> dict[str, str]:
    """Strip Mammoth-owned variables so a claimed job's argv stays self-describing.

    Deliberately does not export ``CUDA_VISIBLE_DEVICES`` or any other
    device-selection variable: the lane's device spec is opaque to Mammoth
    (see the module docstring and the open-question resolution recorded in
    ``docs/ARCHITECTURE.md``), so a job's own argv is responsible for
    targeting the device it declared, exactly as the workflow layer already
    leaves device placement to the caller.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(_MAMMOTH_ENVIRONMENT_PREFIX)
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
