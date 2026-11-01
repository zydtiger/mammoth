"""The queue service: owns job state, runs the scheduler tick, and recovers on restart.

One tick thread makes every launch and executor call. API threads only change the
store under the lock and wake that thread, so slow ``actl`` or ``nvidia-smi`` calls
never block a submit. See ``docs/ARCHITECTURE.md`` for the recovery contract.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from mammoth.queue.executor import ExecutionRecord, Executor, ExecutorError
from mammoth.queue.model import (
    TERMINAL_STATUSES,
    VISIBLE_DEVICES_VARIABLE,
    Job,
    JobNotFoundError,
    JobSpec,
    JobStateError,
    JobStatus,
    JobValidationError,
    QueueError,
    check_local_paths,
    format_size,
)
from mammoth.queue.sampling import HostSample, Sampler, SamplingError
from mammoth.queue.scheduler import (
    GpuState,
    Placement,
    Request,
    Reservation,
    gpu_accounting,
    ram_available,
    schedule,
)
from mammoth.queue.store import JobStore

logger = logging.getLogger(__name__)

RECENT_JOB_LIMIT = 100
MAX_WAIT_SECONDS = 30.0
_LEARNED_MARGIN = 1.1
_RAM_GROWTH_FACTOR = 1.05
_RECONCILE_EVERY_TICKS = 12
_SAMPLE_WRITE_SECONDS = 30.0
_SUBMIT_GRACE_SECONDS = 200.0
_TRUSTED_RUN_SECONDS = 1800.0
_OOM_LOG = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
_OUTCOMES: Mapping[str, JobStatus] = {
    "ok": "completed",
    "failed": "failed",
    "timeout": "failed",
    "skipped": "failed",
    "error": "failed",
    "cancelled": "cancelled",
    "abandoned": "interrupted",
}


@dataclass(frozen=True)
class ServiceConfig:
    """Operator settings for one service; every duration is in seconds."""

    tick_seconds: float = 5.0
    ram_headroom_fraction: float = 0.1
    ram_stable_seconds: float = 300.0
    default_timeout_seconds: int = 14 * 86400
    execution_retention_seconds: float = 7 * 86400.0
    partial_gap_seconds: float = 120.0


class QueueService:
    """Own the job store and every scheduling decision."""

    def __init__(
        self,
        store: JobStore,
        sampler: Sampler,
        executor: Executor,
        config: Union[ServiceConfig, None] = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._sampler = sampler
        self._executor = executor
        self._config = config or ServiceConfig()
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._wake = False
        self._stopped = False
        self._reconciled = False
        self._ticks = 0
        self._sample: Union[HostSample, None] = None
        self._sampling_error: Union[str, None] = None
        self._executor_error: Union[str, None] = None
        self._reasons: dict[str, str] = {}
        self._cancel_sent: set[str] = set()
        self._ram_anchor: dict[str, tuple[int, float]] = {}
        self._anon: dict[str, int] = {}
        self._observed: dict[str, dict[int, int]] = {}
        self._written_at: dict[str, float] = {}

    def submit(self, payload: Mapping[str, Any]) -> Job:
        """Validate and durably accept one job, then wake the scheduler."""
        spec = JobSpec.from_dict(payload)
        check_local_paths(spec)
        with self._condition:
            for dependency in spec.after:
                try:
                    self._store.get(dependency.job_id)
                except JobNotFoundError as error:
                    raise JobValidationError(f"Unknown dependency: {error}") from error
            if spec.like is not None and not self._store.matching(spec.like):
                raise JobValidationError(f"No job or run matches like: {spec.like}")
            self._check_capacity(spec)
            job = self._store.insert(uuid.uuid4().hex[:12], spec, self._clock())
            self._wake_locked()
            return job

    def get(self, job_id: str) -> Job:
        """Return one job."""
        with self._condition:
            return self._store.get(job_id)

    def list_jobs(self) -> list[Job]:
        """Return active jobs and recent terminal jobs in submission order."""
        with self._condition:
            return self._store.recent(RECENT_JOB_LIMIT)

    def run_dirs(self) -> list[str]:
        """Return every run directory any job ever named."""
        with self._condition:
            return self._store.run_dirs()

    def wait_reason(self, job_id: str) -> Union[str, None]:
        """Return why a pending job has not started, as of the latest tick."""
        with self._condition:
            return self._reasons.get(job_id)

    def cancel(self, job_id: str) -> Job:
        """Cancel a pending job now, or ask the tick to stop a live one."""
        with self._condition:
            job = self._store.get(job_id)
            if job.is_terminal:
                raise JobStateError(f"Job {job_id} already ended as {job.status}")
            if job.status == "pending":
                job = self._store.update(job_id, status="cancelled", finished_at=self._clock())
                self._reasons.pop(job_id, None)
            else:
                job = self._store.update(job_id, cancel_requested=1)
            self._wake_locked()
            return job

    def set_priority(self, job_id: str, priority: Any) -> Job:
        """Change a pending job's priority and reschedule."""
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise JobValidationError(f"Priority must be an integer: {priority!r}")
        with self._condition:
            job = self._store.get(job_id)
            if job.status != "pending":
                raise JobStateError(f"Job {job_id} is {job.status}; only pending jobs can change")
            job = self._store.update(job_id, priority=priority)
            self._wake_locked()
            return job

    def wait(
        self,
        job_ids: Sequence[str],
        group: Union[str, None],
        timeout_seconds: float,
    ) -> tuple[bool, list[Job]]:
        """Block until the named jobs all end or the bounded timeout elapses."""
        deadline = time.monotonic() + min(max(0.0, timeout_seconds), MAX_WAIT_SECONDS)
        with self._condition:
            while True:
                jobs = [self._store.get(job_id) for job_id in job_ids]
                if group is not None:
                    members = self._store.group(group)
                    if not members:
                        raise JobNotFoundError(f"Group not found: {group}")
                    jobs += members
                if not jobs:
                    raise JobNotFoundError("Nothing to wait for")
                done = all(job.is_terminal for job in jobs)
                remaining = deadline - time.monotonic()
                if done or remaining <= 0 or self._stopped:
                    return done, jobs
                self._condition.wait(remaining)

    def resources(self) -> dict[str, Any]:
        """Return the latest resource accounting for clients."""
        with self._condition:
            sample = self._sample
            document: dict[str, Any] = {
                "sampling_error": self._sampling_error,
                "executor_error": self._executor_error,
                "reconciled": self._reconciled,
                "gpus": [],
                "ram": None,
            }
            if sample is None:
                return document
            held = self._held_reservations()
            accounting = gpu_accounting(_gpu_states(sample), held)
            uuids = {gpu.index: gpu.uuid for gpu in sample.gpus}
            document["gpus"] = [
                {
                    "index": item.index,
                    "uuid": uuids[item.index],
                    "total": item.total,
                    "external": item.external,
                    "free": item.free,
                    "exclusive": item.exclusive,
                    "jobs": list(item.jobs),
                }
                for item in accounting
            ]
            headroom = self._headroom(sample)
            document["ram"] = {
                "total": sample.memory_total,
                "available": sample.memory_available,
                "headroom": headroom,
                "admissible": ram_available(sample.memory_available, headroom, held)[0],
            }
            return document

    def stop(self) -> None:
        """Ask the tick loop to return; running jobs are unaffected."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def run(self) -> None:
        """Tick until stopped, waking early for submit and cancel."""
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("Scheduler tick failed")
            with self._condition:
                if self._stopped:
                    return
                if not self._wake:
                    self._condition.wait(self._config.tick_seconds)
                self._wake = False
                if self._stopped:
                    return

    def tick(self) -> None:
        """Run one full pass: track executions, sample, resolve dependencies, launch."""
        self._ticks += 1
        tracked = self._track_executions()
        sample = self._read_sample()
        if sample is not None:
            self._measure(sample)
        with self._condition:
            self._resolve_dependencies()
            placements: tuple[Placement, ...] = ()
            if not tracked:
                self._block_pending(f"waiting: ACTL query failed: {self._executor_error}")
            elif sample is None:
                self._block_pending(f"waiting: GPU sampling failed: {self._sampling_error}")
            else:
                placements = self._decide(sample)
            self._condition.notify_all()
        if sample is not None and placements and self._executor_reachable():
            for placement in placements:
                self._launch(placement, sample)
        with self._condition:
            self._condition.notify_all()

    def _executor_reachable(self) -> bool:
        """Tell an executor outage from a rejection, which fails its job for good."""
        try:
            self._executor.find("mammoth-reachability-probe")
        except ExecutorError as error:
            with self._condition:
                self._executor_error = str(error)
                self._block_pending(f"waiting: ACTL query failed: {error}")
            return False
        return True

    def _wake_locked(self) -> None:
        self._wake = True
        self._condition.notify_all()

    def _track_executions(self) -> bool:
        """Reconcile launching and running jobs; ``False`` forbids launches this tick."""
        with self._condition:
            live = [job for job in self._store.active() if job.status != "pending"]
        full_pass = not self._reconciled or self._ticks % _RECONCILE_EVERY_TICKS == 0
        healthy = True
        error = None
        for job in live:
            try:
                self._track(job, full_pass)
            except ExecutorError as failure:
                healthy = False
                error = str(failure)
                logger.warning("Could not track job %s: %s", job.job_id, failure)
        with self._condition:
            self._executor_error = error
            if healthy:
                self._reconciled = True
        return healthy

    def _track(self, job: Job, full_pass: bool) -> None:
        if job.tx_id is None:
            tx_id = self._executor.find(job.execution_name)
            if tx_id is None:
                self._recover_absent(job)
                return
            with self._condition:
                job = self._store.update(
                    job.job_id,
                    tx_id=tx_id,
                    status="running",
                    started_at=self._clock(),
                    message=None,
                )
        tx_id = job.tx_id
        assert tx_id is not None
        with self._condition:
            job = self._store.get(job.job_id)
        if job.cancel_requested and job.job_id not in self._cancel_sent:
            try:
                self._executor.cancel(tx_id)
                self._cancel_sent.add(job.job_id)
            except ExecutorError as error:
                # Retried next tick; a stuck cancel must not stop other jobs launching.
                logger.warning("Could not cancel job %s: %s", job.job_id, error)
        if full_pass:
            self._collect_ignoring_absence(tx_id)
        record = self._executor.status(tx_id)
        if record is None:
            self._finish(
                job, "unknown", message="ACTL no longer has this execution; outcome unknown"
            )
        elif record.is_terminal:
            with suppress(ExecutorError):
                self._executor.collect(tx_id)
            self._finish_from_record(job, record)
        else:
            with self._condition:
                if job.status != "running":
                    self._store.update(job.job_id, status="running")

    def _collect_ignoring_absence(self, tx_id: str) -> None:
        try:
            self._executor.collect(tx_id)
        except ExecutorError:
            # A pruned record fails here too; status() below distinguishes the cases.
            logger.debug("Reconcile failed for %s", tx_id, exc_info=True)

    def _recover_absent(self, job: Job) -> None:
        """Settle a launching job the executor confirms it does not have."""
        with self._condition:
            if job.message is not None:
                self._finish(job, "failed", message=f"ACTL rejected the job: {job.message}")
            elif self._clock() - (job.launching_at or 0.0) < _SUBMIT_GRACE_SECONDS:
                # An orphaned submit from a killed service may still be registering.
                return
            elif job.cancel_requested:
                self._finish(job, "cancelled")
            elif (
                self._clock() - (job.launching_at or 0.0) > self._config.execution_retention_seconds
            ):
                self._finish(job, "unknown", message="No record remains past ACTL retention")
            else:
                self._store.update(
                    job.job_id,
                    status="pending",
                    gpu_indices=(),
                    reserved_vram=None,
                    reserved_ram=None,
                    launching_at=None,
                )
                logger.info("Returned job %s to pending: it never reached ACTL", job.job_id)

    def _finish_from_record(self, job: Job, record: ExecutionRecord) -> None:
        status = _OUTCOMES.get(record.status, "failed")
        label = None
        if status == "failed" and self._ran_out_of_memory(job, record):
            label = "oom"
        message = None if status == "completed" else f"ACTL status {record.status}"
        self._finish(
            job,
            status,
            exit_code=record.exit_code,
            label=label,
            message=message,
            finished_at=record.finished_at,
            started_at=record.started_at,
        )

    def _ran_out_of_memory(self, job: Job, record: ExecutionRecord) -> bool:
        memory = self._sampler.execution_memory(record.tx_id)
        if memory is not None and memory.oom_kills > 0:
            return True
        with self._condition:
            if self._store.get(job.job_id).label == "oom":
                return True
        try:
            return _OOM_LOG.search(self._executor.stderr(record.tx_id)) is not None
        except ExecutorError:
            return False

    def _finish(
        self,
        job: Job,
        status: JobStatus,
        *,
        exit_code: Union[int, None] = None,
        label: Union[str, None] = None,
        message: Union[str, None] = None,
        finished_at: Union[float, None] = None,
        started_at: Union[float, None] = None,
    ) -> None:
        with self._condition:
            current = self._store.get(job.job_id)
            finished = finished_at if finished_at is not None else self._clock()
            sampled = current.last_sampled_at
            partial = (
                current.peak_partial
                or sampled is None
                or finished - sampled > self._config.partial_gap_seconds
            )
            self._store.update(
                job.job_id,
                status=status,
                exit_code=exit_code,
                label=label,
                message=message,
                finished_at=finished,
                started_at=started_at if started_at is not None else current.started_at,
                peak_partial=int(partial),
            )
            for tracked in (self._ram_anchor, self._anon, self._observed, self._written_at):
                tracked.pop(job.job_id, None)
            self._cancel_sent.discard(job.job_id)

    def _read_sample(self) -> Union[HostSample, None]:
        try:
            sample = self._sampler.sample()
        except SamplingError as error:
            logger.warning("Sampling failed: %s", error)
            with self._condition:
                self._sample = None
                self._sampling_error = str(error)
            return None
        with self._condition:
            self._sample = sample
            self._sampling_error = None
        return sample

    def _measure(self, sample: HostSample) -> None:
        """Record each running job's observed memory and persist any new peak."""
        with self._condition:
            running = [
                job
                for job in self._store.active()
                if job.status == "running" and job.tx_id is not None
            ]
        now = self._clock()
        memory = {job.job_id: self._sampler.execution_memory(job.tx_id or "") for job in running}
        with self._condition:
            for job in running:
                observed = {
                    gpu.index: gpu.execution_vram[job.tx_id or ""]
                    for gpu in sample.gpus
                    if (job.tx_id or "") in gpu.execution_vram
                }
                self._observed[job.job_id] = observed
                facts = memory[job.job_id]
                changes: dict[str, Any] = {}
                previous = job.last_sampled_at
                if previous is not None and now - previous > self._config.partial_gap_seconds:
                    changes["peak_partial"] = 1
                vram_peak = max(observed.values(), default=0)
                if vram_peak > (job.peak_vram or 0):
                    changes["peak_vram"] = vram_peak
                if facts is not None:
                    self._anon[job.job_id] = facts.anon
                    if facts.anon > (job.peak_ram or 0):
                        changes["peak_ram"] = facts.anon
                    anchor = self._ram_anchor.get(job.job_id)
                    if anchor is None or facts.anon > anchor[0] * _RAM_GROWTH_FACTOR:
                        self._ram_anchor[job.job_id] = (facts.anon, now)
                    if facts.oom_kills > 0:
                        changes["label"] = "oom"
                else:
                    # Unreadable memory must not hold unknown-RAM jobs behind this one forever.
                    self._ram_anchor.setdefault(job.job_id, (0, now))
                    changes["peak_partial"] = 1
                due = now - self._written_at.get(job.job_id, 0.0) >= _SAMPLE_WRITE_SECONDS
                if changes or due:
                    changes["last_sampled_at"] = now
                    self._store.update(job.job_id, **changes)
                    self._written_at[job.job_id] = now

    def _resolve_dependencies(self) -> None:
        """Fail dependents of failed ``ok`` dependencies until nothing changes."""
        changed = True
        while changed:
            changed = False
            for job in self._store.active():
                if job.status != "pending":
                    continue
                for dependency in job.spec.after:
                    parent = self._store.get(dependency.job_id)
                    if (
                        dependency.mode == "ok"
                        and parent.is_terminal
                        and parent.status != "completed"
                    ):
                        self._store.update(
                            job.job_id,
                            status="dependency_failed",
                            finished_at=self._clock(),
                            message=f"Dependency {parent.job_id} ended as {parent.status}",
                        )
                        self._reasons.pop(job.job_id, None)
                        changed = True
                        break

    def _block_pending(self, reason: str) -> None:
        for job in self._store.active():
            if job.status == "pending":
                self._reasons[job.job_id] = reason

    def _decide(self, sample: HostSample) -> tuple[Placement, ...]:
        pending = [job for job in self._store.active() if job.status == "pending"]
        reasons: dict[str, str] = {}
        ready: list[Job] = []
        for job in pending:
            waiting = [
                dependency.job_id
                for dependency in job.spec.after
                if self._store.get(dependency.job_id).status not in TERMINAL_STATUSES
            ]
            if waiting:
                reasons[job.job_id] = f"waiting: after {', '.join(waiting)}"
            else:
                ready.append(job)
        ready.sort(key=lambda job: (-job.priority, job.sequence))
        held = self._held_reservations()
        headroom = self._headroom(sample)
        ram_capacity = sample.memory_total - headroom
        capacity = {item.index: item.capacity for item in gpu_accounting(_gpu_states(sample), held)}
        decision = schedule(
            [self._request(job, capacity, ram_capacity) for job in ready],
            _gpu_states(sample),
            sample.memory_available,
            headroom,
            held,
            ram_capacity,
        )
        reasons.update(decision.reasons)
        self._reasons = reasons
        return decision.placements

    def _request(self, job: Job, capacity: Mapping[int, int], ram_capacity: int) -> Request:
        spec = job.spec
        vram, ram = spec.vram, spec.ram
        if (vram is None and spec.gpus > 0) or ram is None:
            source = self._peak_source(spec)
            if source is not None:
                if vram is None and spec.gpus > 0 and source.peak_vram:
                    eligible = [
                        size
                        for index, size in capacity.items()
                        if not spec.gpu or index in spec.gpu
                    ]
                    vram = _with_margin(source.peak_vram, max(eligible, default=0))
                if ram is None and source.peak_ram:
                    ram = _with_margin(source.peak_ram, ram_capacity)
        return Request(job.job_id, spec.gpus, spec.gpu, vram, ram)

    def _peak_source(self, spec: JobSpec) -> Union[Job, None]:
        """Return the job whose measured peaks fill in missing reservations.

        An explicit ``like`` accepts any finished job the operator names except an
        out-of-memory one. Automatic ``run_dir`` inheritance needs a complete
        measurement, from a completed job or one that ended after running long
        enough to resume from, since an early failure understates the peak.
        """
        if spec.like is not None:
            for job in self._store.matching(spec.like):
                if job.is_terminal and job.label != "oom" and (job.peak_vram or job.peak_ram):
                    return job
            return None
        if spec.run_dir is not None:
            for job in self._store.by_run_dir(spec.run_dir):
                if job.is_terminal and job.label != "oom" and not job.peak_partial:
                    ran = (job.finished_at or 0.0) - (job.started_at or 0.0)
                    if job.status == "completed" or ran >= _TRUSTED_RUN_SECONDS:
                        return job
        return None

    def _held_reservations(self) -> list[Reservation]:
        held = []
        now = self._clock()
        for job in self._store.active():
            if job.status == "pending":
                continue
            anchor = self._ram_anchor.get(job.job_id)
            stable = anchor is not None and now - anchor[1] >= self._config.ram_stable_seconds
            held.append(
                Reservation(
                    job_id=job.job_id,
                    gpu_indices=job.gpu_indices,
                    vram=job.reserved_vram,
                    ram=job.reserved_ram,
                    observed_vram=self._observed.get(job.job_id, {}),
                    anon_ram=self._anon.get(job.job_id),
                    ram_stable=stable,
                )
            )
        return held

    def _headroom(self, sample: HostSample) -> int:
        return int(sample.memory_total * self._config.ram_headroom_fraction)

    def _launch(self, placement: Placement, sample: HostSample) -> None:
        now = self._clock()
        with self._condition:
            job = self._store.get(placement.job_id)
            if job.status != "pending":
                return
            job = self._store.update(
                job.job_id,
                status="launching",
                gpu_indices=placement.gpu_indices,
                reserved_vram=placement.vram,
                reserved_ram=placement.ram,
                launching_at=now,
            )
            self._reasons.pop(job.job_id, None)
        env = dict(job.spec.env)
        if job.spec.gpus > 0:
            uuids = {gpu.index: gpu.uuid for gpu in sample.gpus}
            env[VISIBLE_DEVICES_VARIABLE] = ",".join(uuids[i] for i in placement.gpu_indices)
        else:
            env.setdefault(VISIBLE_DEVICES_VARIABLE, "")
        timeout = job.spec.timeout_seconds or self._config.default_timeout_seconds
        try:
            tx_id = self._executor.start(
                job.execution_name, job.spec.argv, job.spec.cwd, env, job.spec.env_files, timeout
            )
        except Exception as error:
            # The submission may or may not have been accepted; settle it by name.
            logger.warning("Could not start job %s: %s", job.job_id, error)
            with self._condition:
                self._store.update(job.job_id, message=str(error))
            return
        with self._condition:
            self._store.update(
                job.job_id, status="running", tx_id=tx_id, started_at=self._clock(), message=None
            )

    def _check_capacity(self, spec: JobSpec) -> None:
        sample = self._sample
        if sample is None:
            return
        ram_capacity = sample.memory_total - self._headroom(sample)
        if spec.ram is not None and spec.ram > ram_capacity:
            raise JobValidationError(
                f"Job ram {format_size(spec.ram)} exceeds the "
                f"{format_size(ram_capacity)} this host can admit"
            )
        if spec.gpus == 0:
            return
        indices = {gpu.index for gpu in sample.gpus}
        unknown = set(spec.gpu) - indices
        if unknown:
            raise JobValidationError(f"No such GPU: {sorted(unknown)}")
        if spec.gpus > len(sample.gpus):
            raise JobValidationError(
                f"Job needs {spec.gpus} GPUs; this host has {len(sample.gpus)}"
            )
        eligible = [gpu for gpu in sample.gpus if not spec.gpu or gpu.index in spec.gpu]
        if spec.vram is not None and sum(g.total >= spec.vram for g in eligible) < spec.gpus:
            raise JobValidationError(
                f"Job vram {format_size(spec.vram)} exceeds every eligible GPU's total memory"
            )


def check_prerequisites(commands: Sequence[str]) -> None:
    """Refuse to start on a host missing a required runtime command."""
    if sys.platform != "linux":
        raise QueueError("The queue service requires Linux")
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise QueueError(f"The queue service requires: {', '.join(missing)}")


def prepare_private_directory(path: Path) -> None:
    """Create a missing directory owner-only, leaving an existing one's mode alone."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def _with_margin(peak: int, limit: int) -> int:
    """Add the learned margin without asking for more than the host can ever provide."""
    return max(peak, min(int(peak * _LEARNED_MARGIN), limit))


def _gpu_states(sample: HostSample) -> list[GpuState]:
    return [GpuState(gpu.index, gpu.total, gpu.used) for gpu in sample.gpus]
