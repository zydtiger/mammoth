"""Exercise the queue service with fake clock, sampler, and executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import pytest

from mammoth.queue.executor import ExecutionRecord, ExecutorError
from mammoth.queue.model import (
    JobSpec,
    JobStateError,
    JobValidationError,
    parse_duration,
    parse_size,
)
from mammoth.queue.sampling import ExecutionMemory, GpuSample, HostSample, SamplingError
from mammoth.queue.service import QueueService, ServiceConfig
from mammoth.queue.store import JobStore

G = 1024**3


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeSampler:
    gpus: list[GpuSample]
    memory_total: int = 200 * G
    memory_available: int = 180 * G
    memory: dict[str, ExecutionMemory] = field(default_factory=dict)
    error: Union[str, None] = None

    def sample(self) -> HostSample:
        if self.error:
            raise SamplingError(self.error)
        return HostSample(tuple(self.gpus), self.memory_total, self.memory_available)

    def execution_memory(self, tx_id: str) -> Union[ExecutionMemory, None]:
        return self.memory.get(tx_id)


@dataclass
class Started:
    name: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    env_files: tuple[str, ...]
    timeout: int
    tx_id: str


class FakeExecutor:
    def __init__(self) -> None:
        self.started: list[Started] = []
        self.records: dict[str, ExecutionRecord] = {}
        self.names: dict[str, str] = {}
        self.cancelled: list[str] = []
        self.collected: list[str] = []
        self.log_text = ""
        self.fail_queries = False
        self.fail_start: Union[str, None] = None
        self.accept_then_fail = False

    def start(self, name, argv, cwd, env, env_files, timeout_seconds) -> str:
        tx_id = f"tx:{len(self.started):032x}"
        if self.fail_start and not self.accept_then_fail:
            raise ExecutorError(self.fail_start)
        self.started.append(
            Started(name, tuple(argv), cwd, dict(env), tuple(env_files), timeout_seconds, tx_id)
        )
        self.names[name] = tx_id
        self.records[tx_id] = ExecutionRecord(tx_id, "running")
        if self.accept_then_fail:
            raise ExecutorError("timed out after acceptance")
        return tx_id

    def _check(self) -> None:
        if self.fail_queries:
            raise ExecutorError("actl is down")

    def status(self, tx_id: str) -> Union[ExecutionRecord, None]:
        self._check()
        return self.records.get(tx_id)

    def find(self, name: str) -> Union[str, None]:
        self._check()
        return self.names.get(name)

    def cancel(self, tx_id: str) -> None:
        self._check()
        self.cancelled.append(tx_id)
        self.finish(tx_id, "cancelled", None)

    def collect(self, tx_id: str) -> None:
        self._check()
        self.collected.append(tx_id)

    def stderr(self, tx_id: str) -> str:
        return self.log_text

    def finish(self, tx_id: str, status: str, exit_code: Union[int, None] = 0) -> None:
        self.records[tx_id] = ExecutionRecord(tx_id, status, exit_code, 1000.0, 2000.0)


def gpu(index: int, total: int = 40, used: int = 0, execution=None) -> GpuSample:
    return GpuSample(index, f"GPU-{index}", total * G, used * G, execution or {})


@dataclass
class Harness:
    service: QueueService
    store: JobStore
    sampler: FakeSampler
    executor: FakeExecutor
    clock: FakeClock
    cwd: str

    def payload(self, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"argv": ["train"], "cwd": self.cwd, "ram": G}
        payload.update(fields)
        return payload

    def submit(self, **fields: Any) -> str:
        return self.service.submit(self.payload(**fields)).job_id

    def restart(self) -> Harness:
        service = QueueService(
            self.store,
            self.sampler,
            self.executor,
            ServiceConfig(ram_stable_seconds=60.0),
            clock=self.clock,
        )
        return Harness(service, self.store, self.sampler, self.executor, self.clock, self.cwd)

    def status(self, job_id: str) -> str:
        return self.service.get(job_id).status


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    store = JobStore(tmp_path / "state.db")
    sampler = FakeSampler([gpu(0), gpu(1, total=20)])
    executor = FakeExecutor()
    clock = FakeClock()
    service = QueueService(
        store, sampler, executor, ServiceConfig(ram_stable_seconds=60.0), clock=clock
    )
    return Harness(service, store, sampler, executor, clock, str(tmp_path))


def test_job_launches_with_scheduler_chosen_gpu_and_explicit_env(harness: Harness) -> None:
    job_id = harness.submit(
        argv=["python", "run.py", "--root", "/anywhere"],
        env={"OMP_NUM_THREADS": "1"},
        timeout_seconds=90,
    )
    harness.service.tick()
    started = harness.executor.started[0]
    assert started.name == f"mammoth-{job_id}"
    assert started.argv == ("python", "run.py", "--root", "/anywhere")
    assert started.env == {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "GPU-0"}
    assert started.timeout == 90
    assert harness.status(job_id) == "running"
    assert harness.service.get(job_id).gpu_indices == (0,)


def test_default_timeout_and_hidden_gpus_for_cpu_jobs(harness: Harness) -> None:
    harness.submit(gpus=0)
    harness.service.tick()
    started = harness.executor.started[0]
    assert started.env == {"CUDA_VISIBLE_DEVICES": ""}
    assert started.timeout == parse_duration("14d")


def test_two_undeclared_jobs_never_share_and_the_second_reports_why(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0)]
    first, second = harness.submit(), harness.submit()
    harness.service.tick()
    harness.service.tick()
    assert harness.status(first) == "running"
    assert harness.status(second) == "pending"
    assert harness.service.wait_reason(second) == "waiting: needs a whole GPU; GPU0 is in use"


def test_terminal_job_releases_its_gpu_and_the_next_one_starts(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0)]
    first, second = harness.submit(), harness.submit()
    harness.service.tick()
    harness.executor.finish(harness.executor.started[0].tx_id, "ok")
    harness.service.tick()
    assert harness.status(first) == "completed"
    assert harness.status(second) == "running"


def test_priority_then_fifo_order(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0)]
    low = harness.submit()
    high = harness.submit(priority=5)
    harness.service.tick()
    assert harness.status(high) == "running"
    assert harness.status(low) == "pending"


def test_set_priority_applies_only_to_pending_jobs(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0)]
    first, second = harness.submit(), harness.submit()
    assert harness.service.set_priority(second, 9).priority == 9
    harness.service.tick()
    assert harness.status(second) == "running"
    with pytest.raises(JobStateError):
        harness.service.set_priority(second, 1)
    assert harness.status(first) == "pending"


def test_after_ok_and_any_dependencies(harness: Harness) -> None:
    parent = harness.submit(gpus=0)
    on_ok = harness.submit(gpus=0, after=[{"job_id": parent, "mode": "ok"}])
    on_any = harness.submit(gpus=0, after=[{"job_id": parent, "mode": "any"}])
    grandchild = harness.submit(gpus=0, after=[{"job_id": on_ok}])
    harness.service.tick()
    assert harness.status(on_ok) == "pending"
    assert harness.service.wait_reason(on_ok) == f"waiting: after {parent}"
    harness.executor.finish(harness.executor.started[0].tx_id, "failed", 1)
    harness.service.tick()
    assert harness.status(on_ok) == "dependency_failed"
    assert harness.status(grandchild) == "dependency_failed"
    assert harness.status(on_any) == "running"
    assert harness.service.get(grandchild).message.endswith("ended as dependency_failed")


def test_unknown_dependency_and_like_are_rejected(harness: Harness) -> None:
    with pytest.raises(JobValidationError, match="Unknown dependency"):
        harness.submit(after=[{"job_id": "0" * 12}])
    with pytest.raises(JobValidationError, match="No job or run"):
        harness.submit(like="nothing")


@pytest.mark.parametrize(
    "fields",
    [
        {"env": {"CUDA_VISIBLE_DEVICES": "1"}},
        {"cwd": "relative"},
        {"cwd": "/does/not/exist"},
        {"argv": []},
        {"vram": 0},
        {"gpus": 0, "vram": G},
        {"gpu": [0, 0]},
        {"gpus": 2, "gpu": [0]},
        {"env_files": ["/does/not/exist"]},
        {"group": "bad group"},
    ],
)
def test_invalid_jobs_are_rejected(harness: Harness, fields: dict[str, Any]) -> None:
    with pytest.raises(JobValidationError):
        harness.submit(**fields)


@pytest.mark.parametrize(
    "fields",
    [{"argv": ["a\0b"]}, {"env": {"A": "b\0c"}}],
)
def test_nul_is_rejected_before_launch(harness: Harness, fields: dict[str, Any]) -> None:
    with pytest.raises(JobValidationError, match="NUL"):
        harness.submit(**fields)


def test_unexpected_start_error_fails_the_job_and_keeps_the_tick_going(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0), gpu(1)]
    bad, good = harness.submit(), harness.submit()
    start = harness.executor.start

    def fail_first(name, *rest):
        if name == f"mammoth-{bad}":
            raise ValueError("cannot exec")
        return start(name, *rest)

    harness.executor.start = fail_first  # type: ignore[method-assign]
    harness.service.tick()
    assert harness.status(good) == "running"
    harness.service.tick()
    assert harness.status(bad) == "failed"


def test_cancel_of_a_stranded_launch_still_waits_out_the_orphan_grace(harness: Harness) -> None:
    job_id = harness.submit()
    _strand_launching(harness, job_id, harness.clock.now)
    restarted = harness.restart()
    restarted.service.cancel(job_id)
    restarted.service.tick()
    assert restarted.status(job_id) == "launching"
    harness.clock.now += 300.0
    restarted.service.tick()
    assert restarted.status(job_id) == "cancelled"


def test_declared_ram_above_the_host_is_rejected(harness: Harness) -> None:
    harness.service.tick()
    with pytest.raises(JobValidationError, match="this host can admit"):
        harness.submit(ram=500 * G)


def test_learned_reservations_stay_within_what_the_host_can_provide(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0, total=24)]
    source = harness.submit(ram=None, run_dir="/runs/full")
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.gpus = [gpu(0, total=24, used=23, execution={tx_id: 23 * G})]
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=175 * G, oom_kills=0)
    harness.clock.now = 1950.0
    harness.service.tick()
    harness.executor.finish(tx_id, "ok")
    harness.service.tick()
    assert harness.status(source) == "completed"
    harness.sampler.gpus = [gpu(0, total=24)]
    harness.sampler.memory_available = 200 * G
    resumed = harness.submit(ram=None, run_dir="/runs/full")
    harness.service.tick()
    job = harness.service.get(resumed)
    assert job.status == "running"
    assert job.reserved_vram == 24 * G
    assert job.reserved_ram == 180 * G


def test_cpu_job_may_set_visible_devices(harness: Harness) -> None:
    harness.submit(gpus=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    harness.service.tick()
    assert harness.executor.started[0].env["CUDA_VISIBLE_DEVICES"] == "0"


def test_capacity_is_checked_against_the_latest_sample(harness: Harness) -> None:
    harness.service.tick()
    with pytest.raises(JobValidationError, match="exceeds"):
        harness.submit(vram=50 * G)
    with pytest.raises(JobValidationError, match="No such GPU"):
        harness.submit(gpu=[7])
    with pytest.raises(JobValidationError, match="this host has"):
        harness.submit(gpus=3)


def test_sampling_failure_stops_launches_and_leaves_running_jobs(harness: Harness) -> None:
    running = harness.submit()
    harness.service.tick()
    waiting = harness.submit()
    harness.sampler.error = "nvidia-smi failed"
    harness.service.tick()
    assert harness.status(running) == "running"
    assert harness.status(waiting) == "pending"
    assert "GPU sampling failed" in harness.service.wait_reason(waiting)
    assert not harness.executor.cancelled


def test_measures_peaks_without_job_cooperation_and_warns_on_overage(harness: Harness) -> None:
    job_id = harness.submit(gpu=[0], vram=10 * G, ram=5 * G)
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.gpus = [gpu(0, used=15, execution={tx_id: 15 * G}), gpu(1, total=20)]
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=8 * G, oom_kills=0)
    harness.clock.now = 1950.0  # Sampled shortly before the fake execution's finish time.
    harness.service.tick()
    job = harness.service.get(job_id)
    assert (job.peak_vram, job.peak_ram) == (15 * G, 8 * G)
    assert harness.status(job_id) == "running"
    accounting = harness.service.resources()["gpus"]
    assert accounting[0]["free"] == 25 * G
    harness.executor.finish(tx_id, "ok")
    harness.service.tick()
    assert harness.status(job_id) == "completed"
    assert harness.service.get(job_id).peak_partial is False


def test_run_dir_inherits_a_complete_peak_of_a_completed_job(harness: Harness) -> None:
    run_dir = "/runs/one"
    first = harness.submit(ram=None, gpus=1, gpu=[0], run_dir=run_dir)
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.gpus = [gpu(0, used=10, execution={tx_id: 10 * G}), gpu(1, total=20)]
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=10 * G, oom_kills=0)
    harness.clock.now = 1950.0
    harness.service.tick()
    harness.executor.finish(tx_id, "ok")
    harness.service.tick()
    assert harness.service.get(first).status == "completed"
    assert harness.service.get(first).peak_partial is False
    harness.sampler.gpus = [gpu(0), gpu(1, total=20)]
    second = harness.submit(ram=None, run_dir=run_dir)
    harness.service.tick()
    job = harness.service.get(second)
    assert job.reserved_vram == int(10 * G * 1.1)
    assert job.reserved_ram == int(10 * G * 1.1)


def _finish_measured(harness: Harness, status: str, seconds: float, *, log: str = "") -> str:
    job_id = harness.submit(ram=None, gpu=[0], run_dir="/runs/resumed")
    harness.service.tick()
    tx_id = harness.executor.started[-1].tx_id
    harness.sampler.gpus = [gpu(0, used=10, execution={tx_id: 10 * G}), gpu(1, total=20)]
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=10 * G, oom_kills=0)
    harness.service.tick()
    harness.executor.log_text = log
    harness.executor.records[tx_id] = ExecutionRecord(
        tx_id, status, 1, harness.clock.now, harness.clock.now + seconds
    )
    harness.service.tick()
    harness.sampler.gpus = [gpu(0), gpu(1, total=20)]
    return job_id


def test_run_dir_inherits_after_a_long_interrupted_run(harness: Harness) -> None:
    _finish_measured(harness, "cancelled", 60.0)
    harness.store.update(harness.service.list_jobs()[0].job_id, started_at=0.0, finished_at=3600.0)
    resumed = harness.submit(ram=None, run_dir="/runs/resumed")
    harness.service.tick()
    assert harness.service.get(resumed).reserved_vram == int(10 * G * 1.1)


@pytest.mark.parametrize(
    ("status", "log"),
    [("failed", ""), ("failed", "torch.OutOfMemoryError")],
)
def test_run_dir_ignores_early_failures_and_oom(harness: Harness, status: str, log: str) -> None:
    _finish_measured(harness, status, 60.0, log=log)
    resumed = harness.submit(ram=None, run_dir="/runs/resumed")
    harness.service.tick()
    assert harness.service.get(resumed).reserved_vram is None


def test_like_uses_a_named_jobs_peak_but_not_an_oom_one(harness: Harness) -> None:
    source = harness.submit(gpu=[0], run_dir="/runs/source")
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.gpus = [gpu(0, used=4, execution={tx_id: 4 * G}), gpu(1, total=20)]
    harness.service.tick()
    harness.executor.log_text = "torch.OutOfMemoryError: CUDA out of memory"
    harness.executor.finish(tx_id, "failed", 1)
    harness.service.tick()
    assert harness.service.get(source).label == "oom"
    harness.sampler.gpus = [gpu(0), gpu(1, total=20)]
    other = harness.submit(like="source")
    harness.service.tick()
    assert harness.service.get(other).reserved_vram is None


def test_oom_is_labelled_without_retry(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    harness.executor.log_text = "torch.OutOfMemoryError: CUDA out of memory"
    harness.executor.finish(harness.executor.started[0].tx_id, "failed", 1)
    harness.service.tick()
    job = harness.service.get(job_id)
    assert (job.status, job.label, job.exit_code) == ("failed", "oom", 1)
    assert len(harness.executor.started) == 1


def test_plain_failure_has_no_oom_label(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    harness.executor.log_text = "ValueError: nope"
    harness.executor.finish(harness.executor.started[0].tx_id, "failed", 1)
    harness.service.tick()
    assert harness.service.get(job_id).label is None


def test_cgroup_oom_kill_is_labelled(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=G, oom_kills=1)
    harness.service.tick()
    del harness.sampler.memory[tx_id]
    harness.executor.finish(tx_id, "failed", 137)
    harness.service.tick()
    assert harness.service.get(job_id).label == "oom"


def test_cancel_pending_is_immediate_and_running_goes_through_actl(harness: Harness) -> None:
    running = harness.submit()
    harness.service.tick()
    pending = harness.submit()
    assert harness.service.cancel(pending).status == "cancelled"
    assert harness.service.cancel(running).cancel_requested
    harness.service.tick()
    assert harness.executor.cancelled == [harness.executor.started[0].tx_id]
    assert harness.status(running) == "cancelled"
    with pytest.raises(JobStateError):
        harness.service.cancel(running)


def test_direct_actl_cancel_is_reconciled(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    harness.executor.finish(harness.executor.started[0].tx_id, "cancelled", None)
    harness.service.tick()
    assert harness.status(job_id) == "cancelled"


def test_wait_returns_done_once_jobs_end_and_rejects_unknown_targets(harness: Harness) -> None:
    job_id = harness.submit(group="study")
    harness.service.tick()
    done, jobs = harness.service.wait([job_id], None, 0.0)
    assert not done and jobs[0].job_id == job_id
    harness.executor.finish(harness.executor.started[0].tx_id, "failed", 1)
    harness.service.tick()
    assert harness.service.wait([], "study", 0.0)[0] is True


# Recovery table, one test per row.


def test_recovery_running_job_stays_tracked_without_a_duplicate(harness: Harness) -> None:
    job_id = harness.submit(vram=5 * G)
    harness.service.tick()
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "running"
    assert len(restarted.executor.started) == 1
    assert restarted.executor.collected == [restarted.executor.started[0].tx_id]
    gpus = restarted.service.resources()["gpus"]
    assert gpus[0]["free"] == 40 * G - 5 * G or gpus[1]["free"] == 20 * G - 5 * G


def test_recovery_running_job_that_ended_while_down_records_the_outcome(harness: Harness) -> None:
    job_id = harness.submit()
    follower = harness.submit(gpus=0, after=[{"job_id": job_id}])
    harness.service.tick()
    harness.executor.finish(harness.executor.started[0].tx_id, "ok")
    restarted = harness.restart()
    restarted.service.tick()
    job = restarted.service.get(job_id)
    assert (job.status, job.finished_at) == ("completed", 2000.0)
    assert job.peak_partial is True
    assert restarted.status(follower) == "running"


def test_recovery_abandoned_job_is_interrupted_and_never_rerun(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    harness.executor.finish(harness.executor.started[0].tx_id, "abandoned", None)
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "interrupted"
    assert len(restarted.executor.started) == 1


def test_recovery_pruned_record_is_unknown_and_never_rerun(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    del harness.executor.records[harness.executor.started[0].tx_id]
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "unknown"
    assert len(restarted.executor.started) == 1


def _strand_launching(harness: Harness, job_id: str, at: float) -> None:
    harness.store.update(
        job_id, status="launching", gpu_indices=(0,), launching_at=at, reserved_ram=G
    )


def test_recovery_launching_found_by_name_is_adopted(harness: Harness) -> None:
    job_id = harness.submit()
    # Crash after ACTL accepted, before the tx ID was recorded.
    harness.executor.names[f"mammoth-{job_id}"] = "tx:" + "a" * 32
    harness.executor.records["tx:" + "a" * 32] = ExecutionRecord("tx:" + "a" * 32, "running")
    _strand_launching(harness, job_id, harness.clock.now)
    restarted = harness.restart()
    restarted.service.tick()
    job = restarted.service.get(job_id)
    assert (job.status, job.tx_id) == ("running", "tx:" + "a" * 32)
    assert not restarted.executor.started


def test_recovery_launching_absent_within_retention_returns_to_pending(harness: Harness) -> None:
    job_id = harness.submit()
    # Crash after the launching record, before ACTL accepted anything.
    _strand_launching(harness, job_id, harness.clock.now)
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "launching"
    assert not restarted.executor.started
    harness.clock.now += 300.0
    restarted.service.tick()
    assert len(restarted.executor.started) == 1
    assert restarted.status(job_id) == "running"


def test_recovery_launching_absent_past_retention_is_unknown(harness: Harness) -> None:
    job_id = harness.submit()
    _strand_launching(harness, job_id, harness.clock.now - 8 * 86400)
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "unknown"
    assert not restarted.executor.started


def test_recovery_query_failure_launches_nothing_until_it_recovers(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    waiting = harness.submit()
    harness.executor.fail_queries = True
    restarted = harness.restart()
    restarted.service.tick()
    assert restarted.status(job_id) == "running"
    assert restarted.status(waiting) == "pending"
    assert len(restarted.executor.started) == 1
    assert "ACTL query failed" in restarted.service.wait_reason(waiting)
    harness.executor.fail_queries = False
    restarted.service.tick()
    assert restarted.status(waiting) == "running"


def test_launching_row_blocks_new_launches_through_a_failed_lookup(harness: Harness) -> None:
    job_id = harness.submit()
    waiting = harness.submit()
    _strand_launching(harness, job_id, harness.clock.now)
    harness.executor.fail_queries = True
    harness.service.tick()
    assert harness.status(job_id) == "launching"
    assert harness.status(waiting) == "pending"


def test_executor_outage_with_no_live_jobs_leaves_the_job_pending(harness: Harness) -> None:
    job_id = harness.submit()
    harness.executor.fail_queries = True
    harness.service.tick()
    assert harness.status(job_id) == "pending"
    assert "ACTL query failed" in harness.service.wait_reason(job_id)
    harness.executor.fail_queries = False
    harness.service.tick()
    assert harness.status(job_id) == "running"


def test_a_failing_cancel_does_not_stop_other_launches(harness: Harness) -> None:
    harness.sampler.gpus = [gpu(0), gpu(1)]
    stuck = harness.submit()
    harness.service.tick()
    harness.service.cancel(stuck)
    original = harness.executor.cancel

    def refuse(tx_id: str) -> None:
        raise ExecutorError("cannot stop")

    harness.executor.cancel = refuse  # type: ignore[method-assign]
    other = harness.submit()
    harness.service.tick()
    assert harness.status(other) == "running"
    harness.executor.cancel = original  # type: ignore[method-assign]
    harness.service.tick()
    assert harness.status(stuck) == "cancelled"


def test_unreadable_memory_does_not_block_unknown_ram_jobs_forever(harness: Harness) -> None:
    first = harness.submit(ram=None, gpus=0)
    second = harness.submit(ram=None, gpus=0)
    harness.service.tick()
    harness.service.tick()
    assert harness.status(second) == "pending"
    harness.clock.now += 61.0
    harness.service.tick()
    assert harness.status(second) == "running"
    assert harness.service.get(first).peak_partial is True


def test_rejected_submission_fails_the_job_without_a_retry(harness: Harness) -> None:
    job_id = harness.submit()
    harness.executor.fail_start = "command not found"
    harness.service.tick()
    assert harness.status(job_id) == "launching"
    harness.service.tick()
    job = harness.service.get(job_id)
    assert job.status == "failed"
    assert "command not found" in job.message


def test_ambiguous_submission_is_settled_by_name(harness: Harness) -> None:
    job_id = harness.submit()
    harness.executor.fail_start = "x"
    harness.executor.accept_then_fail = True
    harness.service.tick()
    harness.executor.fail_start = None
    harness.executor.accept_then_fail = False
    harness.service.tick()
    assert harness.status(job_id) == "running"
    assert len(harness.executor.started) == 1


def test_stability_window_gates_the_next_unknown_ram_job(harness: Harness) -> None:
    first = harness.submit(ram=None, gpus=0)
    second = harness.submit(ram=None, gpus=0)
    harness.service.tick()
    assert harness.status(first) == "running"
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=10 * G, oom_kills=0)
    harness.service.tick()
    assert harness.status(second) == "pending"
    harness.clock.now += 30.0
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=20 * G, oom_kills=0)
    harness.service.tick()
    harness.clock.now += 40.0
    harness.service.tick()
    assert harness.status(second) == "pending"
    harness.clock.now += 30.0
    harness.service.tick()
    harness.service.tick()
    assert harness.status(second) == "running"


def test_partial_peak_after_a_sampling_gap(harness: Harness) -> None:
    job_id = harness.submit()
    harness.service.tick()
    tx_id = harness.executor.started[0].tx_id
    harness.sampler.memory[tx_id] = ExecutionMemory(anon=G, oom_kills=0)
    harness.service.tick()
    harness.clock.now += 500.0
    harness.service.tick()
    assert harness.service.get(job_id).peak_partial is True


def test_size_and_duration_parsing() -> None:
    assert parse_size("28G") == 28 * G
    assert parse_size("512MiB") == 512 * 1024**2
    assert parse_duration("14d") == 14 * 86400
    with pytest.raises(JobValidationError):
        parse_size("lots")
    with pytest.raises(JobValidationError):
        parse_duration("0s")


def test_spec_round_trips_through_storage(harness: Harness) -> None:
    job_id = harness.submit(
        env={"A": "1"},
        gpu=[1],
        vram=G,
        group="g",
        priority=2,
        timeout_seconds=9,
        run_dir="/runs/x",
    )
    spec = harness.store.get(job_id).spec
    assert spec == JobSpec.from_dict(spec.to_dict())
    assert spec.gpus == 1
