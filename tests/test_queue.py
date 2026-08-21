"""Exercise the device-aware job queue: spool, lease, runner, and CLI coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.core import validate_device_spec
from mammoth.queue import (
    DeviceLeaseConflictError,
    JobAlreadyClaimedError,
    JobNotFoundError,
    cancel_job,
    claim_device_lease,
    list_jobs,
    reconcile_interrupted_jobs,
    run_serve_loop,
    serve_once,
    submit_job,
)
from mammoth.queue import serve as serve_module
from mammoth.queue.spool import Job, JobOutcome, append_job_outcome
from mammoth.workflow.launch import ProcessResult

runner = CliRunner()


class RecordingLauncher:
    """A stub :class:`~mammoth.workflow.launch.Launcher` returning preset results."""

    def __init__(self, results: Sequence[ProcessResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> ProcessResult:
        self.calls.append(command)
        if not self._results:
            raise AssertionError("RecordingLauncher called more times than results were queued")
        return self._results.pop(0)


def submit(
    entry: Path,
    *,
    device: str = "cuda:0",
    argv: Sequence[str] = ("python", "train.py"),
    run_names: Sequence[str] = ("run-a",),
) -> Job:
    return submit_job(entry, argv=argv, expected_run_names=run_names, device=device)


# --- device-spec validation -------------------------------------------------


def test_validate_device_spec_accepts_indices_and_uuids() -> None:
    assert validate_device_spec("0") == "0"
    assert validate_device_spec("cuda:0") == "cuda:0"
    assert validate_device_spec("GPU-12345678-abcd-1234-abcd-1234567890ab") == (
        "GPU-12345678-abcd-1234-abcd-1234567890ab"
    )


@pytest.mark.parametrize("device", ["", "a/b", "..", ".", "a b", "a" * 129])
def test_validate_device_spec_rejects_unsafe_values(device: str) -> None:
    with pytest.raises(ValueError, match="Device specs"):
        validate_device_spec(device)


# --- submission --------------------------------------------------------------


def test_submit_job_durably_enqueues_with_expected_fields(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0", argv=("python", "train.py", "--epochs", "5"))

    assert job.sequence == 0
    assert job.device == "cuda:0"
    assert job.argv == ("python", "train.py", "--epochs", "5")
    assert job.expected_run_names == ("run-a",)
    snapshot = list_jobs(tmp_path)
    assert [pending.job_id for pending in snapshot.pending] == [job.job_id]


def test_submit_job_assigns_strictly_increasing_sequence(tmp_path: Path) -> None:
    first = submit(tmp_path)
    second = submit(tmp_path)
    third = submit(tmp_path)

    assert (first.sequence, second.sequence, third.sequence) == (0, 1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"argv": ()},
        {"run_names": ()},
        {"device": "bad/device"},
    ],
)
def test_submit_job_rejects_invalid_inputs(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        submit(tmp_path, **kwargs)  # type: ignore[arg-type]


# --- device lease exclusivity and lane concurrency ---------------------------


def test_device_lease_rejects_a_second_lane_on_the_same_device(tmp_path: Path) -> None:
    with claim_device_lease(tmp_path, "cuda:0"), pytest.raises(DeviceLeaseConflictError):
        claim_device_lease(tmp_path, "cuda:0")
    with claim_device_lease(tmp_path, "cuda:0"):
        pass  # released after the first lease closes


def test_device_lease_allows_disjoint_devices_concurrently(tmp_path: Path) -> None:
    with (
        claim_device_lease(tmp_path, "cuda:0") as first,
        claim_device_lease(tmp_path, "cuda:1") as second,
    ):
        assert first.device == "cuda:0"
        assert second.device == "cuda:1"


# --- FIFO ordering, device matching, and single-claim contention -------------


def test_claim_next_job_is_fifo_within_a_matching_device(tmp_path: Path) -> None:
    job_a1 = submit(tmp_path, device="cuda:0", run_names=("run-a1",))
    submit(tmp_path, device="cuda:1", run_names=("run-b",))
    job_a2 = submit(tmp_path, device="cuda:0", run_names=("run-a2",))

    first_claim = serve_module._claim_next_job(tmp_path, "cuda:0")
    second_claim = serve_module._claim_next_job(tmp_path, "cuda:0")
    third_claim = serve_module._claim_next_job(tmp_path, "cuda:0")

    assert first_claim is not None and first_claim.job_id == job_a1.job_id
    assert second_claim is not None and second_claim.job_id == job_a2.job_id
    assert third_claim is None  # only the cuda:1 job remains, which doesn't match


def test_claim_next_job_gives_exactly_one_lane_the_job_under_contention(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(serve_module._claim_next_job, tmp_path, "cuda:0")
        second = pool.submit(serve_module._claim_next_job, tmp_path, "cuda:0")
        results = [first.result(), second.result()]

    winners = [result for result in results if result is not None]
    assert len(winners) == 1


# --- serve_once outcomes ------------------------------------------------------


def test_serve_once_records_completed_outcome(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")
    launcher = RecordingLauncher([ProcessResult(return_code=0, duration_seconds=0.1)])

    outcome = serve_once(tmp_path, "cuda:0", launcher=launcher)

    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.return_code == 0
    snapshot = list_jobs(tmp_path)
    assert snapshot.pending == ()
    assert snapshot.claimed == ()
    assert [completed.job_id for completed in snapshot.completed] == [outcome.job_id]


def test_serve_once_records_failed_outcome_on_nonzero_return_code(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")
    launcher = RecordingLauncher([ProcessResult(return_code=1, duration_seconds=0.1)])

    outcome = serve_once(tmp_path, "cuda:0", launcher=launcher)

    assert outcome is not None
    assert outcome.status == "failed"
    snapshot = list_jobs(tmp_path)
    assert [failed.job_id for failed in snapshot.failed] == [outcome.job_id]


def test_serve_once_returns_none_when_no_job_matches_device(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:1")
    launcher = RecordingLauncher([])

    outcome = serve_once(tmp_path, "cuda:0", launcher=launcher)

    assert outcome is None
    assert launcher.calls == []


def test_serve_once_does_not_export_device_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAMMOTH_RUN_NAME", "leaked")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    submit(tmp_path, device="cuda:0")
    captured: dict[str, Mapping[str, str]] = {}

    def launcher(
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> ProcessResult:
        captured["environment"] = environment
        return ProcessResult(return_code=0, duration_seconds=0.0)

    serve_once(tmp_path, "cuda:0", launcher=launcher)

    assert "CUDA_VISIBLE_DEVICES" not in captured["environment"]
    assert "MAMMOTH_RUN_NAME" not in captured["environment"]


# --- interruption classification: before, during, and after ------------------


def test_reconcile_before_any_claim_finds_nothing(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")

    outcomes = reconcile_interrupted_jobs(tmp_path, "cuda:0")

    assert outcomes == ()
    assert list_jobs(tmp_path).pending  # job is untouched, still pending


def test_reconcile_after_a_crash_mid_job_marks_interrupted(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0")
    claimed = serve_module._claim_next_job(tmp_path, "cuda:0")
    assert claimed is not None and claimed.job_id == job.job_id
    # Simulate a runner that died after claiming but before recording an outcome.

    outcomes = reconcile_interrupted_jobs(tmp_path, "cuda:0")

    assert len(outcomes) == 1
    assert outcomes[0].job_id == job.job_id
    assert outcomes[0].status == "interrupted"
    snapshot = list_jobs(tmp_path)
    assert snapshot.claimed == ()
    assert [entry.job_id for entry in snapshot.interrupted] == [job.job_id]


def test_reconcile_after_a_recorded_completion_only_cleans_up_debris(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0")
    claimed = serve_module._claim_next_job(tmp_path, "cuda:0")
    assert claimed is not None
    # Simulate a runner that died *after* durably recording success but before
    # removing the claimed spool file.
    append_job_outcome(
        tmp_path,
        JobOutcome(
            schema_version=1,
            job_id=job.job_id,
            sequence=job.sequence,
            device="cuda:0",
            status="completed",
            time="2026-01-01T00:00:00Z",
            expected_run_names=job.expected_run_names,
            return_code=0,
        ),
    )

    outcomes = reconcile_interrupted_jobs(tmp_path, "cuda:0")

    assert outcomes == ()  # no duplicate interrupted record
    snapshot = list_jobs(tmp_path)
    assert snapshot.claimed == ()
    assert len(snapshot.completed) == 1
    assert snapshot.interrupted == ()


def test_reconcile_after_full_success_leaves_no_debris(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")
    launcher = RecordingLauncher([ProcessResult(return_code=0, duration_seconds=0.0)])
    serve_once(tmp_path, "cuda:0", launcher=launcher)

    outcomes = reconcile_interrupted_jobs(tmp_path, "cuda:0")

    assert outcomes == ()


# --- run_serve_loop ------------------------------------------------------------


def test_run_serve_loop_processes_fifo_matching_jobs_then_stops_when_idle(tmp_path: Path) -> None:
    job_a1 = submit(tmp_path, device="cuda:0", run_names=("run-a1",))
    job_b = submit(tmp_path, device="cuda:1", run_names=("run-b",))
    job_a2 = submit(tmp_path, device="cuda:0", run_names=("run-a2",))
    launcher = RecordingLauncher(
        [
            ProcessResult(return_code=0, duration_seconds=0.0),
            ProcessResult(return_code=0, duration_seconds=0.0),
        ]
    )

    outcomes = run_serve_loop(tmp_path, "cuda:0", launcher=launcher, stop_when_idle=True)

    assert [outcome.job_id for outcome in outcomes] == [job_a1.job_id, job_a2.job_id]
    remaining_pending = list_jobs(tmp_path).pending
    assert [job.job_id for job in remaining_pending] == [job_b.job_id]


def test_run_serve_loop_reconciles_a_crashed_prior_holder_before_dispatch(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0")
    serve_module._claim_next_job(tmp_path, "cuda:0")  # simulate a dead prior lane

    outcomes = run_serve_loop(
        tmp_path,
        "cuda:0",
        launcher=RecordingLauncher([]),
        stop_when_idle=True,
    )

    assert outcomes == ()  # the reconciled job is not re-run
    snapshot = list_jobs(tmp_path)
    assert [entry.job_id for entry in snapshot.interrupted] == [job.job_id]


def test_run_serve_loop_stops_after_a_live_signal_interruption(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")
    submit(tmp_path, device="cuda:0")
    launcher = RecordingLauncher(
        [ProcessResult(return_code=-2, duration_seconds=0.1, interrupted=True, signal=2)]
    )

    outcomes = run_serve_loop(tmp_path, "cuda:0", launcher=launcher, stop_when_idle=True)

    assert len(outcomes) == 1
    assert outcomes[0].status == "interrupted"
    assert len(launcher.calls) == 1  # the second pending job was never launched
    assert list_jobs(tmp_path).pending  # second job remains queued for a future runner


def test_run_serve_loop_swallows_keyboard_interrupt_while_idle(tmp_path: Path) -> None:
    def raising_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    outcomes = run_serve_loop(
        tmp_path,
        "cuda:0",
        launcher=RecordingLauncher([]),
        sleep=raising_sleep,
    )

    assert outcomes == ()


# --- cancellation --------------------------------------------------------------


def test_cancel_job_removes_an_unclaimed_job(tmp_path: Path) -> None:
    job = submit(tmp_path)

    cancelled = cancel_job(tmp_path, job.job_id)

    assert cancelled.job_id == job.job_id
    assert list_jobs(tmp_path).pending == ()


def test_cancel_job_refuses_a_claimed_job(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0")
    serve_module._claim_next_job(tmp_path, "cuda:0")

    with pytest.raises(JobAlreadyClaimedError):
        cancel_job(tmp_path, job.job_id)


def test_cancel_job_refuses_a_terminal_job(tmp_path: Path) -> None:
    submit(tmp_path, device="cuda:0")
    launcher = RecordingLauncher([ProcessResult(return_code=0, duration_seconds=0.0)])
    outcome = serve_once(tmp_path, "cuda:0", launcher=launcher)
    assert outcome is not None

    with pytest.raises(JobAlreadyClaimedError):
        cancel_job(tmp_path, outcome.job_id)


def test_cancel_job_raises_not_found_for_an_unknown_id(tmp_path: Path) -> None:
    with pytest.raises(JobNotFoundError):
        cancel_job(tmp_path, "does-not-exist")


# --- CLI: submit, list, cancel --------------------------------------------------


def test_cli_queue_submit_list_and_cancel_roundtrip(tmp_path: Path) -> None:
    submit_result = runner.invoke(
        app,
        [
            "queue",
            "submit",
            "--entry",
            str(tmp_path),
            "--device",
            "cuda:0",
            "--run-name",
            "run-a",
            "--",
            "python",
            "train.py",
        ],
    )
    assert submit_result.exit_code == 0, submit_result.output
    assert "submitted" in submit_result.output

    list_result = runner.invoke(app, ["queue", "list", "--entry", str(tmp_path)])
    assert list_result.exit_code == 0
    assert "pending: 1" in list_result.output

    job_id = list_jobs(tmp_path).pending[0].job_id
    cancel_result = runner.invoke(app, ["queue", "cancel", job_id, "--entry", str(tmp_path)])
    assert cancel_result.exit_code == 0
    assert "cancelled" in cancel_result.output
    assert list_jobs(tmp_path).pending == ()


def test_cli_queue_cancel_refuses_a_claimed_job(tmp_path: Path) -> None:
    job = submit(tmp_path, device="cuda:0")
    serve_module._claim_next_job(tmp_path, "cuda:0")

    result = runner.invoke(app, ["queue", "cancel", job.job_id, "--entry", str(tmp_path)])

    assert result.exit_code == 1
    assert "already been claimed" in result.output


def test_cli_queue_cancel_reports_unknown_job_id(tmp_path: Path) -> None:
    result = runner.invoke(app, ["queue", "cancel", "missing", "--entry", str(tmp_path)])

    assert result.exit_code == 2
    assert "No queued job found" in result.output


def test_cli_queue_submit_rejects_invalid_device(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "queue",
            "submit",
            "--entry",
            str(tmp_path),
            "--device",
            "bad/device",
            "--run-name",
            "run-a",
            "--",
            "python",
            "train.py",
        ],
    )

    assert result.exit_code == 2
    assert "Device specs" in result.output


# --- CLI: serve wiring -----------------------------------------------------------


def test_cli_queue_serve_wires_device_entry_and_poll_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = Mock(return_value=())
    monkeypatch.setattr("mammoth.cli.run_serve_loop", recorded)

    result = runner.invoke(
        app,
        [
            "queue",
            "serve",
            "--entry",
            str(tmp_path),
            "--device",
            "cuda:0",
            "--poll-interval",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded.call_count == 1
    args, kwargs = recorded.call_args
    assert args[:2] == (tmp_path, "cuda:0")
    assert kwargs["poll_interval_seconds"] == 0.5


def test_cli_queue_serve_reports_device_lease_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_conflict(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise DeviceLeaseConflictError("Device 'cuda:0' is already served by another lane.")

    monkeypatch.setattr("mammoth.cli.run_serve_loop", raise_conflict)

    result = runner.invoke(
        app,
        ["queue", "serve", "--entry", str(tmp_path), "--device", "cuda:0"],
    )

    assert result.exit_code == 1
    assert "already served" in result.output
