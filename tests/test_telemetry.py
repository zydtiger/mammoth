"""Exercise execution-host telemetry recording, reading, and session ownership."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from mammoth.core import RunLayout, create_execution_context
from mammoth.core.events import ExecutionEventTailReader
from mammoth.core.telemetry import (
    TELEMETRY_STREAM_FILENAME,
    TelemetryRecorder,
    TelemetryTailReader,
    parse_gpu_sample,
    telemetry_stream_path,
    visible_gpu_selector,
)
from mammoth.execution import ExecutionSession, ExecutionSpec
from mammoth.logging import create_execution_logging
from mammoth.monitor import ExecutionMonitor
from mammoth.monitor.model import event_stream_paths

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Fake nvidia-smi is a POSIX script")


@pytest.fixture
def real_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mammoth.execution.TelemetryRecorder", TelemetryRecorder)


def install_fake_nvidia_smi(directory: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    directory.mkdir()
    script = directory / "nvidia-smi"
    script.write_text(f'#!/bin/sh\necho "$@" > {directory}/arguments\n{body}\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return directory


def wait_for(condition: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():  # type: ignore[operator]
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met in time")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", ""),
        ("1", "1"),
        ("0, 2", "0,2"),
        ("1,-1,2", "1"),
        ("-1", ""),
        ("GPU-abc,GPU-def", "GPU-abc,GPU-def"),
    ],
)
def test_visible_gpu_selector_follows_cuda_semantics(
    value: str | None, expected: str | None
) -> None:
    environment = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}

    assert visible_gpu_selector(environment) == expected


def test_parse_gpu_sample_keeps_unavailable_fields_empty() -> None:
    assert parse_gpu_sample("1, 98, [N/A], 2048\n", 5.0) == {
        "kind": "gpu",
        "time": 5.0,
        "gpu": "1",
        "utilization": 98.0,
        "power_watts": None,
        "memory_mib": 2048.0,
    }
    assert parse_gpu_sample("malformed\n", 5.0) is None
    non_finite = parse_gpu_sample("0, nan, inf, -inf\n", 5.0)
    assert non_finite["utilization"] is None
    assert non_finite["power_watts"] is None
    assert non_finite["memory_mib"] is None


def test_recorder_writes_gpu_and_cpu_samples_for_visible_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        'while true; do echo "1, 50, 100.5, 1024" || exit 0; sleep 0.02; done',
    )
    reader = TelemetryTailReader(telemetry_stream_path(tmp_path))
    records: list[dict[str, object]] = []
    recorder = TelemetryRecorder(
        tmp_path, interval_seconds=0.02, environment={"CUDA_VISIBLE_DEVICES": "1"}
    )

    recorder.start()
    try:

        def has_both() -> bool:
            records.extend(reader.poll())
            kinds = {record["kind"] for record in records}
            return {"gpu", "cpu"} <= kinds

        wait_for(has_both)
    finally:
        recorder.stop()

    assert "--id=1" in (fake / "arguments").read_text()
    gpu = next(record for record in records if record["kind"] == "gpu")
    assert gpu["gpu"] == "1"
    assert gpu["utilization"] == 50.0
    assert recorder.path.name == TELEMETRY_STREAM_FILENAME
    # A sealed stream stays readable from the start.
    assert TelemetryTailReader(recorder.path).poll()


def test_recorder_counts_child_cpu_but_not_its_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        'while true; do echo "0, 1, 1, 1" || exit 0; done',
    )
    worker = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    reader = TelemetryTailReader(telemetry_stream_path(tmp_path))
    records: list[dict[str, object]] = []
    recorder = TelemetryRecorder(tmp_path, interval_seconds=0.05, environment={})
    recorder.start()
    try:

        def has_worker_load() -> bool:
            records.extend(reader.poll())
            return any(
                record["kind"] == "cpu" and record["worker_cores"] > 0.2 for record in records
            )

        wait_for(has_worker_load, timeout=10.0)
    finally:
        worker.kill()
        worker.wait()
        recorder.stop()


def test_missing_nvidia_smi_records_unavailable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with TelemetryRecorder(tmp_path, interval_seconds=0.02, environment={}) as recorder:
        reader = TelemetryTailReader(recorder.path)
        records: list[dict[str, object]] = []

        def has_status() -> bool:
            records.extend(reader.poll())
            return any(record["kind"] == "status" for record in records)

        wait_for(has_status)

    status = next(record for record in records if record["kind"] == "status")
    assert status["source"] == "gpu"
    assert "nvidia-smi" in status["detail"]


def test_no_visible_gpu_skips_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake_nvidia_smi(tmp_path / "bin", monkeypatch, "exit 0")

    with TelemetryRecorder(
        tmp_path, interval_seconds=0.02, environment={"CUDA_VISIBLE_DEVICES": ""}
    ) as recorder:
        reader = TelemetryTailReader(recorder.path)
        wait_for(lambda: any(record["kind"] == "status" for record in reader.poll()))

    assert not (fake / "arguments").exists()


def test_sampler_death_mid_run_records_status_and_keeps_cpu_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_nvidia_smi(tmp_path / "bin", monkeypatch, 'echo "0, 10, 10, 10"; exit 3')

    with TelemetryRecorder(tmp_path, interval_seconds=0.02, environment={}) as recorder:
        reader = TelemetryTailReader(recorder.path)
        records: list[dict[str, object]] = []

        def recovered() -> bool:
            records.extend(reader.poll())
            kinds = [record["kind"] for record in records]
            return "status" in kinds and "cpu" in kinds[kinds.index("status") :]

        wait_for(recovered)

    status = next(record for record in records if record["kind"] == "status")
    assert "status 3" in status["detail"]


def test_stop_ends_the_sampler_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        'while true; do echo "0, 1, 1, 1" || exit 0; sleep 0.02; done',
    )
    recorder = TelemetryRecorder(tmp_path, interval_seconds=0.02, environment={})
    recorder.start()
    wait_for(lambda: recorder._process is not None)
    process = recorder._process

    recorder.stop()

    assert process is not None
    assert process.poll() is not None


def test_sampler_child_exits_after_its_parent_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        f'echo $$ > {tmp_path}/pid; while true; do echo "0, 1, 1, 1" || exit 0; sleep 0.02; done',
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from mammoth.core.telemetry import TelemetryRecorder\n"
            "TelemetryRecorder(Path(sys.argv[1]), environment={}).start()\n"
            "time.sleep(60)\n",
            str(tmp_path),
        ]
    )
    try:
        wait_for(lambda: (tmp_path / "pid").is_file() and (tmp_path / "pid").read_text().strip())
        child = int((tmp_path / "pid").read_text())
    finally:
        parent.kill()
        parent.wait()

    def child_gone() -> bool:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return True
        return False

    wait_for(child_gone, timeout=10.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_stop_in_a_forked_child_leaves_the_parent_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        'while true; do echo "0, 1, 1, 1" || exit 0; sleep 0.02; done',
    )
    recorder = TelemetryRecorder(tmp_path, interval_seconds=0.02, environment={})
    recorder.start()
    try:
        wait_for(lambda: recorder._process is not None)
        pid = os.fork()
        if pid == 0:
            recorder.stop()
            os._exit(0)
        os.waitpid(pid, 0)
        assert recorder._process.poll() is None
        assert not recorder._stop.is_set()
    finally:
        recorder.stop()


def test_stop_is_bounded_when_the_sampler_ignores_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        """trap '' TERM; while true; do echo "0, 1, 1, 1" || exit 0; sleep 0.02; done""",
    )
    recorder = TelemetryRecorder(tmp_path, interval_seconds=0.02, environment={})
    recorder.start()
    reader = TelemetryTailReader(recorder.path)
    wait_for(lambda: any(record["kind"] == "gpu" for record in reader.poll()))
    process = recorder._process

    started = time.monotonic()
    recorder.stop()

    assert time.monotonic() - started < 5.0
    assert process.poll() is not None


def test_unwritable_stream_only_disables_telemetry(tmp_path: Path) -> None:
    recorder = TelemetryRecorder(tmp_path / "missing", environment={"CUDA_VISIBLE_DEVICES": ""})

    recorder.start()
    recorder.stop()

    assert not recorder.path.exists()


def test_tail_reader_reports_a_damaged_stream_once(tmp_path: Path) -> None:
    path = telemetry_stream_path(tmp_path)
    path.write_bytes(b"not zstandard")
    reader = TelemetryTailReader(path)

    assert reader.poll() == []
    assert reader.poll() == []
    assert len(reader.warnings) == 1


def test_tail_reader_treats_a_missing_stream_as_empty(tmp_path: Path) -> None:
    reader = TelemetryTailReader(telemetry_stream_path(tmp_path))

    assert reader.poll() == []
    assert reader.warnings == []


def session_spec(tmp_path: Path, *, telemetry: bool = True) -> ExecutionSpec:
    return ExecutionSpec(
        run_dir=tmp_path / "run",
        run_name="run",
        invocation_kind="test",
        intended_phases=("train",),
        telemetry=telemetry,
    )


def test_session_records_telemetry_without_project_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_recorder: None
) -> None:
    install_fake_nvidia_smi(tmp_path / "bin", monkeypatch, 'echo "0, 10, 10, 10"; sleep 5')

    with ExecutionSession.create(session_spec(tmp_path)) as session, session.phase_scope("train"):
        reader = TelemetryTailReader(telemetry_stream_path(session.context.execution_dir))
        wait_for(lambda: any(record["kind"] == "gpu" for record in reader.poll()))

    assert telemetry_stream_path(session.context.execution_dir).is_file()


def test_session_opt_out_records_no_telemetry(tmp_path: Path, real_recorder: None) -> None:
    with (
        ExecutionSession.create(session_spec(tmp_path, telemetry=False)) as session,
        session.phase_scope("train"),
    ):
        pass

    assert not telemetry_stream_path(session.context.execution_dir).exists()


def test_session_without_a_started_phase_records_no_telemetry(
    tmp_path: Path, real_recorder: None
) -> None:
    with ExecutionSession.create(session_spec(tmp_path)) as session:
        pass

    assert not telemetry_stream_path(session.context.execution_dir).exists()


def test_session_on_a_nonzero_rank_records_no_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = mock.Mock()
    monkeypatch.setattr("mammoth.execution.TelemetryRecorder", recorder)
    context = create_execution_context(
        tmp_path / "run",
        run_name="run",
        invocation_kind="test",
        intended_phases=("train",),
        world_size=2,
        execution_mode="distributed",
        command=("train",),
    )
    bundle = create_execution_logging(context, rank=1, world_size=2)

    with ExecutionSession.from_established(context, bundle) as session:
        session.start_phase("train")
        session.complete_phase()

    recorder.assert_not_called()


def test_telemetry_stream_never_refreshes_producer_liveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_recorder: None
) -> None:
    install_fake_nvidia_smi(
        tmp_path / "bin",
        monkeypatch,
        'while true; do echo "0, 1, 1, 1" || exit 0; sleep 0.02; done',
    )
    with ExecutionSession.create(session_spec(tmp_path)) as session, session.phase_scope("train"):
        context = session.context
        session.observer.flush()
        streams = event_stream_paths(context)
        assert telemetry_stream_path(context.execution_dir) not in streams
        reader = TelemetryTailReader(telemetry_stream_path(context.execution_dir))
        wait_for(lambda: any(record["kind"] == "gpu" for record in reader.poll()))
        size_before = sum(len(ExecutionEventTailReader(path).poll()) for path in streams)
        monitor = ExecutionMonitor(RunLayout(tmp_path, "run"))
        before = monitor.poll().producers
        wait_for(lambda: any(record["kind"] == "gpu" for record in reader.poll()))
        assert monitor.poll().producers == before
        assert sum(len(ExecutionEventTailReader(path).poll()) for path in streams) == size_before


@pytest.mark.parametrize("enabled", [True, False])
def test_torch_runtime_session_honors_the_spec_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    torch_module = pytest.importorskip("mammoth.torch")
    recorder = mock.Mock()
    monkeypatch.setattr("mammoth.execution.TelemetryRecorder", recorder)
    config = torch_module.RuntimeConfig(strategy="single", device="cpu")

    with torch_module.initialize_runtime(config) as runtime:
        runtime.create_execution(session_spec(tmp_path, telemetry=enabled))
        runtime.start_execution_logging()
        session = runtime.create_execution_session()
        session.start_phase("train")
        session.complete_phase()
        session.close()

    assert recorder.called is enabled
