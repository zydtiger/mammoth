"""Exercise the queue API over a real socket, the CLI client, sampling, and ``actl``."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_queue_service import FakeExecutor, FakeSampler, gpu
from typer.testing import CliRunner

from mammoth.cli import app
from mammoth.queue import (
    ActlExecutor,
    JobNotFoundError,
    JobStateError,
    JobValidationError,
    QueueApiServer,
    QueueClient,
    QueueError,
    QueueService,
    ServiceAlreadyRunningError,
    ServiceUnavailableError,
)
from mammoth.queue.api import _UnixConnection
from mammoth.queue.daemon import hold_state_lock
from mammoth.queue.executor import ExecutorError
from mammoth.queue.sampling import (
    HostSampler,
    SamplingError,
    read_cgroup_memory,
    read_meminfo,
)
from mammoth.queue.store import JobStore

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="queue service requires Linux")

runner = CliRunner()


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[QueueClient, QueueService, FakeExecutor, Path]]:
    store = JobStore(tmp_path / "state.db")
    executor = FakeExecutor()
    service = QueueService(store, FakeSampler([gpu(0)]), executor)
    socket_path = tmp_path / "queue.sock"
    server = QueueApiServer(socket_path, service)
    server.serve_in_thread()
    yield QueueClient(socket_path), service, executor, socket_path
    server.close()
    service.stop()
    store.close()


def test_socket_and_state_are_owner_only(api, tmp_path: Path) -> None:
    _, _, _, socket_path = api
    assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "state.db").stat().st_mode) == 0o600


def test_submit_list_get_cancel_over_the_socket(api, tmp_path: Path) -> None:
    client, _, _, _ = api
    job = client.submit({"argv": ["x"], "cwd": str(tmp_path), "env": {"SECRET": "value"}})
    assert job["status"] == "pending"
    assert job["env_keys"] == ["SECRET"]
    assert "value" not in json.dumps(job)
    assert client.get_job(job["job_id"])["job_id"] == job["job_id"]
    assert [item["job_id"] for item in client.list_jobs()] == [job["job_id"]]
    assert client.set_priority(job["job_id"], 3)["priority"] == 3
    assert client.cancel(job["job_id"])["status"] == "cancelled"
    with pytest.raises(JobStateError):
        client.cancel(job["job_id"])


def test_errors_map_to_their_types(api, tmp_path: Path) -> None:
    client, _, _, _ = api
    with pytest.raises(JobValidationError):
        client.submit({"argv": [], "cwd": str(tmp_path)})
    with pytest.raises(JobNotFoundError):
        client.get_job("0" * 12)
    with pytest.raises(QueueError):
        client._request("GET", "/v1/nowhere")


def test_blocking_wait_returns_when_the_job_ends(api, tmp_path: Path) -> None:
    client, service, executor, _ = api
    job_id = client.submit({"argv": ["x"], "cwd": str(tmp_path), "ram": 1024**3})["job_id"]
    service.tick()
    outcomes: list = []
    thread = threading.Thread(
        target=lambda: outcomes.extend(client.wait([job_id], None, reconnect_seconds=5.0))
    )
    thread.start()
    time.sleep(0.2)
    assert not outcomes
    executor.finish(executor.started[0].tx_id, "failed", 3)
    service.tick()
    thread.join(timeout=10.0)
    assert outcomes[0]["status"] == "failed"
    assert outcomes[0]["exit_code"] == 3


def test_submit_with_the_service_down_fails_and_persists_nothing(tmp_path: Path) -> None:
    client = QueueClient(tmp_path / "absent.sock")
    with pytest.raises(ServiceUnavailableError, match="systemctl --user start"):
        client.submit({"argv": ["x"], "cwd": str(tmp_path)})
    assert list(tmp_path.iterdir()) == []


def test_wait_survives_a_restart_and_fails_explicitly_at_its_bound(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    executor = FakeExecutor()
    service = QueueService(store, FakeSampler([gpu(0)]), executor)
    socket_path = tmp_path / "queue.sock"
    client = QueueClient(socket_path)
    server = QueueApiServer(socket_path, service)
    server.serve_in_thread()
    job_id = client.submit({"argv": ["x"], "cwd": str(tmp_path), "ram": 1024**3})["job_id"]
    service.tick()
    server.close()

    outcomes: list = []
    thread = threading.Thread(
        target=lambda: outcomes.extend(
            client.wait([job_id], None, reconnect_seconds=30.0, retry_seconds=0.05)
        )
    )
    thread.start()
    time.sleep(0.2)
    assert not outcomes
    executor.finish(executor.started[0].tx_id, "ok")
    restarted = QueueService(store, FakeSampler([gpu(0)]), executor)
    restarted.tick()
    server = QueueApiServer(socket_path, restarted)
    server.serve_in_thread()
    thread.join(timeout=10.0)
    assert outcomes[0]["status"] == "completed"
    server.close()
    store.close()

    with pytest.raises(ServiceUnavailableError, match="job state is unknown"):
        client.wait([job_id], None, reconnect_seconds=0.1, retry_seconds=0.02)


def test_second_instance_cannot_take_the_state_lock(tmp_path: Path) -> None:
    descriptor = hold_state_lock(tmp_path)
    script = (
        "import sys; from pathlib import Path; from mammoth.queue.daemon import hold_state_lock\n"
        "hold_state_lock(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert ServiceAlreadyRunningError.__name__ in result.stderr
    os.close(descriptor)


def test_cli_submit_list_cancel(api, tmp_path: Path) -> None:
    _, _, _, socket_path = api
    socket_option = ["--socket", str(socket_path)]
    result = runner.invoke(
        app,
        [
            "queue",
            "submit",
            *socket_option,
            "--cwd",
            str(tmp_path),
            "--env",
            "A=1",
            "--vram",
            "2G",
            "--ram",
            "1G",
            "--group",
            "study",
            "--timeout",
            "1h",
            "--",
            "python",
            "--flag",
        ],
    )
    assert result.exit_code == 0, result.output
    job_id = result.output.strip()
    result = runner.invoke(app, ["queue", "list", *socket_option])
    assert job_id in result.output
    assert "group=study" in result.output
    result = runner.invoke(app, ["queue", "list", "--json", *socket_option])
    assert json.loads(result.output)["jobs"][0]["argv"] == ["python", "--flag"]
    result = runner.invoke(app, ["queue", "priority", job_id, "4", *socket_option])
    assert result.exit_code == 0
    result = runner.invoke(app, ["queue", "cancel", job_id, *socket_option])
    assert "cancelled" in result.output
    result = runner.invoke(app, ["queue", "wait", job_id, *socket_option])
    assert json.loads(result.output)["jobs"][0]["status"] == "cancelled"


def test_cli_rejects_a_submitted_visible_devices_and_reports_a_down_service(tmp_path: Path) -> None:
    missing = ["--socket", str(tmp_path / "none.sock")]
    result = runner.invoke(
        app, ["queue", "submit", *missing, "--env", "CUDA_VISIBLE_DEVICES=1", "--", "x"]
    )
    assert result.exit_code == 2
    result = runner.invoke(app, ["queue", "submit", *missing, "--", "x"])
    assert result.exit_code == 1
    assert "not answering" in result.output


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def test_host_sampler_parses_nvidia_smi_and_treats_unknown_processes_as_external(
    tmp_path: Path,
) -> None:
    fake = _script(
        tmp_path / "smi",
        'case "$1" in\n'
        '--query-gpu*) echo "0, GPU-a, 1000, 300"; echo "1, GPU-b, 500, 0";;\n'
        f'*) echo "GPU-a, {os.getpid()}, 300";;\n'
        "esac",
    )
    sample = HostSampler(nvidia_smi=fake).sample()
    assert [(item.index, item.uuid) for item in sample.gpus] == [(0, "GPU-a"), (1, "GPU-b")]
    assert sample.gpus[0].total == 1000 * 1024**2
    assert sample.gpus[0].used == 300 * 1024**2
    assert sample.gpus[0].execution_vram == {}


def test_host_sampler_surfaces_failure(tmp_path: Path) -> None:
    with pytest.raises(SamplingError):
        HostSampler(nvidia_smi=_script(tmp_path / "smi", "exit 3")).sample()
    with pytest.raises(SamplingError):
        HostSampler(nvidia_smi=str(tmp_path / "absent")).sample()


def test_memory_readers(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable: 40 kB\n", encoding="utf-8")
    assert read_meminfo(meminfo) == (102400, 40960)
    group = tmp_path / "group"
    group.mkdir()
    assert read_cgroup_memory(group) is None
    (group / "memory.stat").write_text("anon 7\nfile 9\n", encoding="utf-8")
    (group / "memory.events").write_text("oom 1\noom_kill 2\n", encoding="utf-8")
    facts = read_cgroup_memory(group)
    assert facts is not None
    assert (facts.anon, facts.oom_kills) == (7, 2)


def test_actl_executor_against_a_fake_command(tmp_path: Path) -> None:
    tx_id = "tx:" + "a" * 32
    other = "tx:" + "b" * 32
    fake = _script(
        tmp_path / "actl",
        f'echo "$@" >> {tmp_path}/calls\n'
        'case "$1" in\n'
        f'submit) echo "submitted {tx_id}: name"; echo "trigger: immediate";;\n'
        f"list) echo \"{tx_id}  mammoth-job  running  cleanup=pending\"; echo 'TASK  SCHEDULE';;\n"
        'status) if [ "$2" = "' + tx_id + '" ]; then echo \'{"status": "ok", "exit_code": 0, '
        '"started_at": "2026-11-01T00:00:00+00:00", "finished_at": "2026-11-01T00:00:01+00:00"}\'; '
        "else exit 1; fi;;\n"
        "logs) echo out-of-memory-free;;\n"
        "esac",
    )
    executor = ActlExecutor(fake)
    started = executor.start("mammoth-job", ["python", "-x"], "/w", {"A": "1"}, ["/env"], 60)
    assert started == tx_id
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert (
        "submit --transient mammoth-job --cwd /w --timeout 60s --env A=1 --env-file /env "
        "-- python -x" in calls
    )
    record = executor.status(tx_id)
    assert record is not None
    assert record.is_terminal
    assert record.finished_at - record.started_at == 1.0
    assert executor.status(other) is None
    assert executor.find("mammoth-job") == tx_id
    assert executor.find("mammoth-none") is None
    executor.cancel(tx_id)
    executor.collect(tx_id)
    assert "out-of-memory-free" in executor.stderr(tx_id)
    assert "logs " + tx_id + " --stderr" in (tmp_path / "calls").read_text(encoding="utf-8")


def test_actl_timeout_error_does_not_carry_env_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mammoth.queue.executor._COMMAND_TIMEOUT_SECONDS", 0.05)
    executor = ActlExecutor(_script(tmp_path / "actl", "sleep 5"))
    with pytest.raises(ExecutorError) as caught:
        executor.start("name", ["x"], "/w", {"SECRET": "hunter2"}, [], 60)
    assert "hunter2" not in str(caught.value)
    assert "TimeoutExpired" in str(caught.value)


@pytest.mark.parametrize("length", ["nope", "-1", "99999999"])
def test_api_refuses_a_bad_content_length(api, length: str) -> None:
    client, _, _, _ = api
    connection = _UnixConnection(client.socket_path, 5.0)
    connection.putrequest("POST", "/v1/jobs")
    connection.putheader("Content-Length", length)
    connection.endheaders()
    assert connection.getresponse().status == 400
    connection.close()


def test_actl_executor_keeps_listed_failures_as_errors(tmp_path: Path) -> None:
    tx_id = "tx:" + "a" * 32
    fake = _script(
        tmp_path / "actl",
        'case "$1" in\n'
        f'list) echo "{tx_id}  mammoth-job  running  cleanup=pending";;\n'
        "*) exit 1;;\n"
        "esac",
    )
    executor = ActlExecutor(fake)
    with pytest.raises(ExecutorError):
        executor.status(tx_id)
    with pytest.raises(ExecutorError):
        ActlExecutor(str(tmp_path / "absent")).find("x")


@pytest.mark.skipif(shutil.which("actl") is None, reason="actl is not installed")
def test_real_actl_runs_a_job_with_scheduler_env_and_finishes(tmp_path: Path) -> None:
    executor = ActlExecutor()
    script = _script(tmp_path / "job.sh", 'printf "%s" "$CUDA_VISIBLE_DEVICES" > seen')
    tx_id = executor.start(
        f"mammoth-test-{os.getpid()}",
        ["./job.sh"],
        str(tmp_path),
        {"CUDA_VISIBLE_DEVICES": "GPU-test"},
        [],
        60,
    )
    assert script
    deadline = time.monotonic() + 60.0
    record = executor.status(tx_id)
    while (record is None or not record.is_terminal) and time.monotonic() < deadline:
        time.sleep(0.2)
        record = executor.status(tx_id)
    assert record is not None
    assert (record.status, record.exit_code) == ("ok", 0)
    assert (tmp_path / "seen").read_text(encoding="utf-8") == "GPU-test"
    executor.collect(tx_id)
