"""Execution-host CPU and GPU telemetry stored beside one execution's event streams.

:class:`TelemetryRecorder` samples the owning process and its visible NVIDIA GPUs
into ``telemetry.jsonl.zst``. The stream is deliberately not a producer event
stream: a live sampler must not refresh a hung producer's liveness, and frequent
samples must not displace events from bounded fleet tails. Recording never raises
into the workload; failures become ``status`` records and logged warnings.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Union

from mammoth.core._event_stream import CompressedReadError, ZstdEventReader, ZstdEventWriter
from mammoth.core._filesystem.io import open_event_file

TELEMETRY_STREAM_FILENAME = "telemetry.jsonl.zst"
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 1.0
_PROCESS_STOP_SECONDS = 1.0
_GPU_QUERY = "index,utilization.gpu,power.draw,memory.used"

logger = logging.getLogger(__name__)


def telemetry_stream_path(execution_dir: Path) -> Path:
    """Return the telemetry stream path inside one execution directory."""
    return Path(execution_dir) / TELEMETRY_STREAM_FILENAME


def visible_gpu_selector(environment: Mapping[str, str]) -> Union[str, None]:
    """Translate ``CUDA_VISIBLE_DEVICES`` into an ``nvidia-smi -i`` selector.

    Returns ``None`` when the variable is unset (every GPU is visible) and an
    empty string when no GPU is visible. CUDA ignores entries after the first
    invalid negative index, so this does too.
    """
    value = environment.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    visible: list[str] = []
    for token in (item.strip() for item in value.split(",")):
        if not token or token.startswith("-"):
            break
        visible.append(token)
    return ",".join(visible)


def parse_gpu_sample(line: str, observed_at: float) -> Union[dict[str, Any], None]:
    """Parse one ``nvidia-smi`` CSV row, keeping unavailable fields as ``None``."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4 or not parts[0]:
        return None
    return {
        "kind": "gpu",
        "time": observed_at,
        "gpu": parts[0],
        "utilization": _optional_float(parts[1]),
        "power_watts": _optional_float(parts[2]),
        "memory_mib": _optional_float(parts[3]),
    }


def _optional_float(value: str) -> Union[float, None]:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _process_cpu_seconds(pid: int, tick: int) -> float:
    """Return one process's cumulative user and system CPU seconds from ``/proc``."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / tick


def _child_pids(pid: int) -> set[int]:
    """Return direct children of every thread in one process."""
    children: set[int] = set()
    for task in Path(f"/proc/{pid}/task").iterdir():
        with suppress(OSError):
            children.update(int(item) for item in (task / "children").read_text().split())
    return children


class TelemetryRecorder:
    """Sample this process and its visible GPUs until :meth:`stop`."""

    def __init__(
        self,
        execution_dir: Path,
        *,
        interval_seconds: float = DEFAULT_TELEMETRY_INTERVAL_SECONDS,
        environment: Union[Mapping[str, str], None] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = telemetry_stream_path(execution_dir)
        self.interval_seconds = interval_seconds
        self._environment = os.environ if environment is None else environment
        self._clock = clock
        self._pid = os.getpid()
        self._writer: Union[ZstdEventWriter, None] = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._process: Union[subprocess.Popen[str], None] = None
        self._process_lock = threading.Lock()

    def start(self) -> None:
        """Open the stream and begin sampling; failure only disables telemetry."""
        try:
            descriptor = open_event_file(self.path, append=True)
            stream = os.fdopen(descriptor, "r+b", buffering=0)
            try:
                self._writer = ZstdEventWriter(stream, self.path)
            except BaseException:
                stream.close()
                raise
            for name, target in (("gpu", self._sample_gpus), ("cpu", self._sample_cpu)):
                thread = threading.Thread(
                    target=self._run_sampler,
                    args=(name, target),
                    name=f"mammoth-telemetry-{name}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except Exception as error:
            logger.warning("Execution telemetry disabled: %s", error)

    def stop(self) -> None:
        """Stop sampling and seal the stream; a forked child leaves both untouched."""
        if os.getpid() != self._pid:
            return
        self._stop.set()
        self._end_process()
        for thread in self._threads:
            thread.join(timeout=_PROCESS_STOP_SECONDS)
        # The sampler thread may have started its child while stop was running.
        self._end_process()
        with self._write_lock:
            writer = self._writer
            self._writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                logger.warning("Could not seal execution telemetry: %s", error)

    def _end_process(self) -> None:
        """Terminate the sampler child, escalating quickly so close never waits long."""
        with self._process_lock:
            process = self._process
        if process is None:
            return
        with suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=_PROCESS_STOP_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            with suppress(OSError):
                process.kill()
            with suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=_PROCESS_STOP_SECONDS)

    def __enter__(self) -> TelemetryRecorder:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._write_lock:
            if self._writer is None:
                return
            try:
                self._writer.write(line)
            except Exception as error:
                logger.warning("Execution telemetry disabled after write failure: %s", error)
                self._writer = None
                self._stop.set()

    def _status(self, source: str, detail: str) -> None:
        self._write(
            {
                "kind": "status",
                "time": self._clock(),
                "source": source,
                "status": "unavailable",
                "detail": detail,
            }
        )

    def _run_sampler(self, source: str, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception as error:
            logger.warning("Execution %s telemetry stopped: %s", source, error)
            self._status(source, str(error))

    def _sample_gpus(self) -> None:
        selector = visible_gpu_selector(self._environment)
        if selector == "":
            self._status("gpu", "No GPU is visible to this process")
            return
        command = [
            "nvidia-smi",
            f"--query-gpu={_GPU_QUERY}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={max(1, round(self.interval_seconds * 1000))}",
        ]
        if selector is not None:
            command.append(f"--id={selector}")
        try:
            # The child stays in this process group, so supervised cleanup reaches
            # it, and exits through SIGPIPE once its parent is gone.
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as error:
            self._status("gpu", f"Could not start nvidia-smi: {error}")
            return
        with self._process_lock:
            self._process = process
        if self._stop.is_set():
            # Stop may have run before this child existed.
            process.terminate()
            return
        assert process.stdout is not None
        for line in process.stdout:
            if self._stop.is_set():
                return
            sample = parse_gpu_sample(line, self._clock())
            if sample is not None:
                self._write(sample)
        if not self._stop.is_set():
            self._status("gpu", f"nvidia-smi exited with status {process.wait()}")

    def _sample_cpu(self) -> None:
        if not Path("/proc/self/stat").exists() or not hasattr(os, "sysconf"):
            self._status("cpu", "Process CPU sampling requires /proc")
            return
        tick = os.sysconf("SC_CLK_TCK")
        main_seconds: Union[float, None] = None
        worker_seconds: dict[int, float] = {}
        observed_at = self._clock()
        while not self._stop.wait(self.interval_seconds):
            now = self._clock()
            main = _process_cpu_seconds(self._pid, tick)
            with self._process_lock:
                excluded = self._process.pid if self._process is not None else None
            workers: dict[int, float] = {}
            for child in _child_pids(self._pid) - {excluded}:
                with suppress(OSError, IndexError, ValueError):
                    workers[child] = _process_cpu_seconds(child, tick)
            elapsed = now - observed_at
            if main_seconds is not None and elapsed > 0:
                # A forked child's counters start at zero, so a new worker counts in full.
                worker_delta = sum(
                    max(0.0, seconds - worker_seconds.get(pid, 0.0))
                    for pid, seconds in workers.items()
                )
                self._write(
                    {
                        "kind": "cpu",
                        "time": now,
                        "main_cores": round((main - main_seconds) / elapsed, 4),
                        "worker_cores": round(worker_delta / elapsed, 4),
                    }
                )
            main_seconds, worker_seconds, observed_at = main, workers, now


class TelemetryTailReader:
    """Incrementally read telemetry records, tolerating a missing or damaged stream."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._reader = ZstdEventReader(self.path)
        self._buffer = b""
        self._failed = False
        self.warnings: list[str] = []

    def poll(self) -> list[dict[str, Any]]:
        """Return complete records appended since the previous poll."""
        if self._failed or not self.path.is_file():
            return []
        try:
            data = self._reader.poll()
        except CompressedReadError as error:
            self._failed = True
            self.warnings.append(f"Telemetry stream {self.path}: {error}")
            data = error.valid_bytes
        except OSError as error:
            self._failed = True
            self.warnings.append(f"Telemetry stream {self.path}: {error}")
            return []
        lines = (self._buffer + data).split(b"\n")
        self._buffer = lines.pop()
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                self.warnings.append(f"Telemetry stream {self.path}: skipped a malformed record")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
