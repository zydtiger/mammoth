"""Host sampling without job cooperation: GPU memory, host memory, and job cgroups.

GPU compute processes map to jobs through ``/proc/<pid>/cgroup``. A process that
maps to no job counts as an external tenant, which is the safe direction.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union

from mammoth.compat import DATACLASS_SLOTS
from mammoth.queue.model import QueueError

_BYTES_PER_MIB = 1024**2
_COMMAND_TIMEOUT_SECONDS = 30.0
_JOB_UNIT = re.compile(r"actl-tx-([0-9a-f]{32})\.service")
_CGROUP_ROOT = Path("/sys/fs/cgroup")


class SamplingError(QueueError):
    """Raised when host resources cannot be read; new launches must stop."""


@dataclass(frozen=True, **DATACLASS_SLOTS)
class GpuSample:
    """One GPU's memory in bytes, with per-execution use keyed by ``tx:`` ID."""

    index: int
    uuid: str
    total: int
    used: int
    execution_vram: Mapping[str, int]


@dataclass(frozen=True, **DATACLASS_SLOTS)
class HostSample:
    """One consistent read of GPUs and host memory, in bytes."""

    gpus: tuple[GpuSample, ...]
    memory_total: int
    memory_available: int


@dataclass(frozen=True, **DATACLASS_SLOTS)
class ExecutionMemory:
    """One execution's cgroup memory facts."""

    anon: int
    oom_kills: int


class Sampler(Protocol):
    """Read host resources for the scheduler."""

    def sample(self) -> HostSample:
        """Return current host resources, or raise :class:`SamplingError`."""
        ...

    def execution_memory(self, tx_id: str) -> Union[ExecutionMemory, None]:
        """Return one execution's memory facts, or ``None`` once its cgroup is gone."""
        ...


class HostSampler:
    """Sample this host through ``nvidia-smi``, ``/proc``, and systemd cgroups."""

    def __init__(self, nvidia_smi: str = "nvidia-smi", systemctl: str = "systemctl") -> None:
        self._nvidia_smi = nvidia_smi
        self._systemctl = systemctl

    def sample(self) -> HostSample:
        """Return current host resources, or raise :class:`SamplingError`."""
        gpu_rows = self._query("--query-gpu=index,uuid,memory.total,memory.used")
        process_rows = self._query("--query-compute-apps=gpu_uuid,pid,used_memory")
        vram_by_gpu: dict[str, dict[str, int]] = {}
        for gpu_uuid, pid_text, used_text in process_rows:
            tx_id = execution_of_pid(pid_text)
            if tx_id is None or not used_text.isdigit():
                continue
            per_gpu = vram_by_gpu.setdefault(gpu_uuid, {})
            per_gpu[tx_id] = per_gpu.get(tx_id, 0) + int(used_text) * _BYTES_PER_MIB
        try:
            gpus = tuple(
                GpuSample(
                    index=int(index),
                    uuid=gpu_uuid,
                    total=int(total) * _BYTES_PER_MIB,
                    used=int(used) * _BYTES_PER_MIB,
                    execution_vram=vram_by_gpu.get(gpu_uuid, {}),
                )
                for index, gpu_uuid, total, used in gpu_rows
            )
        except ValueError as error:
            raise SamplingError(f"Unexpected nvidia-smi output: {error}") from error
        memory_total, memory_available = read_meminfo()
        return HostSample(gpus, memory_total, memory_available)

    def execution_memory(self, tx_id: str) -> Union[ExecutionMemory, None]:
        """Return one execution's memory facts, or ``None`` once its cgroup is gone."""
        unit = f"actl-tx-{tx_id[3:]}.service"
        try:
            result = subprocess.run(
                [self._systemctl, "--user", "show", "-P", "ControlGroup", unit],
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        group = result.stdout.strip()
        if result.returncode != 0 or not group:
            return None
        return read_cgroup_memory(_CGROUP_ROOT / group.lstrip("/"))

    def _query(self, query: str) -> list[list[str]]:
        try:
            result = subprocess.run(
                [self._nvidia_smi, query, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SamplingError(f"nvidia-smi failed: {error}") from error
        if result.returncode != 0:
            raise SamplingError(f"nvidia-smi failed: {result.stderr.strip() or result.stdout}")
        return [
            [column.strip() for column in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ]


def execution_of_pid(pid: str) -> Union[str, None]:
    """Return the ``tx:`` ID whose unit owns one process, if any."""
    try:
        with open(os.path.join("/proc", pid, "cgroup"), encoding="utf-8") as stream:
            match = _JOB_UNIT.search(stream.read())
    except OSError:
        return None
    return f"tx:{match.group(1)}" if match else None


def read_meminfo(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    """Return ``(MemTotal, MemAvailable)`` in bytes."""
    fields = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if name in ("MemTotal", "MemAvailable") and parts:
                fields[name] = int(parts[0]) * 1024
        return fields["MemTotal"], fields["MemAvailable"]
    except (OSError, KeyError, ValueError) as error:
        raise SamplingError(f"Could not read {path}: {error!r}") from error


def read_cgroup_memory(group: Path) -> Union[ExecutionMemory, None]:
    """Read anonymous memory and the OOM-kill count from one cgroup directory."""
    try:
        anon = _keyed_value(group / "memory.stat", "anon")
        oom_kills = _keyed_value(group / "memory.events", "oom_kill")
    except (OSError, ValueError):
        return None
    return ExecutionMemory(anon, oom_kills)


def _keyed_value(path: Path, key: str) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            return int(value)
    raise ValueError(f"{key} is missing from {path}")
