"""Optional viewer-host CPU, RAM, and NVIDIA GPU telemetry.

Import this module only with the ``monitor`` extra installed. Static CPU and
DIMM identity is cached in memory while dynamic utilization is sampled on each
dashboard refresh.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import socket
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Protocol

import psutil  # type: ignore[import-untyped]

_COMMAND_TIMEOUT_SECONDS = 1.0
_SUDO_PROMPT_TIMEOUT_SECONDS = 120.0
_MEMORY_QUERY = ("dmidecode", "--type", "memory")
_CPU_POWER_QUERY = ("sensors", "-j", "zenpower-*")
_GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,power.draw,clocks.current.graphics",
    "--format=csv,noheader,nounits",
)
_UNAVAILABLE_VALUES = frozenset(
    {"", "n/a", "na", "not supported", "unknown", "[n/a]", "[not supported]"}
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


class CommandRunner(Protocol):
    """Callable shape used for bounded, shell-free hardware commands."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run one command with the supplied subprocess policy."""
        ...


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    """One physical NVIDIA GPU's identity and dynamic core state."""

    index: int
    name: str
    utilization_percent: float | None
    core_clock_mhz: float | None
    power_draw_w: float | None = None


@dataclass(frozen=True, slots=True)
class PsutilViewerTelemetry:
    """Resources sampled on the host running the monitor viewer."""

    host_role: str
    hostname: str
    sampled_at: str
    cpu_percent: float | None
    memory_percent: float | None
    load_average_1m: float | None
    cpu_frequency_mhz: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    cpu_model_name: str | None = None
    ram_ddr_generation: str | None = None
    ram_speed: str | None = None
    cpu_power_w: float | None = None
    gpus: tuple[GpuTelemetry, ...] = ()


@dataclass(frozen=True, slots=True)
class _StaticHostTelemetry:
    hostname: str
    cpu_model_name: str | None
    ram_ddr_generation: str | None
    ram_speed: str | None


class PsutilViewerTelemetrySampler:
    """Cache hardware identity and sample dynamic viewer-host utilization."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        allow_sudo_password_prompt: bool = False,
    ) -> None:
        self._command_runner = command_runner or subprocess.run
        self._uses_default_runner = command_runner is None
        self._allow_sudo_password_prompt = allow_sudo_password_prompt
        self._static: _StaticHostTelemetry | None = None

    def sample(self) -> PsutilViewerTelemetry:
        """Return one best-effort sample without persisting host information."""
        if self._static is None:
            self._static = self._sample_static()
        cpu_percent = _sample_cpu_percent()
        memory_percent, memory_used, memory_total = _sample_memory_usage()
        return PsutilViewerTelemetry(
            host_role="viewer",
            hostname=self._static.hostname,
            sampled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            load_average_1m=_sample_load_average(),
            cpu_frequency_mhz=_sample_cpu_frequency(),
            memory_used_bytes=memory_used,
            memory_total_bytes=memory_total,
            cpu_model_name=self._static.cpu_model_name,
            ram_ddr_generation=self._static.ram_ddr_generation,
            ram_speed=self._static.ram_speed,
            cpu_power_w=self._sample_cpu_package_power(),
            gpus=self._sample_gpus(),
        )

    def _sample_static(self) -> _StaticHostTelemetry:
        try:
            hostname = _sanitized_text(socket.gethostname())
        except OSError:
            hostname = "unknown"
        ram_generation, ram_speed = self._sample_memory_hardware()
        return _StaticHostTelemetry(
            hostname=hostname,
            cpu_model_name=_sample_cpu_model_name(),
            ram_ddr_generation=ram_generation,
            ram_speed=ram_speed,
        )

    def _sample_memory_hardware(self) -> tuple[str | None, str | None]:
        """Try cached sudo credentials, then optionally permit a TTY prompt."""
        result = self._run_command(("sudo", "-n", *_MEMORY_QUERY))
        if result is None and self._allow_sudo_password_prompt:
            result = self._run_prompted_memory_command()
        return _parse_memory_hardware(result.stdout) if result is not None else (None, None)

    def _run_prompted_memory_command(
        self,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run sudo with inherited terminal input so it can request a password."""
        if not self._uses_default_runner:
            return self._run_command(
                ("sudo", *_MEMORY_QUERY),
                timeout=_SUDO_PROMPT_TIMEOUT_SECONDS,
            )
        try:
            result = subprocess.run(
                ("sudo", *_MEMORY_QUERY),
                stdout=subprocess.PIPE,
                text=True,
                timeout=_SUDO_PROMPT_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired, UnicodeError):
            return None
        return result if result.returncode == 0 else None

    def _sample_gpus(self) -> tuple[GpuTelemetry, ...]:
        result = self._run_command(_GPU_QUERY)
        return _parse_gpus(result.stdout) if result is not None else ()

    def _sample_cpu_package_power(self) -> float | None:
        """Read package power from the zenpower lm-sensors JSON payload."""
        result = self._run_command(_CPU_POWER_QUERY)
        return _parse_cpu_package_power(result.stdout) if result is not None else None

    def _run_command(
        self,
        command: Sequence[str],
        *,
        timeout: float = _COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str] | None:
        try:
            result = self._command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired, UnicodeError):
            return None
        return result if result.returncode == 0 else None


def sample_psutil_viewer_telemetry() -> PsutilViewerTelemetry:
    """Return one local sample without allowing an interactive sudo prompt."""
    return PsutilViewerTelemetrySampler().sample()


def _sample_load_average() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def _sample_cpu_percent() -> float | None:
    try:
        value = float(psutil.cpu_percent(interval=None))
    except (OSError, psutil.Error, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None


def _sample_cpu_frequency() -> float | None:
    try:
        frequencies = psutil.cpu_freq(percpu=True)
    except (OSError, psutil.Error, NotImplementedError, TypeError, ValueError):
        return None
    current = [
        float(frequency.current)
        for frequency in frequencies or ()
        if isinstance(frequency.current, int | float)
        and not isinstance(frequency.current, bool)
        and math.isfinite(float(frequency.current))
        and float(frequency.current) > 0
    ]
    return fmean(current) if current else None


def _sample_memory_usage() -> tuple[float | None, int | None, int | None]:
    try:
        memory = psutil.virtual_memory()
        percent = float(memory.percent)
        used = int(memory.used)
        total = int(memory.total)
    except (OSError, psutil.Error, AttributeError, TypeError, ValueError):
        return None, None, None
    percent_value: float | None = (
        percent if math.isfinite(percent) and 0.0 <= percent <= 100.0 else None
    )
    if used < 0 or total <= 0 or used > total:
        return percent_value, None, None
    return percent_value, used, total


def _sample_cpu_model_name() -> str | None:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        field, separator, value = line.partition(":")
        if separator and field.strip().lower() == "model name" and value.strip():
            return _sanitized_text(value)
    try:
        fallback = platform.processor().strip()
    except (OSError, RuntimeError):
        return None
    return _sanitized_text(fallback) if fallback else None


def _parse_gpus(output: str) -> tuple[GpuTelemetry, ...]:
    parsed: list[GpuTelemetry] = []
    seen: set[int] = set()
    for row in csv.reader(output.splitlines()):
        if len(row) != 5:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        name = _sanitized_text(row[1]) or "--"
        utilization = _optional_float(row[2], minimum=0.0, maximum=100.0)
        power_draw = _optional_float(row[3], minimum=0.0)
        core_clock = _optional_float(row[4], minimum=0.0)
        parsed.append(
            GpuTelemetry(
                index=index,
                name=name,
                utilization_percent=utilization,
                core_clock_mhz=core_clock,
                power_draw_w=power_draw,
            )
        )
    return tuple(sorted(parsed, key=lambda gpu: gpu.index))


def _parse_cpu_package_power(output: str) -> float | None:
    """Sum zenpower RAPL package readings reported by lm-sensors."""
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    values: list[float] = []
    for chip_name, chip in payload.items():
        if not isinstance(chip_name, str) or not chip_name.startswith("zenpower-"):
            continue
        if not isinstance(chip, dict):
            continue
        package = chip.get("RAPL_P_Package")
        if not isinstance(package, dict):
            continue
        raw_power = package.get("power1_input")
        if isinstance(raw_power, bool) or not isinstance(raw_power, int | float):
            continue
        power = float(raw_power)
        if math.isfinite(power) and power >= 0:
            values.append(power)
    return sum(values) if values else None


def _parse_memory_hardware(output: str) -> tuple[str | None, str | None]:
    """Parse installed DDR generations and configured or rated DIMM speeds."""
    generations: set[str] = set()
    configured_speeds: set[str] = set()
    rated_speeds: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Type:"):
            memory_type = line.partition(":")[2].strip().upper()
            if re.fullmatch(r"(?:LP)?DDR\d+(?:E)?", memory_type):
                generations.add(memory_type)
        elif line.startswith("Configured Memory Speed:"):
            speed = _normalized_memory_speed(line.partition(":")[2])
            if speed is not None:
                configured_speeds.add(speed)
        elif line.startswith("Speed:"):
            speed = _normalized_memory_speed(line.partition(":")[2])
            if speed is not None:
                rated_speeds.add(speed)
    generation = "/".join(sorted(generations)) or None
    speeds = configured_speeds or rated_speeds
    speed = "/".join(sorted(speeds, key=_memory_speed_sort_key)) or None
    return generation, speed


def _normalized_memory_speed(value: str) -> str | None:
    normalized = " ".join(value.strip().split())
    if normalized.lower() in _UNAVAILABLE_VALUES:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(MT/s|MHz)", normalized, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        return None
    rendered = f"{number:,.0f}" if number.is_integer() else f"{number:,.1f}"
    return f"{rendered} {match.group(2)}"


def _memory_speed_sort_key(value: str) -> tuple[float, str]:
    try:
        return float(value.split()[0].replace(",", "")), value
    except ValueError:
        return 0.0, value


def _optional_float(
    value: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    normalized = value.strip().lower()
    if normalized in _UNAVAILABLE_VALUES:
        return None
    try:
        converted = float(normalized)
    except ValueError:
        return None
    if (
        not math.isfinite(converted)
        or converted < minimum
        or (maximum is not None and converted > maximum)
    ):
        return None
    return converted


def _sanitized_text(value: object) -> str:
    return " ".join(_CONTROL_CHARACTERS.sub(" ", str(value)).split())
