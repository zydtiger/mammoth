"""Optional psutil-backed viewer-host resource telemetry.

Import this module only with the ``monitor`` extra installed.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime

import psutil  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class PsutilViewerTelemetry:
    """CPU and memory sampled on the host running the monitor viewer."""

    host_role: str
    hostname: str
    sampled_at: str
    cpu_percent: float | None
    memory_percent: float | None
    load_average_1m: float | None
    cpu_frequency_mhz: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None


def sample_psutil_viewer_telemetry() -> PsutilViewerTelemetry:
    """Return a local psutil sample explicitly labelled ``viewer``."""
    try:
        load_average = os.getloadavg()[0]
    except (AttributeError, OSError):
        load_average = None
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unknown"
    try:
        cpu_percent = float(psutil.cpu_percent(interval=None))
    except (OSError, psutil.Error):
        cpu_percent = None
    try:
        memory = psutil.virtual_memory()
        memory_percent = float(memory.percent)
        memory_used_bytes = int(memory.used)
        memory_total_bytes = int(memory.total)
    except (OSError, psutil.Error):
        memory_percent = None
        memory_used_bytes = None
        memory_total_bytes = None
    try:
        frequency = psutil.cpu_freq()
        cpu_frequency_mhz = float(frequency.current) if frequency is not None else None
    except (OSError, psutil.Error):
        cpu_frequency_mhz = None
    return PsutilViewerTelemetry(
        host_role="viewer",
        hostname=hostname,
        sampled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        load_average_1m=load_average,
        cpu_frequency_mhz=cpu_frequency_mhz,
        memory_used_bytes=memory_used_bytes,
        memory_total_bytes=memory_total_bytes,
    )
