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
        memory_percent = float(psutil.virtual_memory().percent)
    except (OSError, psutil.Error):
        memory_percent = None
    return PsutilViewerTelemetry(
        host_role="viewer",
        hostname=hostname,
        sampled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        load_average_1m=load_average,
    )
