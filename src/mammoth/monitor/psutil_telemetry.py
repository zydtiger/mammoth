"""Optional psutil-backed viewer-host resource telemetry.

Import this module only with the ``monitor`` extra installed.
"""

from __future__ import annotations

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
    cpu_percent: float
    memory_percent: float


def sample_psutil_viewer_telemetry() -> PsutilViewerTelemetry:
    """Return a local psutil sample explicitly labelled ``viewer``."""
    return PsutilViewerTelemetry(
        host_role="viewer",
        hostname=socket.gethostname(),
        sampled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        memory_percent=float(psutil.virtual_memory().percent),
    )
