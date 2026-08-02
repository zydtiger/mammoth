"""Explicitly viewer-host telemetry values for optional monitor display.

Execution-host resource provenance must arrive through execution events. These
local samples are always labelled as the machine running the viewer.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class ViewerTelemetry:
    """One local viewer-host sample with no execution-host implication."""

    host_role: str
    hostname: str
    sampled_at: str
    load_average_1m: float | None


def sample_viewer_telemetry() -> ViewerTelemetry:
    """Sample standard-library telemetry and label its source explicitly."""
    try:
        load_average = os.getloadavg()[0]
    except (AttributeError, OSError):
        load_average = None
    return ViewerTelemetry(
        host_role="viewer",
        hostname=socket.gethostname(),
        sampled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        load_average_1m=load_average,
    )
