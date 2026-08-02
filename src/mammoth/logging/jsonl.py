"""JSONL observation sink backed by the core append-only event writer.

``RunObserver`` feeds this adapter; monitors consume the resulting schema-v1
stream through ``mammoth.core.events``.
"""

from __future__ import annotations

from mammoth.core.events import ExecutionEventWriter
from mammoth.logging.model import Observation


class JsonlEventSink:
    """Retain lifecycle records and bounded display metrics in JSONL."""

    def __init__(self, writer: ExecutionEventWriter) -> None:
        self.writer = writer

    def observe(self, observation: Observation) -> None:
        """Translate a sink-neutral observation into one core event append."""
        fields = dict(observation.fields)
        if observation.display_metrics:
            fields["display_metrics"] = observation.display_metrics
        self.writer.emit(observation.event, **fields)

    def flush(self) -> None:
        """Flush any replaceable progress record."""
        self.writer.flush_progress(final=False)

    def close(self) -> None:
        """Close the owned producer stream."""
        self.writer.close()
