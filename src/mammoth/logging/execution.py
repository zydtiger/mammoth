"""Compose one process's exclusive text log and append-only observation sinks.

Distributed and single-process runtimes create this framework-neutral bundle
after joining an execution context. Trainers and project loops consume its
observer, while applications decide where to attach the returned text handler.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from mammoth.core.events import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    ExecutionEventWriter,
)
from mammoth.core.execution import ExecutionContext
from mammoth.logging.jsonl import JsonlEventSink
from mammoth.logging.observer import ObservationSink, RunObserver
from mammoth.logging.text import ProcessTextLogHandler, create_process_text_handler


@dataclass
class ExecutionObservability:
    """Own one rank's JSONL writer and sink-neutral observer lifecycle."""

    context: ExecutionContext
    rank: int
    event_writer: ExecutionEventWriter
    observer: RunObserver
    _closed: bool = False

    def close(self) -> None:
        """Flush and close every structured observation sink once."""
        if self._closed:
            return
        self._closed = True
        self.observer.close()

    def __enter__(self) -> ExecutionObservability:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


@dataclass
class ExecutionLogging:
    """Own one rank's observer sinks and process-exclusive diagnostic handler."""

    context: ExecutionContext
    rank: int
    observer: RunObserver
    text_handler: ProcessTextLogHandler
    event_writer: ExecutionEventWriter | None = None
    _closed: bool = False

    def close(self) -> None:
        """Flush structured observations before releasing the text-log lease."""
        if self._closed:
            return
        self._closed = True
        try:
            self.observer.close()
        finally:
            self.text_handler.close()

    def __enter__(self) -> ExecutionLogging:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def create_execution_logging(
    context: ExecutionContext,
    *,
    rank: int,
    world_size: int | None = None,
    additional_sinks: Sequence[ObservationSink] = (),
    text_level: int = logging.INFO,
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> ExecutionLogging:
    """Create JSONL-first observation fan-out plus one exclusive text handler."""
    observability = create_execution_observability(
        context,
        rank=rank,
        world_size=world_size,
        additional_sinks=additional_sinks,
        progress_interval_seconds=progress_interval_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    try:
        text_handler = create_process_text_handler(
            context,
            rank=rank,
            world_size=world_size,
            level=text_level,
        )
    except BaseException:
        observability.close()
        raise
    return ExecutionLogging(
        context=context,
        rank=rank,
        observer=observability.observer,
        text_handler=text_handler,
        event_writer=observability.event_writer,
    )


def create_execution_observability(
    context: ExecutionContext,
    *,
    rank: int,
    world_size: int | None = None,
    additional_sinks: Sequence[ObservationSink] = (),
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> ExecutionObservability:
    """Create one rank's JSONL-first observer without claiming its text log."""
    writer = ExecutionEventWriter.for_process(
        context,
        rank=rank,
        world_size=world_size,
        progress_interval_seconds=progress_interval_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    observer = RunObserver(
        (JsonlEventSink(writer), *additional_sinks),
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    return ExecutionObservability(
        context=context,
        rank=rank,
        event_writer=writer,
        observer=observer,
    )
