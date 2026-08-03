"""Compose one process's exclusive text log and append-only observation sinks.

Distributed and single-process runtimes create this framework-neutral bundle
after joining an execution context. Trainers and project loops consume its
observer, while applications decide where to attach the returned text handler.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from mammoth.core.events import ExecutionEventWriter
from mammoth.core.execution import ExecutionContext
from mammoth.logging.jsonl import JsonlEventSink
from mammoth.logging.observer import ObservationSink, RunObserver
from mammoth.logging.text import ProcessTextLogHandler, create_process_text_handler


@dataclass
class ExecutionLogging:
    """Own one rank's observer sinks and process-exclusive diagnostic handler."""

    context: ExecutionContext
    rank: int
    observer: RunObserver
    text_handler: ProcessTextLogHandler
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
    additional_sinks: Sequence[ObservationSink] = (),
    text_level: int = logging.INFO,
) -> ExecutionLogging:
    """Create JSONL-first observation fan-out plus one exclusive text handler."""
    writer = ExecutionEventWriter.for_process(context, rank=rank)
    observer = RunObserver((JsonlEventSink(writer), *additional_sinks))
    try:
        text_handler = create_process_text_handler(context, rank=rank, level=text_level)
    except BaseException:
        observer.close()
        raise
    return ExecutionLogging(
        context=context,
        rank=rank,
        observer=observer,
        text_handler=text_handler,
    )
