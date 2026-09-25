"""Process-exclusive plain-text logging for human diagnostics and tracebacks.

Applications attach the returned handler to their own Python logger. Mammoth
monitoring never parses these files as machine state.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, Union, cast

from mammoth.compat import add_exception_note
from mammoth.core._filesystem.text import ProcessTextLogLease as ProcessTextLogLease
from mammoth.core._filesystem.text import claim_process_text_log as claim_process_text_log
from mammoth.core._filesystem.text import open_text_stream
from mammoth.core.execution import ExecutionContext

DEFAULT_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


if TYPE_CHECKING:
    _TextStreamHandler = logging.StreamHandler[TextIO]
else:
    _TextStreamHandler = logging.StreamHandler


class ProcessTextLogHandler(_TextStreamHandler):
    """A plain UTF-8 handler that owns and closes one rank log descriptor."""

    def __init__(self, path: Path, *, level: int = logging.INFO) -> None:
        self._mammoth_closed = False
        self.path = Path(path)
        self._lease = claim_process_text_log(self.path)
        descriptor: Union[int, None] = None
        stream: Union[TextIO, None] = None
        try:
            descriptor = open_text_stream(self._lease)
            descriptor_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_dev != self._lease._device
                or descriptor_stat.st_ino != self._lease._inode
            ):
                raise OSError(f"Text log changed while ownership was established: {self.path}")
            stream = cast(TextIO, os.fdopen(descriptor, "a", encoding="utf-8", buffering=1))
            descriptor = None
            super().__init__(stream)
            self.setLevel(level)
            self.setFormatter(logging.Formatter(DEFAULT_TEXT_FORMAT))
        except BaseException as error:
            self._mammoth_closed = True
            try:
                if stream is not None:
                    stream.close()
                elif descriptor is not None:
                    os.close(descriptor)
            except BaseException as cleanup_error:
                add_exception_note(
                    error,
                    "Text log descriptor cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            try:
                self._lease.close()
            except BaseException as cleanup_error:
                add_exception_note(
                    error,
                    "Text log lease cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            raise

    def close(self) -> None:
        """Flush and close the file stream owned by this handler."""
        if self._mammoth_closed:
            return
        self._mammoth_closed = True
        try:
            if not self.stream.closed:
                self.flush()
                self.stream.close()
        finally:
            try:
                self._lease.close()
            finally:
                super().close()


def create_process_text_handler(
    context: ExecutionContext,
    *,
    rank: int,
    world_size: Union[int, None] = None,
    level: int = logging.INFO,
) -> ProcessTextLogHandler:
    """Create the exclusive plain-text handler for one execution process."""
    return ProcessTextLogHandler(
        context.rank_log_path(rank, world_size=world_size),
        level=level,
    )
