"""Process-exclusive plain-text logging for human diagnostics and tracebacks.

Applications attach the returned handler to their own Python logger. Mammoth
monitoring never parses these files as machine state.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import TextIO, cast

from mammoth.core.execution import ExecutionContext

DEFAULT_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ProcessTextLogHandler(logging.StreamHandler[TextIO]):
    """A plain UTF-8 handler that owns and closes one rank log descriptor."""

    def __init__(self, path: Path, *, level: int = logging.INFO) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            os.close(descriptor)
            raise OSError(f"Text log must be a regular file: {self.path}")
        stream = cast(TextIO, os.fdopen(descriptor, "a", encoding="utf-8", buffering=1))
        super().__init__(stream)
        self.setLevel(level)
        self.setFormatter(logging.Formatter(DEFAULT_TEXT_FORMAT))

    def close(self) -> None:
        """Flush and close the file stream owned by this handler."""
        try:
            self.flush()
            self.stream.close()
        finally:
            super().close()


def create_process_text_handler(
    context: ExecutionContext,
    *,
    rank: int,
    level: int = logging.INFO,
) -> ProcessTextLogHandler:
    """Create the exclusive plain-text handler for one execution process."""
    return ProcessTextLogHandler(context.rank_log_path(rank), level=level)
