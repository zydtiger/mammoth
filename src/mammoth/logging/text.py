"""Process-exclusive plain-text logging for human diagnostics and tracebacks.

Applications attach the returned handler to their own Python logger. Mammoth
monitoring never parses these files as machine state.
"""

from __future__ import annotations

import fcntl
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from mammoth.core.execution import ExecutionContext

DEFAULT_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@dataclass
class ProcessTextLogLease:
    """Hold exclusive process ownership of one append-only rank log."""

    path: Path
    _descriptor: int
    _device: int
    _inode: int
    _closed: bool = False

    def close(self) -> None:
        """Release ownership while retaining the append-only log inode."""
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> ProcessTextLogLease:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def claim_process_text_log(path: Path) -> ProcessTextLogLease:
    """Claim one regular append-only log without truncating earlier diagnostics."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(log_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = os.open(log_path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise RuntimeError(f"Text log must be a regular file: {log_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Text log is already owned by another process: {log_path}"
            ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return ProcessTextLogLease(
        path=log_path,
        _descriptor=descriptor,
        _device=descriptor_stat.st_dev,
        _inode=descriptor_stat.st_ino,
    )


class ProcessTextLogHandler(logging.StreamHandler[TextIO]):
    """A plain UTF-8 handler that owns and closes one rank log descriptor."""

    def __init__(self, path: Path, *, level: int = logging.INFO) -> None:
        self._mammoth_closed = False
        self.path = Path(path)
        self._lease = claim_process_text_log(self.path)
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        stream: TextIO | None = None
        try:
            descriptor = os.open(self.path, flags)
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
                error.add_note(
                    "Text log descriptor cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            try:
                self._lease.close()
            except BaseException as cleanup_error:
                error.add_note(
                    "Text log lease cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
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
    world_size: int | None = None,
    level: int = logging.INFO,
) -> ProcessTextLogHandler:
    """Create the exclusive plain-text handler for one execution process."""
    return ProcessTextLogHandler(
        context.rank_log_path(rank, world_size=world_size),
        level=level,
    )
