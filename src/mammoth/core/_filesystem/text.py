"""Exclusive append-only text descriptor ownership across local platforms."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from mammoth.core._filesystem import windows as _windows
from mammoth.core._filesystem.locking import lock_exclusive, unlock


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
            if os.name != "nt":
                unlock(self._descriptor)
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
    if os.name == "nt":
        try:
            descriptor = _windows.open_file(log_path, write=True, create=True, append=True, share=1)
        except PermissionError as error:
            raise RuntimeError(f"Text log is already owned or inaccessible: {log_path}") from error
        try:
            info = os.fstat(descriptor)
            return ProcessTextLogLease(log_path, descriptor, info.st_dev, info.st_ino)
        except BaseException:
            os.close(descriptor)
            raise
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
            lock_exclusive(descriptor)
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


def open_text_stream(lease: ProcessTextLogLease) -> int:
    """Open the writer stream while preserving the platform ownership handle."""
    if os.name == "nt":
        return os.dup(lease._descriptor)
    return os.open(lease.path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
