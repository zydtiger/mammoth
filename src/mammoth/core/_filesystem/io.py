"""Binary descriptor operations and native replacement behind a portable boundary."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import cast

from mammoth.core._filesystem import windows


def artifact_open_flags() -> int:
    """Return secure read flags that reject final-component symlinks."""
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        raise NotImplementedError("artifact inspection requires os.O_NOFOLLOW")
    return os.O_RDONLY | os.O_NONBLOCK | no_follow


def validate_regular_artifact_stat(file_stat: os.stat_result, path: Path) -> None:
    """Reject symlinks and every non-regular artifact type deterministically."""
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"artifact path must not be a symlink: {path}")
    if os.name == "nt" and getattr(file_stat, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Windows reparse points are not supported: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"artifact path must be a regular file: {path}")


def open_artifact(path: Path) -> int:
    """Open an existing regular artifact without following its final component."""
    return windows.open_file(path) if os.name == "nt" else os.open(path, artifact_open_flags())


def open_serialized(path: Path) -> int:
    """Open serializer output for inspection and content synchronization."""
    if os.name == "nt":
        return windows.open_file(path, write=True)
    return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))


def replace_file(source: Path, destination: Path) -> None:
    """Replace one file using the platform's supported namespace operation."""
    if os.name == "nt":
        windows.replace_file(source, destination)
    else:
        os.replace(source, destination)


def open_event_file(path: Path, *, append: bool = False) -> int:
    """Open a binary event descriptor without following symlinks."""
    if os.name == "nt":
        return windows.open_file(path, write=append, create=append, append=append)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT if append else os.O_RDONLY
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)


def read_at(descriptor: int, length: int, offset: int) -> bytes:
    """Read a private reader descriptor at an absolute byte offset."""
    pread = getattr(os, "pread", None)
    if pread is not None:
        return cast(bytes, pread(descriptor, length, offset))
    os.lseek(descriptor, offset, os.SEEK_SET)
    return os.read(descriptor, length)
