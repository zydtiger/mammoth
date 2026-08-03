"""Atomic local-filesystem publication helpers used by Mammoth producers.

Core metadata and optional trainer checkpoints call these helpers. Payload
meaning and serialization remain the responsibility of the caller.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _DirectoryHandle:
    """Mutable ownership cell for one prepared artifact's parent descriptor."""

    descriptor: int | None


@dataclass(frozen=True)
class PreparedArtifact:
    """One fully written local artifact awaiting atomic publication."""

    destination: Path
    temporary: Path
    _directory_handle: _DirectoryHandle = field(repr=False, compare=False)


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    """Durably replace ``path`` with opaque bytes from the same directory."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        with suppress(OSError):
            temporary.unlink()
        raise
    return destination


def atomic_write_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> Path:
    """Durably replace ``path`` with encoded text."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    return atomic_write_bytes(Path(path), payload.encode(encoding), mode=mode)


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> Path:
    """Validate and atomically publish a deterministic JSON object."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    serialized = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return atomic_write_text(Path(path), f"{serialized}\n", mode=mode)


def atomic_publish(path: Path, writer: Callable[[Path], None]) -> Path:
    """Publish a caller-written opaque file by same-directory atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        if not temporary.is_file():
            raise FileNotFoundError(f"artifact writer did not create {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    return destination


def prepare_artifact(
    path: Path,
    writer: Callable[[Path], None],
    *,
    mode: int = 0o600,
    preserve_permissions: bool = True,
) -> PreparedArtifact:
    """Serialize and sync one opaque artifact without publishing its destination."""
    validate_artifact_writer(writer, mode)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory_descriptor = os.open(destination.parent, directory_open_flags())
    return prepare_artifact_in_directory(
        destination,
        writer,
        directory_descriptor=directory_descriptor,
        mode=mode,
        preserve_permissions=preserve_permissions,
    )


def prepare_artifact_in_directory(
    path: Path,
    writer: Callable[[Path], None],
    *,
    directory_descriptor: int,
    mode: int = 0o600,
    preserve_permissions: bool = True,
) -> PreparedArtifact:
    """Prepare an artifact while taking ownership of an opened parent directory."""
    validate_artifact_writer(writer, mode)
    destination = Path(path)
    if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
        with suppress(OSError):
            os.close(directory_descriptor)
        raise RuntimeError(f"artifact parent changed before preparation: {destination.parent}")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
            dir_fd=directory_descriptor,
        )
    except BaseException:
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    try:
        if preserve_permissions:
            try:
                destination_mode = (
                    stat.S_IMODE(
                        os.stat(
                            destination.name,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        ).st_mode
                    )
                    & 0o777
                )
            except FileNotFoundError:
                pass
            else:
                os.fchmod(descriptor, destination_mode)
        prepared_mode = stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o777
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        with suppress(OSError):
            os.unlink(temporary.name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    try:
        os.close(descriptor)
    except BaseException:
        if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
            with suppress(OSError):
                temporary.unlink()
        with suppress(OSError):
            os.unlink(temporary.name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    try:
        os.unlink(temporary.name, dir_fd=directory_descriptor)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary.name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise

    try:
        writer(temporary)
        if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
            with suppress(OSError):
                temporary.unlink()
            raise RuntimeError(
                f"artifact parent changed during serialization: {destination.parent}"
            )
        open_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        serialized_descriptor = os.open(
            temporary.name,
            open_flags,
            dir_fd=directory_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(serialized_descriptor).st_mode):
                raise FileNotFoundError(f"artifact writer did not create a file at {temporary}")
            os.fchmod(serialized_descriptor, prepared_mode)
            os.fsync(serialized_descriptor)
        finally:
            os.close(serialized_descriptor)
    except BaseException:
        if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
            with suppress(OSError):
                temporary.unlink()
        with suppress(OSError):
            os.unlink(temporary.name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    return PreparedArtifact(
        destination=destination,
        temporary=temporary,
        _directory_handle=_DirectoryHandle(directory_descriptor),
    )


def validate_artifact_writer(writer: Callable[[Path], None], mode: int) -> None:
    """Validate shared prepared-artifact writer arguments."""
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
        raise ValueError("mode must be an integer from 0o000 through 0o777")
    if not callable(writer):
        raise TypeError("artifact writer must be callable")


def publish_prepared_artifact(artifact: PreparedArtifact) -> Path:
    """Atomically publish one prepared artifact and sync its parent directory."""
    destination = Path(artifact.destination)
    temporary = Path(artifact.temporary)
    if temporary.parent != destination.parent:
        raise ValueError("prepared artifact must use a same-directory temporary file")
    directory_descriptor = artifact._directory_handle.descriptor
    if directory_descriptor is None:
        raise RuntimeError("prepared artifact has already been published or discarded")
    if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
        raise RuntimeError(f"artifact parent changed before publication: {destination.parent}")
    temporary_stat = os.stat(
        temporary.name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(temporary_stat.st_mode):
        raise FileNotFoundError(f"prepared artifact is unavailable: {temporary}")
    os.replace(
        temporary.name,
        destination.name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    try:
        sync_directory_descriptor(directory_descriptor)
    finally:
        artifact._directory_handle.descriptor = None
        with suppress(OSError):
            os.close(directory_descriptor)
    return destination


def discard_prepared_artifact(artifact: PreparedArtifact) -> None:
    """Remove one unpublished temporary artifact if it still exists."""
    directory_descriptor = artifact._directory_handle.descriptor
    if directory_descriptor is None:
        return
    with suppress(OSError):
        os.unlink(Path(artifact.temporary).name, dir_fd=directory_descriptor)
    artifact._directory_handle.descriptor = None
    with suppress(OSError):
        os.close(directory_descriptor)


def directory_open_flags() -> int:
    """Return flags for opening a directory without following its final component."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    return flags | getattr(os, "O_NOFOLLOW", 0)


def directory_path_matches_descriptor(path: Path, descriptor: int) -> bool:
    """Return whether a directory path still names the held directory descriptor."""
    try:
        path_stat = Path(path).stat()
    except OSError:
        return False
    descriptor_stat = os.fstat(descriptor)
    return (path_stat.st_dev, path_stat.st_ino) == (descriptor_stat.st_dev, descriptor_stat.st_ino)


def sync_directory_descriptor(descriptor: int) -> None:
    """Durably sync an already opened directory descriptor."""
    os.fsync(descriptor)


def sync_directory(path: Path) -> None:
    """Best-effort fsync a directory after publishing one of its entries."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
