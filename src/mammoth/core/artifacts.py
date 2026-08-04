"""Atomic local-filesystem publication helpers used by Mammoth producers.

Core metadata and optional trainer checkpoints call these helpers. Payload
meaning and serialization remain the responsibility of the caller.
"""

from __future__ import annotations

import inspect
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
    """Mutable ownership cell for one prepared artifact's staging resources."""

    parent_descriptor: int | None
    staging_descriptor: int | None
    staging_name: str


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


def atomic_publish(path: Path, writer: Callable[[Path], object]) -> Path:
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
    writer: Callable[[Path], object],
    *,
    mode: int | None = 0o600,
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
    writer: Callable[[Path], object],
    *,
    directory_descriptor: int,
    mode: int | None = 0o600,
    preserve_permissions: bool = True,
) -> PreparedArtifact:
    """Prepare an artifact while taking ownership of an opened parent directory."""
    try:
        validate_artifact_writer(writer, mode)
    except BaseException:
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    destination = Path(path)
    if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
        with suppress(OSError):
            os.close(directory_descriptor)
        raise RuntimeError(f"artifact parent changed before preparation: {destination.parent}")

    staging_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    artifact_name = destination.name
    staging_descriptor: int | None = None
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=directory_descriptor)
        staging_descriptor = os.open(
            staging_name,
            directory_open_flags(),
            dir_fd=directory_descriptor,
        )
        try:
            destination_stat = os.stat(
                destination.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(destination_stat.st_mode):
                raise ValueError(
                    f"artifact destination must be a regular file: {destination}"
                )
            if preserve_permissions:
                destination_mode = stat.S_IMODE(destination_stat.st_mode) & 0o777
                mode = destination_mode
    except BaseException:
        if staging_descriptor is not None:
            _remove_staged_artifact(artifact_name, staging_descriptor)
            with suppress(OSError):
                os.close(staging_descriptor)
        with suppress(OSError):
            os.rmdir(staging_name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise

    try:
        writer(descriptor_relative_writer_path(staging_descriptor, artifact_name))
        if not directory_path_matches_descriptor(destination.parent, directory_descriptor):
            raise RuntimeError(
                f"artifact parent changed during serialization: {destination.parent}"
            )
        open_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        serialized_descriptor = os.open(
            artifact_name,
            open_flags,
            dir_fd=staging_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(serialized_descriptor).st_mode):
                raise FileNotFoundError(
                    f"artifact writer did not create a file for {destination}"
                )
            if mode is not None:
                os.fchmod(serialized_descriptor, mode)
            os.fsync(serialized_descriptor)
        finally:
            os.close(serialized_descriptor)
    except BaseException:
        _remove_staged_artifact(artifact_name, staging_descriptor)
        with suppress(OSError):
            os.close(staging_descriptor)
        with suppress(OSError):
            os.rmdir(staging_name, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(directory_descriptor)
        raise
    temporary = destination.parent / staging_name / artifact_name
    return PreparedArtifact(
        destination=destination,
        temporary=temporary,
        _directory_handle=_DirectoryHandle(
            parent_descriptor=directory_descriptor,
            staging_descriptor=staging_descriptor,
            staging_name=staging_name,
        ),
    )


def descriptor_relative_writer_path(directory_descriptor: int, name: str) -> Path:
    """Return a serializer path anchored to an already opened directory."""
    return descriptor_filesystem_path(directory_descriptor) / name


def descriptor_filesystem_path(descriptor: int) -> Path:
    """Return the descriptor-filesystem path for one open descriptor."""
    for descriptor_root in (Path("/proc/self/fd"), Path("/dev/fd")):
        anchored_path = descriptor_root / str(descriptor)
        try:
            anchored_path.stat()
        except OSError:
            continue
        return anchored_path
    raise NotImplementedError(
        "prepared artifact operations require a descriptor filesystem at "
        "/proc/self/fd or /dev/fd"
    )


def validate_artifact_writer(writer: Callable[[Path], object], mode: int | None) -> None:
    """Validate shared prepared-artifact writer arguments."""
    if mode is not None and (
        isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777
    ):
        raise ValueError("mode must be None or an integer from 0o000 through 0o777")
    if not callable(writer):
        raise TypeError("artifact writer must be callable")


def _remove_staged_artifact(name: str, directory_descriptor: int) -> None:
    """Best-effort removal for a writer-created file or empty directory."""
    try:
        staged_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(staged_stat.st_mode):
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            os.unlink(name, dir_fd=directory_descriptor)
    except OSError:
        return


def publish_prepared_artifact(artifact: PreparedArtifact) -> Path:
    """Atomically publish one prepared artifact and sync its parent directory."""
    destination = Path(artifact.destination)
    temporary = Path(artifact.temporary)
    parent_descriptor = artifact._directory_handle.parent_descriptor
    staging_descriptor = artifact._directory_handle.staging_descriptor
    if parent_descriptor is None or staging_descriptor is None:
        raise RuntimeError("prepared artifact has already been published or discarded")
    if not directory_path_matches_descriptor(destination.parent, parent_descriptor):
        raise RuntimeError(f"artifact parent changed before publication: {destination.parent}")
    temporary_stat = os.stat(
        temporary.name,
        dir_fd=staging_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(temporary_stat.st_mode):
        raise FileNotFoundError(f"prepared artifact is unavailable: {temporary}")
    replace_prepared_artifact(
        temporary.name,
        destination.name,
        staging_descriptor=staging_descriptor,
        parent_descriptor=parent_descriptor,
    )
    try:
        os.rmdir(artifact._directory_handle.staging_name, dir_fd=parent_descriptor)
        sync_directory_descriptor(parent_descriptor)
    finally:
        try:
            os.close(staging_descriptor)
        finally:
            artifact._directory_handle.staging_descriptor = None
        try:
            os.close(parent_descriptor)
        finally:
            artifact._directory_handle.parent_descriptor = None
    return destination


def replace_prepared_artifact(
    source_name: str,
    destination_name: str,
    *,
    staging_descriptor: int,
    parent_descriptor: int,
) -> None:
    """Replace through descriptor anchors without masking adapter failures."""
    try:
        parameters = inspect.signature(os.replace).parameters.values()
    except (TypeError, ValueError):
        supports_descriptors = inspect.isbuiltin(os.replace)
    else:
        supports_descriptors = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ) or {"src_dir_fd", "dst_dir_fd"}.issubset(
            parameter.name for parameter in parameters
        )
    if supports_descriptors:
        os.replace(
            source_name,
            destination_name,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=parent_descriptor,
        )
    else:
        os.replace(
            descriptor_relative_writer_path(staging_descriptor, source_name),
            descriptor_relative_writer_path(parent_descriptor, destination_name),
        )


def release_prepared_artifact(artifact: PreparedArtifact, *, unlink: bool) -> None:
    """Release one prepared artifact's private staging resources."""
    parent_descriptor = artifact._directory_handle.parent_descriptor
    staging_descriptor = artifact._directory_handle.staging_descriptor
    if parent_descriptor is None or staging_descriptor is None:
        return
    if unlink:
        with suppress(OSError):
            os.unlink(Path(artifact.temporary).name, dir_fd=staging_descriptor)
    with suppress(OSError):
        os.close(staging_descriptor)
    artifact._directory_handle.staging_descriptor = None
    with suppress(OSError):
        os.rmdir(artifact._directory_handle.staging_name, dir_fd=parent_descriptor)
    with suppress(OSError):
        os.close(parent_descriptor)
    artifact._directory_handle.parent_descriptor = None


def discard_prepared_artifact(artifact: PreparedArtifact) -> None:
    """Remove one unpublished temporary artifact if it still exists."""
    release_prepared_artifact(artifact, unlink=True)


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
