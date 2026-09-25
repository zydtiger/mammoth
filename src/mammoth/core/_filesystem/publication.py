"""Prepared-file ownership, platform publication, and directory durability."""

from __future__ import annotations

import inspect
import os
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from mammoth.core._filesystem import windows as _windows
from mammoth.core._filesystem.io import validate_regular_artifact_stat


@dataclass
class _DirectoryHandle:
    """Mutable ownership cell for one prepared artifact's staging resources."""

    parent_descriptor: Union[int, None]
    staging_descriptor: Union[int, None]
    staging_name: str
    windows_parent: Union[_windows.PinnedDirectory, None] = None
    windows_staging: Union[_windows.PinnedDirectory, None] = None


@dataclass(frozen=True)
class PreparedArtifact:
    """One fully written local artifact awaiting atomic publication."""

    destination: Path
    temporary: Path
    _directory_handle: _DirectoryHandle = field(repr=False, compare=False)


def prepare_artifact(
    path: Path,
    writer: Callable[[Path], object],
    *,
    mode: Union[int, None] = 0o600,
    preserve_permissions: bool = True,
) -> PreparedArtifact:
    """Serialize and sync one opaque artifact without publishing its destination."""
    validate_artifact_writer(writer, mode)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return _prepare_windows_artifact(
            destination, writer, mode=mode, preserve_permissions=preserve_permissions
        )
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
    mode: Union[int, None] = 0o600,
    preserve_permissions: bool = True,
    inspect_serialized: Union[Callable[[int], object], None] = None,
) -> PreparedArtifact:
    """Prepare and optionally inspect a validated artifact before final permissions."""
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
    staging_descriptor: Union[int, None] = None
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
                raise ValueError(f"artifact destination must be a regular file: {destination}")
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
        open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        serialized_descriptor = os.open(
            artifact_name,
            open_flags,
            dir_fd=staging_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(serialized_descriptor).st_mode):
                raise FileNotFoundError(f"artifact writer did not create a file for {destination}")
            if inspect_serialized is not None:
                inspect_serialized(serialized_descriptor)
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
        "prepared artifact operations require a descriptor filesystem at /proc/self/fd or /dev/fd"
    )


def validate_artifact_writer(writer: Callable[[Path], object], mode: Union[int, None]) -> None:
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
    if os.name == "nt":
        return _publish_windows_artifact(artifact)
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
        ) or {"src_dir_fd", "dst_dir_fd"}.issubset(parameter.name for parameter in parameters)
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
    if os.name == "nt":
        _release_windows_artifact(artifact, unlink=unlink)
        return
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


def _prepare_windows_artifact(
    destination: Path,
    writer: Callable[[Path], object],
    *,
    mode: Union[int, None] = 0o600,
    preserve_permissions: bool = True,
    inspect_serialized: Union[Callable[[int], object], None] = None,
) -> PreparedArtifact:
    """Stage bytes under pinned directories, retaining Windows read-only state."""
    validate_artifact_writer(writer, mode)
    parent = _windows.pin_directory(destination.parent, create=True)
    destination = parent.path / destination.name
    staging = parent.path / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    artifact = PreparedArtifact(
        destination,
        staging / destination.name,
        _DirectoryHandle(None, None, staging.name, windows_parent=parent),
    )
    try:
        try:
            existing = destination.lstat()
        except FileNotFoundError:
            pass
        else:
            validate_regular_artifact_stat(existing, destination)
            if preserve_permissions:
                mode = stat.S_IMODE(existing.st_mode)
        staging.mkdir()
        artifact._directory_handle.windows_staging = _windows.pin_directory(staging)
        writer(artifact.temporary)
        descriptor = _windows.open_file(artifact.temporary, write=True, share=1)
        try:
            if inspect_serialized is not None:
                inspect_serialized(descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if mode is not None:
            os.chmod(artifact.temporary, mode)
        return artifact
    except BaseException:
        _release_windows_artifact(artifact, unlink=True)
        raise


def _publish_windows_artifact(artifact: PreparedArtifact) -> Path:
    """Replace after serializers close their handles, then release directory pins."""
    handle = artifact._directory_handle
    if handle.windows_parent is None or handle.windows_staging is None:
        raise RuntimeError("prepared artifact has already been published or discarded")
    validate_regular_artifact_stat(artifact.temporary.lstat(), artifact.temporary)
    _windows.replace_file(artifact.temporary, artifact.destination)
    _release_windows_artifact(artifact, unlink=False)
    return artifact.destination


def _release_windows_artifact(artifact: PreparedArtifact, *, unlink: bool) -> None:
    handle = artifact._directory_handle
    if handle.windows_parent is None:
        return
    try:
        if unlink:
            with suppress(OSError):
                if artifact.temporary.is_dir() and not artifact.temporary.is_symlink():
                    artifact.temporary.rmdir()
                else:
                    info = artifact.temporary.lstat()
                    if (
                        stat.S_ISREG(info.st_mode)
                        and not getattr(info, "st_file_attributes", 0) & 0x400
                        and not info.st_mode & stat.S_IWRITE
                    ):
                        # Only our unpublished staging file: release its requested read-only bit.
                        os.chmod(artifact.temporary, stat.S_IWRITE)
                    artifact.temporary.unlink()
        if handle.windows_staging is not None:
            handle.windows_staging.close()
            handle.windows_staging = None
        with suppress(OSError):
            artifact.temporary.parent.rmdir()
    finally:
        handle.windows_parent.close()
        handle.windows_parent = None
