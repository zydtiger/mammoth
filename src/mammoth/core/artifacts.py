"""Local artifact publication and descriptor-bound parser access for Mammoth.

Core metadata and optional trainer checkpoints call publication helpers, while
consuming projects use read sessions to parse verified opaque bytes. Payload
meaning and serialization remain the responsibility of the caller.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, BinaryIO, Self

_DEFAULT_ARTIFACT_CHUNK_SIZE = 1024 * 1024


class ArtifactChangedError(RuntimeError):
    """Raised when an artifact changes before its exact bytes are received."""


class ArtifactVerificationError(RuntimeError):
    """Raised when a visible artifact does not match a recorded receipt."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Immutable exact-byte identity for one regular local file observation."""

    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("artifact receipt path must be a pathlib.Path")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact receipt size_bytes must be non-negative")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("artifact receipt sha256 must be lowercase hexadecimal")


class ArtifactReadSession:
    """Context-managed parser access bound to one inspected regular artifact.

    ``open_artifact_session()`` constructs this value for callers that need to
    parse opaque bytes while Mammoth retains the local-file identity boundary.
    Entering the session creates an exact-byte receipt on a private descriptor.
    Each ``open_reader()`` context yields one binary reader over that same file
    object, starting at offset zero.  Readers may seek freely, but only one can
    be active and every reader must close before the outer session exits.

    A successful outer exit rechecks the visible path and exact bytes.  The
    session therefore detects changes at its boundaries, rather than promising
    an immutable snapshot to a seek-heavy parser.  Callers retain all payload
    interpretation and must discard parse results when the session raises.
    """

    def __init__(
        self,
        path: Path,
        *,
        chunk_size: int = _DEFAULT_ARTIFACT_CHUNK_SIZE,
    ) -> None:
        self._path = Path(path)
        validate_artifact_chunk_size(chunk_size)
        self._chunk_size = chunk_size
        self._anchor_descriptor: int | None = None
        self._anchor_stat: os.stat_result | None = None
        self._receipt: ArtifactReceipt | None = None
        self._active_reader: BinaryIO | None = None
        self._entered = False
        self._closed = False
        self._state_lock = Lock()

    @property
    def receipt(self) -> ArtifactReceipt:
        """Return the entry receipt once this session has entered successfully."""
        with self._state_lock:
            if self._receipt is None:
                raise RuntimeError("artifact read session has not entered successfully")
            return self._receipt

    def __enter__(self) -> Self:
        """Open and inspect the private descriptor before parser access begins."""
        with self._state_lock:
            if self._entered:
                raise RuntimeError("artifact read session may only be entered once")
            self._entered = True
            try:
                descriptor, descriptor_stat = _open_artifact_descriptor(self._path)
            except BaseException:
                self._closed = True
                raise
            try:
                receipt = inspect_artifact_descriptor(
                    descriptor,
                    path=self._path,
                    chunk_size=self._chunk_size,
                    verify_stable_reads=True,
                )
                _validate_artifact_session_binding(
                    descriptor,
                    path=self._path,
                    expected_stat=descriptor_stat,
                )
            except BaseException:
                self._closed = True
                os.close(descriptor)
                raise
            self._anchor_descriptor = descriptor
            self._anchor_stat = descriptor_stat
            self._receipt = receipt
            return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Verify successful sessions and release all session-owned descriptors."""
        del exception, traceback
        with self._state_lock:
            descriptor = self._anchor_descriptor
            expected_stat = self._anchor_stat
            reader = self._active_reader
            self._closed = True
            self._active_reader = None
            self._anchor_descriptor = None
            self._anchor_stat = None

        if descriptor is None or expected_stat is None:
            return
        try:
            if reader is not None:
                with suppress(OSError, ValueError):
                    reader.close()
                if exception_type is None:
                    raise RuntimeError("artifact read session cannot exit while a reader is active")
                return
            if exception_type is None:
                _validate_artifact_session_binding(
                    descriptor,
                    path=self._path,
                    expected_stat=expected_stat,
                )
                final_receipt = inspect_artifact_descriptor(
                    descriptor,
                    path=self._path,
                    chunk_size=self._chunk_size,
                    verify_stable_reads=True,
                )
                _validate_artifact_session_binding(
                    descriptor,
                    path=self._path,
                    expected_stat=expected_stat,
                )
                if final_receipt != self.receipt:
                    raise ArtifactChangedError(f"artifact changed while being read: {self._path}")
        finally:
            if exception_type is None:
                os.close(descriptor)
            else:
                with suppress(OSError):
                    os.close(descriptor)

    @contextmanager
    def open_reader(self) -> Iterator[BinaryIO]:
        """Yield one offset-zero binary reader bound to this session's anchor FD.

        The reader is session-owned and must be used through this nested context.
        A parser exception is propagated unchanged after reader cleanup; the
        post-reader binding check runs only when that nested context succeeds.
        """
        with self._state_lock:
            descriptor, expected_stat = self._active_anchor()
            if self._active_reader is not None:
                raise RuntimeError("artifact read session permits only one active reader")
            _validate_artifact_session_binding(
                descriptor,
                path=self._path,
                expected_stat=expected_stat,
            )
            reader_descriptor = os.dup(descriptor)
            try:
                os.lseek(reader_descriptor, 0, os.SEEK_SET)
                reader = os.fdopen(reader_descriptor, "rb")
            except BaseException:
                os.close(reader_descriptor)
                raise
            self._active_reader = reader

        parser_succeeded = False
        try:
            yield reader
            parser_succeeded = True
        finally:
            try:
                if parser_succeeded:
                    reader.close()
                else:
                    with suppress(OSError, ValueError):
                        reader.close()
            finally:
                with self._state_lock:
                    if self._active_reader is reader:
                        self._active_reader = None
            if parser_succeeded:
                with self._state_lock:
                    descriptor, expected_stat = self._active_anchor()
                    _validate_artifact_session_binding(
                        descriptor,
                        path=self._path,
                        expected_stat=expected_stat,
                    )

    def _active_anchor(self) -> tuple[int, os.stat_result]:
        """Return the private descriptor only while the outer session is active."""
        if not self._entered or self._closed:
            raise RuntimeError("artifact read session is not active")
        if self._anchor_descriptor is None or self._anchor_stat is None:
            raise RuntimeError("artifact read session is not active")
        return self._anchor_descriptor, self._anchor_stat


def open_artifact_session(
    path: Path,
    *,
    chunk_size: int = _DEFAULT_ARTIFACT_CHUNK_SIZE,
) -> ArtifactReadSession:
    """Create a one-use descriptor-bound parser session for one local artifact."""
    return ArtifactReadSession(path, chunk_size=chunk_size)


def inspect_artifact(
    path: Path,
    *,
    chunk_size: int = _DEFAULT_ARTIFACT_CHUNK_SIZE,
) -> ArtifactReceipt:
    """Return an exact-byte receipt after safely inspecting one regular file.

    The final path component must not be a symlink.  This uses ``O_NOFOLLOW``
    where available and refuses inspection when the host cannot enforce that
    invariant.  Missing paths preserve ``FileNotFoundError``; directories,
    special files, and symlinks raise ``ValueError``; observed replacement or
    mutation raises ``ArtifactChangedError``.
    """
    artifact_path = Path(path)
    validate_artifact_chunk_size(chunk_size)
    descriptor, descriptor_stat = _open_artifact_descriptor(artifact_path)
    try:
        receipt = inspect_artifact_descriptor(
            descriptor,
            path=artifact_path,
            chunk_size=chunk_size,
            verify_stable_reads=True,
        )
        final_path_stat = os.lstat(artifact_path)
        if not artifact_stats_match(descriptor_stat, final_path_stat):
            raise ArtifactChangedError(f"artifact changed while being inspected: {artifact_path}")
        return receipt
    finally:
        os.close(descriptor)


def _open_artifact_descriptor(path: Path) -> tuple[int, os.stat_result]:
    """Open one regular artifact and bind its descriptor to its visible path."""
    initial_path_stat = os.lstat(path)
    validate_regular_artifact_stat(initial_path_stat, path)
    try:
        descriptor = os.open(path, artifact_open_flags())
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ArtifactChangedError(f"artifact changed before inspection: {path}") from error
        try:
            after_open_failure = os.lstat(path)
        except OSError:
            raise ArtifactChangedError(f"artifact changed before inspection: {path}") from error
        if not artifact_stats_match(initial_path_stat, after_open_failure):
            raise ArtifactChangedError(f"artifact changed before inspection: {path}") from error
        raise
    try:
        descriptor_stat = os.fstat(descriptor)
        if not artifact_stats_match(initial_path_stat, descriptor_stat):
            raise ArtifactChangedError(f"artifact changed before inspection: {path}")
        validate_regular_artifact_stat(descriptor_stat, path)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, descriptor_stat


def _validate_artifact_session_binding(
    descriptor: int,
    *,
    path: Path,
    expected_stat: os.stat_result,
) -> None:
    """Require a session descriptor and visible path to remain the entry artifact."""
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(path)
    except OSError as error:
        raise ArtifactChangedError(f"artifact changed while being read: {path}") from error
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or not artifact_stats_match(expected_stat, descriptor_stat)
        or not artifact_stats_match(expected_stat, path_stat)
    ):
        raise ArtifactChangedError(f"artifact changed while being read: {path}")


def verify_artifact(
    receipt: ArtifactReceipt,
    *,
    chunk_size: int = _DEFAULT_ARTIFACT_CHUNK_SIZE,
) -> None:
    """Require the currently visible artifact to match one exact-byte receipt."""
    if not isinstance(receipt, ArtifactReceipt):
        raise TypeError("receipt must be an ArtifactReceipt")
    observed = inspect_artifact(receipt.path, chunk_size=chunk_size)
    if observed.size_bytes != receipt.size_bytes or observed.sha256 != receipt.sha256:
        raise ArtifactVerificationError(f"artifact does not match receipt: {receipt.path}")


def inspect_artifact_descriptor(
    descriptor: int,
    *,
    path: Path,
    chunk_size: int = _DEFAULT_ARTIFACT_CHUNK_SIZE,
    verify_stable_reads: bool = False,
) -> ArtifactReceipt:
    """Build a receipt from one already-open regular-file descriptor.

    Prepared-artifact and checkpoint publication use this before atomic rename,
    so the receipt remains bound to the exact descriptor that will be committed.
    Path inspection enables a second bounded read to detect same-size changes on
    filesystems whose metadata timestamps are too coarse to expose a mutation.
    """
    if isinstance(descriptor, bool) or not isinstance(descriptor, int):
        raise TypeError("artifact descriptor must be an integer")
    if not isinstance(verify_stable_reads, bool):
        raise TypeError("verify_stable_reads must be a boolean")
    artifact_path = Path(path)
    validate_artifact_chunk_size(chunk_size)
    before = os.fstat(descriptor)
    validate_regular_artifact_stat(before, artifact_path)
    digest = hashlib.sha256()
    size_bytes = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, chunk_size):
        size_bytes += len(chunk)
        digest.update(chunk)
    if verify_stable_reads:
        verification_digest = hashlib.sha256()
        verification_size = 0
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, chunk_size):
            verification_size += len(chunk)
            verification_digest.update(chunk)
        if (verification_size, verification_digest.digest()) != (size_bytes, digest.digest()):
            raise ArtifactChangedError(f"artifact changed while being inspected: {artifact_path}")
    after = os.fstat(descriptor)
    if not artifact_stats_match(before, after):
        raise ArtifactChangedError(f"artifact changed while being inspected: {artifact_path}")
    return ArtifactReceipt(
        path=artifact_path,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def validate_artifact_chunk_size(chunk_size: int) -> None:
    """Reject chunk sizes that cannot provide bounded incremental reads."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


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
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"artifact path must be a regular file: {path}")


def artifact_stats_match(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare filesystem state that detects replacement or byte mutation."""
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


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


def atomic_publish(
    path: Path,
    writer: Callable[[Path], object],
    *,
    inspect_serialized: Callable[[int], object] | None = None,
) -> Path:
    """Publish a validated caller-written file by same-directory atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        try:
            temporary_mode = os.lstat(temporary).st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"artifact writer did not create {temporary}") from None
        if not stat.S_ISREG(temporary_mode):
            raise FileNotFoundError(f"artifact writer did not create {temporary}")
        open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        serialized_descriptor = os.open(temporary, open_flags)
        try:
            if not stat.S_ISREG(os.fstat(serialized_descriptor).st_mode):
                raise FileNotFoundError(f"artifact writer did not create {temporary}")
            if inspect_serialized is not None:
                inspect_serialized(serialized_descriptor)
            os.fsync(serialized_descriptor)
        finally:
            os.close(serialized_descriptor)
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
    inspect_serialized: Callable[[int], object] | None = None,
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
