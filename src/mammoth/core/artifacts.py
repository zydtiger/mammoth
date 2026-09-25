"""Local artifact publication and descriptor-bound parser access for Mammoth.

Core metadata and optional trainer checkpoints call publication helpers, while
consuming projects use read sessions to parse verified opaque bytes. Payload
meaning and serialization remain the responsibility of the caller.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, BinaryIO, Union

from typing_extensions import Self

from mammoth.compat import DATACLASS_SLOTS
from mammoth.core._filesystem.io import artifact_open_flags as artifact_open_flags
from mammoth.core._filesystem.io import open_artifact, open_serialized, replace_file, stat_artifact
from mammoth.core._filesystem.io import (
    validate_regular_artifact_stat as validate_regular_artifact_stat,
)
from mammoth.core._filesystem.publication import (
    PreparedArtifact as PreparedArtifact,
)
from mammoth.core._filesystem.publication import (
    descriptor_filesystem_path as descriptor_filesystem_path,
)
from mammoth.core._filesystem.publication import (
    descriptor_relative_writer_path as descriptor_relative_writer_path,
)
from mammoth.core._filesystem.publication import (
    directory_open_flags as directory_open_flags,
)
from mammoth.core._filesystem.publication import (
    directory_path_matches_descriptor as directory_path_matches_descriptor,
)
from mammoth.core._filesystem.publication import (
    discard_prepared_artifact as discard_prepared_artifact,
)
from mammoth.core._filesystem.publication import (
    prepare_artifact as prepare_artifact,
)
from mammoth.core._filesystem.publication import (
    prepare_artifact_in_directory as prepare_artifact_in_directory,
)
from mammoth.core._filesystem.publication import (
    publish_prepared_artifact as publish_prepared_artifact,
)
from mammoth.core._filesystem.publication import (
    release_prepared_artifact as release_prepared_artifact,
)
from mammoth.core._filesystem.publication import (
    replace_prepared_artifact as replace_prepared_artifact,
)
from mammoth.core._filesystem.publication import (
    sync_directory as sync_directory,
)
from mammoth.core._filesystem.publication import (
    sync_directory_descriptor as sync_directory_descriptor,
)
from mammoth.core._filesystem.publication import (
    validate_artifact_writer as validate_artifact_writer,
)

_DEFAULT_ARTIFACT_CHUNK_SIZE = 1024 * 1024


class ArtifactChangedError(RuntimeError):
    """Raised when an artifact changes before its exact bytes are received."""


class ArtifactVerificationError(RuntimeError):
    """Raised when a visible artifact does not match a recorded receipt."""


@dataclass(frozen=True, **DATACLASS_SLOTS)
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
        self._anchor_descriptor: Union[int, None] = None
        self._anchor_stat: Union[os.stat_result, None] = None
        self._receipt: Union[ArtifactReceipt, None] = None
        self._active_reader: Union[BinaryIO, None] = None
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
        exception_type: Union[type[BaseException], None],
        exception: Union[BaseException, None],
        traceback: Union[TracebackType, None],
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
        try:
            final_path_stat = stat_artifact(artifact_path)
        except ValueError as error:
            raise ArtifactChangedError(
                f"artifact changed while being inspected: {artifact_path}"
            ) from error
        if not artifact_stats_match(descriptor_stat, final_path_stat):
            raise ArtifactChangedError(f"artifact changed while being inspected: {artifact_path}")
        return receipt
    finally:
        os.close(descriptor)


def _open_artifact_descriptor(path: Path) -> tuple[int, os.stat_result]:
    """Open one regular artifact and bind its descriptor to its visible path."""
    initial_path_stat = stat_artifact(path)
    validate_regular_artifact_stat(initial_path_stat, path)
    try:
        descriptor = open_artifact(path)
    except (OSError, ValueError) as error:
        if isinstance(error, OSError) and error.errno == errno.ELOOP:
            raise ArtifactChangedError(f"artifact changed before inspection: {path}") from error
        try:
            after_open_failure = stat_artifact(path)
        except (OSError, ValueError):
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
        path_stat = stat_artifact(path)
    except (OSError, ValueError) as error:
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
        replace_file(temporary, destination)
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
    inspect_serialized: Union[Callable[[int], object], None] = None,
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
        serialized_descriptor = open_serialized(temporary)
        try:
            if not stat.S_ISREG(os.fstat(serialized_descriptor).st_mode):
                raise FileNotFoundError(f"artifact writer did not create {temporary}")
            if inspect_serialized is not None:
                inspect_serialized(serialized_descriptor)
            os.fsync(serialized_descriptor)
        finally:
            os.close(serialized_descriptor)
        replace_file(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    return destination
