"""Retireable local-filesystem lease namespaces for publication lifecycles.

Framework-neutral publishers use this module to keep advisory lock inodes in a
private, caller-selected directory.  Ordinary close preserves that directory
for crash recovery; explicit retirement durably removes the canonical
generation before releasing its descriptor and then reclaims only authenticated
Mammoth-owned state.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LEASE_NAMESPACE_SCHEMA_VERSION = 1

_METADATA_NAME = ".mammoth-lease-namespace.json"
_LOCK_NAME = ".mammoth-lease.lock"
_TERMINAL_NAME = ".mammoth-lease-terminal.json"
_KNOWN_CHILDREN = frozenset({_METADATA_NAME, _LOCK_NAME, _TERMINAL_NAME})
_RETIRED_PROOF_SUFFIX = ".mammoth-retired-proof"
_CREATING_SUFFIX = ".mammoth-creating"
_RENAME_NOREPLACE = 1

type LeaseNamespaceRecoveryStatus = Literal["absent", "active", "reclaimed"]


class LeaseNamespaceError(RuntimeError):
    """Base error for unsafe or unsupported lease-namespace state."""


class LeaseNamespaceConflictError(LeaseNamespaceError):
    """Raised when another process owns the canonical namespace generation."""


class LeaseNamespaceRecoveryError(LeaseNamespaceError):
    """Raised when crash recovery cannot authenticate state before cleanup."""


@dataclass(slots=True)
class RetireableLeaseNamespace:
    """Own one authenticated canonical lease-namespace generation."""

    path: Path
    generation: str
    _directory_descriptor: int
    _lease_descriptor: int
    _owned_parents: tuple[Path, ...] = ()
    _closed: bool = False

    def close(self) -> None:
        """Release ownership while preserving the namespace for later recovery."""
        if self._closed:
            return
        try:
            fcntl.flock(self._lease_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._lease_descriptor)
            os.close(self._directory_descriptor)
            self._closed = True

    def terminalize(self) -> None:
        """Durably retire and reclaim this generation after caller validation.

        The caller owns publication semantics and must invoke this method only
        after validating its complete final artifact bundle.
        """
        if self._closed:
            return
        try:
            self._terminalize_owned_generation()
        except BaseException:
            self.close()
            raise

    def _terminalize_owned_generation(self) -> None:
        """Retire the held generation, leaving public failure paths released."""
        _require_current_generation(self)
        retired_path = _retired_path(self.path)
        if os.path.lexists(retired_path) or os.path.lexists(_retired_proof_path(self.path)):
            raise LeaseNamespaceRecoveryError(
                "A retired lease namespace already exists beside the active generation; "
                f"preserving both as ambiguous evidence: {self.path}"
            )
        _write_terminal_marker(self)
        _require_current_generation(self)
        parent_descriptor = _open_directory(self.path.parent, require_owned=False)
        try:
            canonical_stat = os.stat(
                self.path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            held_stat = os.fstat(self._directory_descriptor)
            if (canonical_stat.st_dev, canonical_stat.st_ino) != (
                held_stat.st_dev,
                held_stat.st_ino,
            ):
                raise LeaseNamespaceRecoveryError(
                    f"Canonical lease namespace changed before retirement: {self.path}"
                )
            _rename_no_replace_at(
                parent_descriptor,
                self.path.name,
                parent_descriptor,
                retired_path.name,
            )
            os.fsync(parent_descriptor)
        except OSError as error:
            raise LeaseNamespaceRecoveryError(
                f"Lease namespace could not be atomically retired: {self.path}"
            ) from error
        finally:
            os.close(parent_descriptor)
        _remove_authenticated_retired(
            retired_path,
            canonical_path=self.path,
            expected_generation=self.generation,
            releasing_lease=self,
        )
        _remove_empty_owned_parents(self._owned_parents)

    def retire(self) -> None:
        """Alias :meth:`terminalize` using the filesystem transition name."""
        self.terminalize()

    def __enter__(self) -> RetireableLeaseNamespace:
        """Return this acquired namespace for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release ownership without terminalizing recoverable state."""
        self.close()


def claim_lease_namespace(
    path: Path,
    *,
    owned_parents: tuple[Path, ...] = (),
) -> RetireableLeaseNamespace:
    """Create or acquire one stable same-filesystem lease namespace.

    Acquisition is nonblocking.  After ``flock`` succeeds, the canonical path
    is revalidated against the already-open directory and lock descriptors.
    A contender paused across retirement therefore rejects the old inode rather
    than proceeding concurrently with a newly created canonical generation.
    """
    canonical = _absolute_path(path)
    _require_no_follow_support()
    parents = _validate_owned_parents(canonical, owned_parents)
    for _attempt in range(8):
        try:
            _reconcile_retired_only(canonical, parents)
        except FileNotFoundError:
            continue
        try:
            _create_namespace_if_absent(canonical)
        except FileNotFoundError:
            continue
        try:
            lease = _claim_existing_namespace(canonical, parents)
        except FileNotFoundError:
            continue
        if _terminal_marker_exists(lease._directory_descriptor):
            lease.retire()
            continue
        return lease
    raise LeaseNamespaceRecoveryError(
        f"Lease namespace changed repeatedly during acquisition: {canonical}."
    )


def reconcile_lease_namespace(
    path: Path,
    *,
    owned_parents: tuple[Path, ...] = (),
) -> LeaseNamespaceRecoveryStatus:
    """Reclaim authenticated terminal crash state without creating new state."""
    canonical = _absolute_path(path)
    _require_no_follow_support()
    parents = _validate_owned_parents(canonical, owned_parents)
    for _attempt in range(8):
        try:
            reclaimed = _reconcile_retired_only(canonical, parents)
        except FileNotFoundError:
            continue
        if not os.path.lexists(canonical):
            _remove_empty_owned_parents(parents)
            return "reclaimed" if reclaimed else "absent"
        try:
            lease = _claim_existing_namespace(canonical, parents)
        except FileNotFoundError:
            continue
        except LeaseNamespaceConflictError:
            return "active"
        if _terminal_marker_exists(lease._directory_descriptor):
            lease.retire()
            return "reclaimed"
        lease.close()
        return "active"
    raise LeaseNamespaceRecoveryError(
        f"Lease namespace changed repeatedly during reconciliation: {canonical}."
    )


def _claim_existing_namespace(
    canonical: Path, parents: tuple[Path, ...]
) -> RetireableLeaseNamespace:
    """Acquire one existing generation without creating replacement state."""
    directory_descriptor = _open_directory(canonical)
    lease_descriptor = -1
    try:
        metadata = _read_metadata(directory_descriptor, canonical)
        generation = _metadata_generation(metadata, canonical)
        lease_descriptor = _open_lock(directory_descriptor, canonical)
        try:
            fcntl.flock(lease_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LeaseNamespaceConflictError(
                f"Another process owns lease namespace {canonical}."
            ) from error
        except OSError as error:
            raise LeaseNamespaceError(
                f"Lease namespace could not be locked: {canonical}."
            ) from error
        lease = RetireableLeaseNamespace(
            path=canonical,
            generation=generation,
            _directory_descriptor=directory_descriptor,
            _lease_descriptor=lease_descriptor,
            _owned_parents=parents,
        )
        _require_current_generation(lease)
        return lease
    except BaseException:
        if lease_descriptor >= 0:
            try:
                fcntl.flock(lease_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lease_descriptor)
        os.close(directory_descriptor)
        raise


def _create_namespace_if_absent(path: Path) -> None:
    """Durably promote one recoverable scratch generation when absent."""
    parent_descriptor = _open_or_create_directory(path.parent)
    creating_path = _creating_path(path)
    try:
        if _entry_exists(parent_descriptor, path.name):
            _discard_creating_namespace(path, parent_descriptor)
            _require_safe_directory(path, description="lease namespace")
            return
        try:
            os.mkdir(creating_path.name, 0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileExistsError:
            pass
        directory_descriptor = os.open(
            creating_path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        _require_owned_mode(os.fstat(directory_descriptor), path=creating_path, directory=True)
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LeaseNamespaceConflictError(
                f"Another process is creating lease namespace {path}."
            ) from error
        children = set(os.listdir(directory_descriptor))
        unknown = children - {_METADATA_NAME, _LOCK_NAME}
        if unknown:
            raise LeaseNamespaceRecoveryError(
                f"Lease namespace creation scratch has unknown children: {creating_path}"
            )
        for name in children:
            child_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            _require_owned_mode(child_stat, path=creating_path / name, directory=False)
            os.unlink(name, dir_fd=directory_descriptor)
        generation = uuid.uuid4().hex
        payload = {
            "schema_version": LEASE_NAMESPACE_SCHEMA_VERSION,
            "generation": generation,
            "canonical_path": str(path),
        }
        _write_new_json_at(directory_descriptor, _METADATA_NAME, payload)
        lease_descriptor = _open_lock(directory_descriptor, path)
        os.fsync(lease_descriptor)
        os.close(lease_descriptor)
        os.fsync(directory_descriptor)
        try:
            _rename_no_replace_at(
                parent_descriptor,
                creating_path.name,
                parent_descriptor,
                path.name,
            )
        except FileExistsError:
            for name in (_LOCK_NAME, _METADATA_NAME):
                os.unlink(name, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
            os.rmdir(creating_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(directory_descriptor)
        os.close(parent_descriptor)


def _discard_creating_namespace(path: Path, parent_descriptor: int) -> None:
    """Reclaim an abandoned authenticated creation scratch beside canonical state."""
    creating_path = _creating_path(path)
    if not _entry_exists(parent_descriptor, creating_path.name):
        return
    descriptor = os.open(
        creating_path.name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        _require_owned_mode(os.fstat(descriptor), path=creating_path, directory=True)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        children = set(os.listdir(descriptor))
        unknown = children - {_METADATA_NAME, _LOCK_NAME}
        if unknown:
            raise LeaseNamespaceRecoveryError(
                f"Lease namespace creation scratch has unknown children: {creating_path}"
            )
        for name in children:
            child_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _require_owned_mode(child_stat, path=creating_path / name, directory=False)
            os.unlink(name, dir_fd=descriptor)
        os.fsync(descriptor)
        os.rmdir(creating_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(descriptor)


def _reconcile_retired_only(path: Path, owned_parents: tuple[Path, ...]) -> bool:
    """Remove a lone authenticated retired generation left by a crash."""
    retired_path = _retired_path(path)
    proof_path = _retired_proof_path(path)
    canonical_exists = os.path.lexists(path)
    retired_exists = os.path.lexists(retired_path)
    proof_exists = os.path.lexists(proof_path)
    if canonical_exists and (retired_exists or proof_exists):
        raise LeaseNamespaceRecoveryError(
            "Active and retired lease namespace state both exist; preserving ambiguous evidence: "
            f"{path}"
        )
    if not retired_exists and not proof_exists:
        return False
    if not retired_exists:
        _remove_retirement_proof(proof_path, canonical_path=path, expected_generation=None)
        _remove_empty_owned_parents(owned_parents)
        return True
    _remove_authenticated_retired(
        retired_path,
        canonical_path=path,
        expected_generation=None,
    )
    _remove_empty_owned_parents(owned_parents)
    return True


def _require_current_generation(lease: RetireableLeaseNamespace) -> None:
    """Bind canonical and child paths back to the held descriptors."""
    parent_descriptor = _open_directory(lease.path.parent, require_owned=False)
    try:
        canonical_stat = os.stat(lease.path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    finally:
        os.close(parent_descriptor)
    directory_stat = os.fstat(lease._directory_descriptor)
    if not stat.S_ISDIR(canonical_stat.st_mode) or (
        canonical_stat.st_dev,
        canonical_stat.st_ino,
    ) != (directory_stat.st_dev, directory_stat.st_ino):
        raise LeaseNamespaceRecoveryError(
            f"Canonical lease namespace generation changed during acquisition: {lease.path}"
        )
    lock_stat = os.stat(_LOCK_NAME, dir_fd=lease._directory_descriptor, follow_symlinks=False)
    descriptor_stat = os.fstat(lease._lease_descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or (lock_stat.st_dev, lock_stat.st_ino) != (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
    ):
        raise LeaseNamespaceRecoveryError(
            f"Lease inode changed during acquisition: {lease.path / _LOCK_NAME}"
        )
    metadata = _read_metadata(lease._directory_descriptor, lease.path)
    if _metadata_generation(metadata, lease.path) != lease.generation:
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace generation metadata changed: {lease.path}"
        )


def _write_terminal_marker(lease: RetireableLeaseNamespace) -> None:
    """Publish and synchronize the caller-declared terminal boundary."""
    payload = {
        "schema_version": LEASE_NAMESPACE_SCHEMA_VERSION,
        "generation": lease.generation,
        "canonical_path": str(lease.path),
        "terminal": True,
    }
    try:
        _write_new_json_at(lease._directory_descriptor, _TERMINAL_NAME, payload)
    except FileExistsError:
        existing = _read_json_at(lease._directory_descriptor, _TERMINAL_NAME, lease.path)
        if existing != payload:
            raise LeaseNamespaceRecoveryError(
                f"Lease terminal marker is inconsistent: {lease.path / _TERMINAL_NAME}"
            ) from None
    os.fsync(lease._directory_descriptor)


def _remove_authenticated_retired(
    retired_path: Path,
    *,
    canonical_path: Path,
    expected_generation: str | None,
    releasing_lease: RetireableLeaseNamespace | None = None,
) -> None:
    """Delete terminal state, including authenticated partial-deletion remnants."""
    proof_path = _retired_proof_path(canonical_path)
    descriptor = _open_directory(retired_path)
    retired_lock_descriptor = -1
    proof_descriptor = -1
    parent_descriptor = -1
    lock_identity: tuple[int, int] | None = None
    try:
        children = set(os.listdir(descriptor))
        unknown = children - _KNOWN_CHILDREN
        if unknown:
            raise LeaseNamespaceRecoveryError(
                f"Retired lease namespace has unknown children; preserving evidence: {retired_path}"
            )
        if _LOCK_NAME in children:
            current_lock_stat = os.stat(_LOCK_NAME, dir_fd=descriptor, follow_symlinks=False)
            _require_owned_mode(
                current_lock_stat,
                path=retired_path / _LOCK_NAME,
                directory=False,
            )
            if releasing_lease is not None:
                held_lock_stat = os.fstat(releasing_lease._lease_descriptor)
                if (current_lock_stat.st_dev, current_lock_stat.st_ino) != (
                    held_lock_stat.st_dev,
                    held_lock_stat.st_ino,
                ):
                    raise LeaseNamespaceRecoveryError(
                        f"Retired lease inode changed before handoff: {retired_path}"
                    )
                lock_identity = (held_lock_stat.st_dev, held_lock_stat.st_ino)
            else:
                retired_lock_descriptor = os.open(
                    _LOCK_NAME,
                    os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                _require_owned_mode(
                    os.fstat(retired_lock_descriptor),
                    path=retired_path / _LOCK_NAME,
                    directory=False,
                )
                try:
                    fcntl.flock(retired_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise LeaseNamespaceConflictError(
                        f"Retired lease namespace is still owned: {retired_path}"
                    ) from error
                held_lock_stat = os.fstat(retired_lock_descriptor)
                lock_identity = (held_lock_stat.st_dev, held_lock_stat.st_ino)
        if _TERMINAL_NAME in children:
            terminal = _read_json_at(descriptor, _TERMINAL_NAME, retired_path)
            generation = _validate_terminal_payload(
                terminal, canonical_path=canonical_path, context=retired_path
            )
        elif os.path.lexists(proof_path):
            terminal = _read_retirement_proof(proof_path, canonical_path=canonical_path)
            generation = _validate_terminal_payload(
                terminal, canonical_path=canonical_path, context=proof_path
            )
        else:
            raise LeaseNamespaceRecoveryError(
                "Partially deleted retired namespace lacks its terminal authenticator; "
                f"preserving evidence: {retired_path}"
            )
        if expected_generation is not None and generation != expected_generation:
            raise LeaseNamespaceRecoveryError(
                f"Retired lease namespace generation mismatch: {retired_path}"
            )
        if _METADATA_NAME in children:
            metadata = _read_metadata(descriptor, canonical_path)
            if _metadata_generation(metadata, canonical_path) != generation:
                raise LeaseNamespaceRecoveryError(
                    f"Retired lease metadata and terminal generations differ: {retired_path}"
                )
        parent_descriptor = _open_directory(retired_path.parent, require_owned=False)
        if _TERMINAL_NAME in children:
            if os.path.lexists(proof_path):
                raise LeaseNamespaceRecoveryError(
                    f"Retired namespace and proof overlap ambiguously: {proof_path}"
                )
            _rename_no_replace_at(
                descriptor,
                _TERMINAL_NAME,
                parent_descriptor,
                proof_path.name,
            )
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        proof_descriptor = os.open(
            proof_path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        proof_stat = os.fstat(proof_descriptor)
        _require_owned_mode(proof_stat, path=proof_path, directory=False)
        proof = _read_json_descriptor(proof_descriptor, proof_path)
        proof_generation = _validate_terminal_payload(
            proof, canonical_path=canonical_path, context=proof_path
        )
        if proof_generation != generation:
            raise LeaseNamespaceRecoveryError(
                f"Lease retirement proof generation mismatch: {proof_path}"
            )
        try:
            fcntl.flock(proof_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LeaseNamespaceConflictError(
                f"Lease retirement proof is still owned: {proof_path}"
            ) from error
        if releasing_lease is not None:
            releasing_lease.close()
        current_proof_stat = os.stat(
            proof_path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (current_proof_stat.st_dev, current_proof_stat.st_ino) != (
            proof_stat.st_dev,
            proof_stat.st_ino,
        ):
            raise LeaseNamespaceRecoveryError(
                f"Lease retirement proof changed during cleanup: {proof_path}"
            )
        if _METADATA_NAME in children:
            child_stat = os.stat(_METADATA_NAME, dir_fd=descriptor, follow_symlinks=False)
            _require_owned_mode(
                child_stat,
                path=retired_path / _METADATA_NAME,
                directory=False,
            )
            os.unlink(_METADATA_NAME, dir_fd=descriptor)
            os.fsync(descriptor)
        if _LOCK_NAME in children:
            current_lock_stat = os.stat(_LOCK_NAME, dir_fd=descriptor, follow_symlinks=False)
            if (
                lock_identity is None
                or (
                    current_lock_stat.st_dev,
                    current_lock_stat.st_ino,
                )
                != lock_identity
            ):
                raise LeaseNamespaceRecoveryError(
                    f"Retired lease inode changed during cleanup: {retired_path}"
                )
            os.unlink(_LOCK_NAME, dir_fd=descriptor)
            os.fsync(descriptor)
        if os.listdir(descriptor):
            raise LeaseNamespaceRecoveryError(
                f"Retired lease namespace changed during cleanup: {retired_path}"
            )
        os.rmdir(retired_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        current_proof_stat = os.stat(
            proof_path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (current_proof_stat.st_dev, current_proof_stat.st_ino) != (
            proof_stat.st_dev,
            proof_stat.st_ino,
        ):
            raise LeaseNamespaceRecoveryError(
                f"Lease retirement proof changed before removal: {proof_path}"
            )
        os.unlink(proof_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if proof_descriptor >= 0:
            try:
                fcntl.flock(proof_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(proof_descriptor)
        if retired_lock_descriptor >= 0:
            try:
                fcntl.flock(retired_lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(retired_lock_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        os.close(descriptor)


def _open_directory(path: Path, *, require_owned: bool = True) -> int:
    """Open a directory through an anchored no-follow walk."""
    _require_no_follow_support()
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        current = Path("/")
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current /= component
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise LeaseNamespaceRecoveryError(f"Lease namespace path is not a directory: {path}")
        if require_owned:
            _require_owned_mode(metadata, path=absolute, directory=True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_directory(path: Path) -> int:
    """Create missing ancestors through a root-anchored no-follow walk."""
    _require_no_follow_support()
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_lock(directory_descriptor: int, namespace_path: Path) -> int:
    """Open the namespace's fixed regular lock inode relative to its directory."""
    _require_no_follow_support()
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | os.O_NOFOLLOW
    descriptor = os.open(_LOCK_NAME, flags, 0o600, dir_fd=directory_descriptor)
    try:
        _require_owned_mode(os.fstat(descriptor), path=namespace_path / _LOCK_NAME, directory=False)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_metadata(directory_descriptor: int, canonical_path: Path) -> dict[str, Any]:
    """Read strict generation metadata through the held directory descriptor."""
    return _read_json_at(directory_descriptor, _METADATA_NAME, canonical_path)


def _read_metadata_from_path(retired_path: Path, *, canonical_path: Path) -> dict[str, Any]:
    """Read metadata from an authenticated retired directory."""
    descriptor = _open_directory(retired_path)
    try:
        return _read_metadata(descriptor, canonical_path)
    finally:
        os.close(descriptor)


def _read_json_at(directory_descriptor: int, name: str, context: Path) -> dict[str, Any]:
    """Read one owner-only regular JSON file relative to a held directory."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace metadata is missing or unsafe: {context / name}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        _require_owned_mode(file_stat, path=context / name, directory=False)
        payload = _read_json_descriptor(descriptor, context / name)
    finally:
        os.close(descriptor)
    return payload


def _read_json_descriptor(descriptor: int, context: Path) -> dict[str, Any]:
    """Parse one JSON object without transferring ownership of its descriptor."""
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace metadata is invalid: {context}"
        ) from error
    if not isinstance(payload, dict):
        raise LeaseNamespaceRecoveryError(f"Lease namespace metadata must be an object: {context}")
    return payload


def _write_new_json_at(directory_descriptor: int, name: str, payload: dict[str, Any]) -> None:
    """Create and fsync one owner-only JSON file relative to a held directory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    try:
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError(f"short lease metadata write: {written} of {len(data)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _metadata_generation(payload: dict[str, Any], canonical_path: Path) -> str:
    """Validate metadata and return its generation identity."""
    generation = payload.get("generation")
    if (
        payload.get("schema_version") != LEASE_NAMESPACE_SCHEMA_VERSION
        or not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
        or payload.get("canonical_path") != str(canonical_path)
    ):
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace generation metadata is incompatible: {canonical_path}"
        )
    return generation


def _terminal_marker_exists(directory_descriptor: int) -> bool:
    """Return whether a no-follow terminal marker is present."""
    try:
        marker_stat = os.stat(_TERMINAL_NAME, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(marker_stat.st_mode):
        raise LeaseNamespaceRecoveryError("Lease namespace terminal marker is unsafe.")
    return True


def _retired_path(path: Path) -> Path:
    """Return the deterministic crash-recoverable retired sibling."""
    return path.parent / f".{path.name}.mammoth-retired"


def _creating_path(path: Path) -> Path:
    """Return the deterministic recoverable creation scratch path."""
    return path.parent / f".{path.name}{_CREATING_SUFFIX}"


def _retired_proof_path(path: Path) -> Path:
    """Return the sibling proof that authenticates an emptied retired directory."""
    return path.parent / f".{path.name}{_RETIRED_PROOF_SUFFIX}"


def _entry_exists(directory_descriptor: int, name: str) -> bool:
    """Check one exact no-follow directory entry."""
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _rename_no_replace_at(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    """Atomically rename one descriptor-relative entry without replacement."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError(
            "retireable lease namespaces require Linux renameat2(RENAME_NOREPLACE)"
        )
    result = renameat2(
        source_directory,
        os.fsencode(source_name),
        destination_directory,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _require_no_follow_support() -> None:
    """Fail closed when the platform cannot reject symlink traversal."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("retireable lease namespaces require os.O_NOFOLLOW")


def _read_retirement_proof(proof_path: Path, *, canonical_path: Path) -> dict[str, Any]:
    """Read and validate a no-follow owner-only retirement proof."""
    parent_descriptor = _open_directory(proof_path.parent, require_owned=False)
    try:
        proof = _read_json_at(parent_descriptor, proof_path.name, proof_path.parent)
    finally:
        os.close(parent_descriptor)
    _validate_terminal_payload(proof, canonical_path=canonical_path, context=proof_path)
    return proof


def _validate_terminal_payload(
    payload: dict[str, Any], *, canonical_path: Path, context: Path
) -> str:
    """Authenticate one complete terminal marker or retirement proof."""
    generation = payload.get("generation")
    if (
        payload.get("schema_version") != LEASE_NAMESPACE_SCHEMA_VERSION
        or payload.get("terminal") is not True
        or payload.get("canonical_path") != str(canonical_path)
        or not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace terminal evidence is invalid: {context}"
        )
    return generation


def _remove_retirement_proof(
    proof_path: Path,
    *,
    canonical_path: Path,
    expected_generation: str | None,
) -> None:
    """Remove an authenticated proof after its retired directory is gone."""
    parent_descriptor = _open_directory(proof_path.parent, require_owned=False)
    descriptor = -1
    try:
        descriptor = os.open(proof_path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
        proof_stat = os.fstat(descriptor)
        _require_owned_mode(proof_stat, path=proof_path, directory=False)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LeaseNamespaceConflictError(
                f"Lease retirement proof is still owned: {proof_path}"
            ) from error
        proof = _read_json_descriptor(descriptor, proof_path)
        generation = _validate_terminal_payload(
            proof, canonical_path=canonical_path, context=proof_path
        )
        if expected_generation is not None and generation != expected_generation:
            raise LeaseNamespaceRecoveryError(
                f"Lease retirement proof generation mismatch: {proof_path}"
            )
        current_stat = os.stat(proof_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (current_stat.st_dev, current_stat.st_ino) != (
            proof_stat.st_dev,
            proof_stat.st_ino,
        ):
            raise LeaseNamespaceRecoveryError(
                f"Lease retirement proof changed before removal: {proof_path}"
            )
        os.unlink(proof_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _absolute_path(path: Path) -> Path:
    """Normalize a caller path without following its final component."""
    value = Path(path)
    if value.name in {"", ".", ".."}:
        raise LeaseNamespaceError(f"Lease namespace path must name a child directory: {path}")
    return Path(os.path.abspath(value))


def _validate_owned_parents(path: Path, parents: tuple[Path, ...]) -> tuple[Path, ...]:
    """Require explicit cleanup parents to be ancestors, deepest first."""
    normalized = tuple(_absolute_path(parent) for parent in parents)
    if any(path == parent or not path.is_relative_to(parent) for parent in normalized):
        raise LeaseNamespaceError("Owned cleanup parents must strictly contain the namespace.")
    return tuple(sorted(dict.fromkeys(normalized), key=lambda item: len(item.parts), reverse=True))


def _remove_empty_owned_parents(parents: tuple[Path, ...]) -> None:
    """Remove only authenticated empty directories explicitly owned by the caller."""
    for parent in parents:
        parent_descriptor = _open_directory(parent.parent, require_owned=False)
        try:
            try:
                candidate_descriptor = os.open(
                    parent.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                continue
            try:
                _require_owned_mode(os.fstat(candidate_descriptor), path=parent, directory=True)
                try:
                    os.rmdir(parent.name, dir_fd=parent_descriptor)
                except OSError as error:
                    if error.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                        continue
                    raise LeaseNamespaceRecoveryError(
                        f"Empty Mammoth-owned lease parent could not be removed: {parent}"
                    ) from error
                os.fsync(parent_descriptor)
            finally:
                os.close(candidate_descriptor)
        finally:
            os.close(parent_descriptor)


def _require_safe_directory(path: Path, *, description: str) -> None:
    """Authenticate a path-bound owner-only directory without following symlinks."""
    try:
        descriptor = _open_directory(path)
    except (OSError, LeaseNamespaceRecoveryError) as error:
        raise LeaseNamespaceError(f"{description} is not safely accessible: {path}") from error
    os.close(descriptor)


def _require_owned_mode(metadata: os.stat_result, *, path: Path, directory: bool) -> None:
    """Require the current effective user and no group/other write permission."""
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(metadata.st_mode):
        raise LeaseNamespaceRecoveryError(f"Lease namespace object has unsafe type: {path}")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise LeaseNamespaceRecoveryError(
            f"Lease namespace object has unsafe ownership or permissions: {path}"
        )


def _fsync_directory(path: Path) -> None:
    """Synchronize one directory boundary used by namespace transitions."""
    descriptor = _open_directory(path, require_owned=False)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise LeaseNamespaceRecoveryError(f"Directory sync target is unsafe: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
