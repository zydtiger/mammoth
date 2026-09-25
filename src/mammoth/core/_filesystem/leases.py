"""Windows logical-run ownership with a stable kernel-owned coordinator.

A persistent empty coordinator prevents stale-lock races during namespace
retirement. It is separate from POSIX publication namespaces, which remain
unsupported on Windows. All recovery below runs with exclusive ownership.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mammoth.core._filesystem import windows as _windows
from mammoth.core._filesystem.locking import lock_exclusive
from mammoth.core.artifacts import atomic_write_json, open_artifact_session
from mammoth.core.leases import (
    LeaseNamespaceConflictError,
    LeaseNamespaceRecoveryError,
    claim_lease_namespace,
)

WINDOWS_RUN_LOCK = ".mammoth-windows.lock"
_METADATA = ".mammoth-windows-lease.json"
_TERMINAL = ".mammoth-windows-terminal.json"


def _read(path: Path) -> dict[str, Any]:
    with open_artifact_session(path) as session, session.open_reader() as reader:
        payload = json.load(reader)
    if not isinstance(payload, dict):
        raise LeaseNamespaceRecoveryError(f"Invalid Windows lease metadata: {path}")
    return payload


def _validate(path: Path, canonical: Path) -> dict[str, Any]:
    with _windows.pin_directory(path):
        names = {child.name for child in path.iterdir()}
        if _METADATA not in names or names - {_METADATA, _TERMINAL}:
            raise LeaseNamespaceRecoveryError(f"Unknown or incomplete Windows lease state: {path}")
        payload = _read(path / _METADATA)
        if (
            payload.get("schema") != "mammoth-windows-run-lease-v1"
            or payload.get("path") != os.path.normcase(str(canonical))
            or not isinstance(payload.get("generation"), str)
            or len(payload["generation"]) != 32
        ):
            raise LeaseNamespaceRecoveryError(f"Invalid Windows lease generation: {path}")
        if _TERMINAL in names and _read(path / _TERMINAL) != payload:
            raise LeaseNamespaceRecoveryError(f"Windows lease terminal marker disagrees: {path}")
        return payload


def _retired(path: Path) -> Path:
    return path.with_name(f".{path.name}.mammoth-retired")


def _reclaim(path: Path, canonical: Path) -> None:
    # Metadata is removed last so a partial cleanup can still be authenticated.
    with _windows.pin_directory(path):
        empty = not any(path.iterdir())
    if empty:
        # Recovery after the last authenticated child was removed, before rmdir.
        path.rmdir()
        return
    _validate(path, canonical)
    with suppress(FileNotFoundError):
        (path / _TERMINAL).unlink()
    (path / _METADATA).unlink()
    path.rmdir()


@dataclass
class WindowsRunLease:
    """Own one logical run through logging shutdown and terminal retirement."""

    path: Path
    _descriptor: int
    _logs: _windows.PinnedDirectory
    _metadata: dict[str, Any]
    _closed: bool = False

    def close(self) -> None:
        """Release process ownership while preserving recoverable namespace state."""
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._descriptor)
        finally:
            self._logs.close()

    def retire(self) -> None:
        """Retire a validated generation under the non-retiring coordinator."""
        if self._closed:
            return
        try:
            retired = _retired(self.path)
            if os.path.lexists(retired):
                raise LeaseNamespaceRecoveryError(
                    f"A retired lease namespace already exists: {retired}"
                )
            if _validate(self.path, self.path) != self._metadata:
                raise LeaseNamespaceRecoveryError(f"Windows lease generation changed: {self.path}")
            atomic_write_json(self.path / _TERMINAL, self._metadata)
            self.path.rename(retired)
            _reclaim(retired, self.path)
            with suppress(OSError):
                self.path.parent.rmdir()
        finally:
            self.close()


def claim_windows_run_lease(path: Path) -> WindowsRunLease:
    """Claim a run nonblockingly; a killed process releases its handle automatically."""
    canonical = Path(os.path.abspath(path))
    logs = _windows.pin_directory(canonical.parent.parent, create=True)
    try:
        try:
            descriptor = _windows.open_file(
                logs.path / WINDOWS_RUN_LOCK, write=True, create=True, share=3
            )
        except PermissionError as error:
            raise LeaseNamespaceConflictError(
                f"Windows run is active or inaccessible: {canonical}"
            ) from error
    except BaseException:
        logs.close()
        raise
    try:
        try:
            lock_exclusive(descriptor)
        except BlockingIOError as error:
            raise LeaseNamespaceConflictError(f"Windows run is active: {canonical}") from error
        if os.fstat(descriptor).st_size:
            raise LeaseNamespaceRecoveryError("Windows coordinator must be an empty file")
        with _windows.pin_directory(canonical.parent, create=True):
            retired = _retired(canonical)
            if os.path.lexists(retired):
                if os.path.lexists(canonical):
                    raise LeaseNamespaceRecoveryError(
                        f"Active and retired Windows leases coexist: {canonical}"
                    )
                _reclaim(retired, canonical)
            if os.path.lexists(canonical):
                with _windows.pin_directory(canonical):
                    empty = not any(canonical.iterdir())
                if empty:
                    # Recovery after mkdir but before the first metadata publication.
                    canonical.rmdir()
            if os.path.lexists(canonical):
                metadata = _validate(canonical, canonical)
                if (canonical / _TERMINAL).exists():
                    canonical.rename(retired)
                    _reclaim(retired, canonical)
            if not canonical.exists():
                canonical.mkdir()
                metadata = {
                    "schema": "mammoth-windows-run-lease-v1",
                    "path": os.path.normcase(str(canonical)),
                    "generation": uuid.uuid4().hex,
                }
                atomic_write_json(canonical / _METADATA, metadata)
            return WindowsRunLease(canonical, descriptor, logs, metadata)
    except BaseException:
        os.close(descriptor)
        logs.close()
        raise


class RunLease(Protocol):
    """Ownership token independent of the on-disk platform lease format."""

    def close(self) -> None:
        """Release ownership, retaining recoverable state."""
        ...

    def retire(self) -> None:
        """Retire successful execution state and release ownership."""
        ...


def claim_run_lease(path: Path) -> tuple[Path, RunLease]:
    """Select the existing platform lease format and its ownership path."""
    if os.name == "nt":
        lease = claim_windows_run_lease(path)
        return path.parent.parent / WINDOWS_RUN_LOCK, lease
    namespace = claim_lease_namespace(path, owned_parents=(path.parent,))
    return namespace.path / ".mammoth-lease.lock", namespace
