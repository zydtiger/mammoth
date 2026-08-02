"""Atomic local-filesystem publication helpers used by Mammoth producers.

Core metadata and optional trainer checkpoints call these helpers. Payload
meaning and serialization remain the responsibility of the caller.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any


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
