"""Portable descriptor-lock contracts used by filesystem adapters."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mammoth.core._filesystem.locking import lock_exclusive, unlock


@pytest.mark.parametrize("abrupt_exit", [False, True])
def test_descriptor_lock_contention_and_process_exit(tmp_path: Path, abrupt_exit: bool) -> None:
    path = tmp_path / "coordinator.lock"
    path.touch()
    identity = path.stat().st_ino
    script = """
import os, sys
from mammoth.core._filesystem.locking import lock_exclusive, unlock
fd = os.open(sys.argv[1], os.O_RDWR)
lock_exclusive(fd)
print('locked', flush=True)
sys.stdin.readline()
if sys.argv[2] == 'True':
    os._exit(0)
unlock(fd)
os.close(fd)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(abrupt_exit)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    descriptor = os.open(path, os.O_RDWR)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(BlockingIOError):
            lock_exclusive(descriptor)
        _, errors = process.communicate("release\n", timeout=15)
        assert process.returncode == 0, errors
        # Reuse the descriptor opened before the prior owner's exit.
        lock_exclusive(descriptor)
        unlock(descriptor)
        assert path.stat().st_ino == identity
        assert path.read_bytes() == b""
    finally:
        os.close(descriptor)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=15)


def test_blocking_lock_waits_for_existing_owner(tmp_path: Path) -> None:
    path = tmp_path / "sequence.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    script = """
import os, sys
from mammoth.core._filesystem.locking import lock_exclusive, unlock
fd = os.open(sys.argv[1], os.O_RDWR)
print('ready', flush=True)
lock_exclusive(fd, blocking=True)
print('acquired', flush=True)
unlock(fd)
os.close(fd)
"""
    lock_exclusive(descriptor)
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            process.communicate(timeout=0.2)
        unlock(descriptor)
        output, errors = process.communicate(timeout=15)
        assert process.returncode == 0, errors
        assert output.strip() == "acquired"
    finally:
        os.close(descriptor)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=15)


@pytest.mark.skipif(os.name == "nt", reason="POSIX errno recovery contract")
def test_invalid_descriptor_preserves_os_error(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path / "closed.lock", os.O_RDWR | os.O_CREAT, 0o600)
    os.close(descriptor)
    with pytest.raises(OSError) as caught:
        lock_exclusive(descriptor)
    assert caught.value.errno == errno.EBADF
