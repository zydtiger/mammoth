from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from mammoth.core import (
    atomic_publish,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    discard_prepared_artifact,
    prepare_artifact,
    publish_prepared_artifact,
)


def test_atomic_writers_publish_complete_payloads(tmp_path: Path) -> None:
    binary = atomic_write_bytes(tmp_path / "nested" / "payload.bin", b"abc")
    text = atomic_write_text(tmp_path / "payload.txt", "hello")
    document = atomic_write_json(tmp_path / "payload.json", {"b": 2, "a": 1})

    assert binary.read_bytes() == b"abc"
    assert text.read_text() == "hello"
    assert json.loads(document.read_text()) == {"a": 1, "b": 2}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_publish_keeps_previous_file_when_writer_fails(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"old")

    def fail_writer(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_publish(target, fail_writer)

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publish_requires_writer_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="did not create"):
        atomic_publish(tmp_path / "missing.bin", lambda _path: None)


def test_prepared_artifact_preserves_replacement_mode_until_publication(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")
    destination.chmod(0o640)

    def write_new(temporary: Path) -> None:
        temporary.write_bytes(b"new")

    prepared = prepare_artifact(
        destination,
        write_new,
        mode=0o666,
    )

    assert destination.read_bytes() == b"old"
    assert prepared.temporary.read_bytes() == b"new"
    assert stat.S_IMODE(prepared.temporary.stat().st_mode) == 0o640
    assert publish_prepared_artifact(prepared) == destination
    assert destination.read_bytes() == b"new"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_discard_prepared_artifact_preserves_destination(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")

    def write_new(temporary: Path) -> None:
        temporary.write_bytes(b"new")

    prepared = prepare_artifact(destination, write_new)

    discard_prepared_artifact(prepared)

    assert destination.read_bytes() == b"old"
    assert not prepared.temporary.exists()


def test_prepared_artifact_requires_writer_output(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")

    with pytest.raises(FileNotFoundError):
        prepare_artifact(destination, lambda _temporary: None)

    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_artifact_rejects_fifo_output_without_blocking(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"

    def write_fifo(temporary: Path) -> None:
        os.mkfifo(temporary)

    with pytest.raises(FileNotFoundError, match="did not create a file"):
        prepare_artifact(destination, write_fifo)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_artifact_reapplies_mode_after_writer_replaces_temporary(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")
    destination.chmod(0o640)

    def replace_temporary(temporary: Path) -> None:
        replacement = temporary.with_suffix(".replacement")
        replacement.write_bytes(b"new")
        replacement.chmod(0o666)
        os.replace(replacement, temporary)

    prepared = prepare_artifact(destination, replace_temporary)

    assert stat.S_IMODE(prepared.temporary.stat().st_mode) == 0o640
    publish_prepared_artifact(prepared)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_prepared_artifact_applies_read_only_mode_after_serialization(tmp_path: Path) -> None:
    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"old")
    existing.chmod(0o444)

    def write_existing(temporary: Path) -> None:
        temporary.write_bytes(b"new")

    prepared_existing = prepare_artifact(existing, write_existing)
    publish_prepared_artifact(prepared_existing)

    created = tmp_path / "created.bin"

    def write_created(temporary: Path) -> None:
        temporary.write_bytes(b"created")

    prepared_created = prepare_artifact(
        created,
        write_created,
        mode=0o400,
        preserve_permissions=False,
    )
    publish_prepared_artifact(prepared_created)

    assert existing.read_bytes() == b"new"
    assert stat.S_IMODE(existing.stat().st_mode) == 0o444
    assert created.read_bytes() == b"created"
    assert stat.S_IMODE(created.stat().st_mode) == 0o400


def test_prepared_artifact_cleans_temporary_when_permission_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")

    def fail_fchmod(file_descriptor: int, mode: int) -> None:
        del file_descriptor, mode
        raise OSError("permission setup failed")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)

    def write_new(temporary: Path) -> None:
        temporary.write_bytes(b"new")

    with pytest.raises(OSError, match="permission setup failed"):
        prepare_artifact(destination, write_new)

    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_artifact_cleans_temporary_when_initial_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    close_calls = 0

    def fail_first_close(file_descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("initial close failed")
        real_close(file_descriptor)

    monkeypatch.setattr(os, "close", fail_first_close)

    with pytest.raises(OSError, match="initial close failed"):
        prepare_artifact(tmp_path / "checkpoint.bin", lambda _temporary: None)

    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_artifact_rejects_parent_replacement_during_serialization(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "checkpoints" / "nested"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-nested"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "checkpoint.bin"

    def replace_parent_and_write(temporary: Path) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)
        temporary.write_bytes(b"outside")

    with pytest.raises(RuntimeError, match="parent changed during serialization"):
        prepare_artifact(destination, replace_parent_and_write)

    assert not (outside / destination.name).exists()
    assert not list(outside.glob(".*.tmp"))
    assert not list(moved_parent.glob(".*.tmp"))


def test_prepared_artifact_rejects_parent_replacement_before_publication(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "checkpoints" / "nested"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-nested"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "checkpoint.bin"

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"checkpoint")

    prepared = prepare_artifact(destination, write)
    parent.rename(moved_parent)
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="parent changed before publication"):
        publish_prepared_artifact(prepared)
    discard_prepared_artifact(prepared)

    assert not (outside / destination.name).exists()
    assert not list(outside.glob(".*.tmp"))
    assert not list(moved_parent.glob(".*.tmp"))


def test_prepared_artifact_supports_two_argument_replace_failure_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.bin"

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"checkpoint")

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace adapter failed")

    prepared = prepare_artifact(destination, write)
    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace adapter failed"):
        publish_prepared_artifact(prepared)
    discard_prepared_artifact(prepared)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))
