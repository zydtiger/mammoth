from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

import pytest

from mammoth.core import (
    ArtifactChangedError,
    ArtifactReceipt,
    ArtifactVerificationError,
    artifacts,
    atomic_publish,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    discard_prepared_artifact,
    inspect_artifact,
    prepare_artifact,
    publish_prepared_artifact,
    verify_artifact,
)


@pytest.mark.parametrize(
    ("path", "size_bytes", "sha256", "error"),
    [
        ("not-a-path", 0, "0" * 64, TypeError),
        (Path("artifact.bin"), -1, "0" * 64, ValueError),
        (Path("artifact.bin"), 0, "A" * 64, ValueError),
        (Path("artifact.bin"), 0, "0" * 63, ValueError),
    ],
)
def test_artifact_receipt_validates_canonical_fields(
    path: object,
    size_bytes: int,
    sha256: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ArtifactReceipt(path=path, size_bytes=size_bytes, sha256=sha256)  # type: ignore[arg-type]


def test_inspect_artifact_records_empty_small_and_multi_chunk_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    small = tmp_path / "small.bin"
    multi_chunk = tmp_path / "multi.bin"
    empty.write_bytes(b"")
    small.write_bytes(b"small")
    multi_chunk.write_bytes(b"abcdefgh")

    assert inspect_artifact(empty, chunk_size=3) == ArtifactReceipt(
        path=empty,
        size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    assert inspect_artifact(small, chunk_size=3).size_bytes == 5
    assert inspect_artifact(multi_chunk, chunk_size=3).sha256 == (
        "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab"
    )


def test_inspect_artifact_rejects_missing_directory_special_file_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_artifact(tmp_path / "missing.bin")
    with pytest.raises(ValueError, match="regular file"):
        inspect_artifact(tmp_path)

    fifo = tmp_path / "stream"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        inspect_artifact(fifo)

    target = tmp_path / "target.bin"
    link = tmp_path / "target-link.bin"
    target.write_bytes(b"target")
    link.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        inspect_artifact(link)


def test_inspect_artifact_rejects_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first-second")
    original_read = artifacts.os.read
    mutated = False

    def read_and_mutate(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            artifact.write_bytes(b"changed-data")
        return chunk

    monkeypatch.setattr(artifacts.os, "read", read_and_mutate)

    with pytest.raises(ArtifactChangedError, match="changed while"):
        inspect_artifact(artifact, chunk_size=5)


def test_inspect_artifact_rejects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    replacement = tmp_path / "replacement.bin"
    artifact.write_bytes(b"first-second")
    replacement.write_bytes(b"replacement!")
    original_read = artifacts.os.read
    replaced = False

    def read_and_replace(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, artifact)
        return chunk

    monkeypatch.setattr(artifacts.os, "read", read_and_replace)

    with pytest.raises(ArtifactChangedError, match="changed while"):
        inspect_artifact(artifact, chunk_size=5)


def test_inspect_artifact_rejects_symlink_replacement_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    target = tmp_path / "target.bin"
    replacement = tmp_path / "replacement-link.bin"
    artifact.write_bytes(b"artifact")
    target.write_bytes(b"target")
    replacement.symlink_to(target)
    original_open = artifacts.os.open
    replaced = False

    def replace_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if Path(path) == artifact and not replaced:
            replaced = True
            os.replace(replacement, artifact)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", replace_then_open)

    with pytest.raises(ArtifactChangedError, match="changed before"):
        inspect_artifact(artifact)


@pytest.mark.parametrize("replacement_kind", ("directory", "fifo"))
def test_inspect_artifact_rejects_non_regular_replacement_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    artifact = tmp_path / "artifact.bin"
    replacement = tmp_path / "replacement"
    artifact.write_bytes(b"artifact")
    if replacement_kind == "directory":
        replacement.mkdir()
    else:
        os.mkfifo(replacement)
    original_open = artifacts.os.open
    replaced = False

    def replace_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if Path(path) == artifact and not replaced:
            replaced = True
            if replacement_kind == "directory":
                artifact.unlink()
            os.replace(replacement, artifact)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", replace_then_open)

    with pytest.raises(ArtifactChangedError, match="changed before"):
        inspect_artifact(artifact)


def test_inspect_artifact_rejects_deletion_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    original_open = artifacts.os.open
    deleted = False

    def delete_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal deleted
        if Path(path) == artifact and not deleted:
            deleted = True
            artifact.unlink()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", delete_then_open)

    with pytest.raises(ArtifactChangedError, match="changed before"):
        inspect_artifact(artifact)


def test_inspect_artifact_treats_no_follow_loop_as_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")

    def reject_with_no_follow_loop(
        _path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        _flags: int,
        _mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        del dir_fd
        raise OSError(errno.ELOOP, "too many levels of symbolic links")

    monkeypatch.setattr(artifacts.os, "open", reject_with_no_follow_loop)

    with pytest.raises(ArtifactChangedError, match="changed before"):
        inspect_artifact(artifact)


def test_verify_artifact_requires_current_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    receipt = inspect_artifact(artifact)

    assert verify_artifact(receipt) is None

    for replacement in (b"short", b"original-more", b"replaced"):
        artifact.write_bytes(replacement)
        with pytest.raises(ArtifactVerificationError, match="does not match"):
            verify_artifact(receipt)

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"new-bytes")
    os.replace(replacement, artifact)
    with pytest.raises(ArtifactVerificationError, match="does not match"):
        verify_artifact(receipt)

    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        verify_artifact(receipt)


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


def test_prepared_artifact_cleans_directory_output_without_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.bin"
    destination.write_bytes(b"old")
    original_unlink = os.unlink

    def reject_directory_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        if path == destination.name:
            raise PermissionError(errno.EPERM, "directory unlink is not permitted")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", reject_directory_unlink)

    def write_directory(temporary: Path) -> None:
        temporary.mkdir()

    with pytest.raises(FileNotFoundError, match="did not create a file"):
        prepare_artifact(destination, write_directory)

    assert destination.read_bytes() == b"old"
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


def test_prepared_artifact_can_retain_serializer_mode_for_new_files(tmp_path: Path) -> None:
    created = tmp_path / "created.bin"

    def write_created(temporary: Path) -> None:
        temporary.write_bytes(b"created")
        temporary.chmod(0o640)

    prepared_created = prepare_artifact(created, write_created, mode=None)
    publish_prepared_artifact(prepared_created)

    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"old")
    existing.chmod(0o400)

    def write_existing(temporary: Path) -> None:
        temporary.write_bytes(b"new")
        temporary.chmod(0o640)

    prepared_existing = prepare_artifact(existing, write_existing, mode=None)
    publish_prepared_artifact(prepared_existing)

    assert stat.S_IMODE(created.stat().st_mode) == 0o640
    assert stat.S_IMODE(existing.stat().st_mode) == 0o400


def test_prepared_artifact_supports_create_without_truncation(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"

    def write_created(temporary: Path) -> None:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, b"checkpoint")
        finally:
            os.close(descriptor)

    prepared = prepare_artifact(destination, write_created, mode=None)
    try:
        assert prepared.temporary.read_bytes() == b"checkpoint"
        assert stat.S_IMODE(prepared.temporary.stat().st_mode) == 0o600
    finally:
        discard_prepared_artifact(prepared)


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


def test_prepared_artifact_cleans_private_staging_when_output_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open

    def fail_output_open(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        if path == "checkpoint.bin" and flags & os.O_RDONLY == os.O_RDONLY:
            raise OSError("output open failed")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_output_open)

    with pytest.raises(OSError, match="output open failed"):
        prepare_artifact(
            tmp_path / "checkpoint.bin",
            lambda temporary: temporary.write_bytes(b"checkpoint"),
        )

    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_artifact_uses_private_staging_directory(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.bin"

    def write(temporary: Path) -> None:
        assert temporary.parent != destination.parent
        assert stat.S_IMODE(temporary.parent.stat().st_mode) == 0o700
        temporary.write_bytes(b"checkpoint")

    prepared = prepare_artifact(destination, write)
    publish_prepared_artifact(prepared)

    assert destination.read_bytes() == b"checkpoint"
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


def test_prepared_artifact_anchors_writer_during_transient_parent_replacement(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "checkpoints" / "nested"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-nested"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "checkpoint.bin"
    destination.write_bytes(b"old")

    def replace_restore_and_write(temporary: Path) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)
        temporary.write_bytes(b"secret-checkpoint")
        parent.unlink()
        moved_parent.rename(parent)

    prepared = prepare_artifact(destination, replace_restore_and_write)
    publish_prepared_artifact(prepared)

    assert destination.read_bytes() == b"secret-checkpoint"
    assert not list(outside.iterdir())
    assert not list(parent.glob(".*.tmp"))


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


def test_prepared_artifact_propagates_type_error_from_keyword_replace_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.bin"
    prepared = prepare_artifact(
        destination,
        lambda temporary: temporary.write_bytes(b"checkpoint"),
    )

    def fail_replace(source: str, target: str, **kwargs: object) -> None:
        del source, target, kwargs
        raise TypeError("replace adapter failed internally")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(TypeError, match="failed internally"):
        publish_prepared_artifact(prepared)
    discard_prepared_artifact(prepared)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))
