"""Focused coverage for the recoverable chunked work-store primitive."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import mammoth.core.workstore as workstore_module
from mammoth.core import (
    WorkStoreConflictError,
    WorkStoreDamagedError,
    WorkStoreIncompatibleError,
    WorkStoreSession,
    WorkStoreValidationError,
    claim_work_store_lease,
    cleanup_work_store,
    inspect_work_store,
)


def _identity(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "consumer": "widget-exporter",
        "source_sha256": "a" * 64,
        "chunk_count": 4,
    }
    payload.update(overrides)
    return payload


class _FailingAppendHandle:
    """Wrap a real journal file handle to inject one controlled I/O failure.

    ``fail_at="write"`` raises before any bytes reach the file, matching an
    outright write failure. ``fail_at="flush"`` first delivers the bytes
    directly to the underlying descriptor (bypassing Python's own
    buffering, so they are visible to a later raw read exactly as a
    deferred ENOSPC/EIO surfacing only at flush would leave them) and then
    raises, reproducing "bytes already in the page cache before the
    failure is observed".
    """

    def __init__(self, real_handle: object, *, fail_at: str) -> None:
        self._real = real_handle
        self._fail_at = fail_at

    def fileno(self) -> int:
        return self._real.fileno()  # type: ignore[attr-defined]

    def write(self, data: bytes) -> int:
        if self._fail_at == "write":
            raise OSError("simulated write failure")
        return os.write(self.fileno(), data)

    def flush(self) -> None:
        if self._fail_at == "flush":
            raise OSError("simulated flush failure")

    def close(self) -> None:
        self._real.close()  # type: ignore[attr-defined]

    def __enter__(self) -> _FailingAppendHandle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _patch_journal_append_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str,
) -> None:
    """Make exactly the next ``os.fdopen`` call return a failure-injecting wrapper.

    One-shot and self-disarming: a later ``os.fdopen`` call (for example a
    subsequent reopen's journal *read*) goes through the real function
    unaffected, without requiring an explicit ``monkeypatch.undo()``.
    """
    real_fdopen = workstore_module.os.fdopen
    armed = True

    def patched_fdopen(descriptor: int, *args: object, **kwargs: object) -> object:
        nonlocal armed
        if not armed:
            return real_fdopen(descriptor, *args, **kwargs)
        armed = False
        real_handle = real_fdopen(descriptor, *args, **kwargs)
        return _FailingAppendHandle(real_handle, fail_at=fail_at)

    monkeypatch.setattr(workstore_module.os, "fdopen", patched_fdopen)


class _CustomSequence(Sequence[str]):
    """A ``Sequence`` implementation that is neither ``list`` nor ``tuple``."""

    def __init__(self, items: list[str]) -> None:
        self._items = list(items)

    def __getitem__(self, index: int) -> str:  # type: ignore[override]
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


class _DuplicateKeyMapping(Mapping[str, str]):
    """A genuine ``Mapping`` whose ``items()`` yields a duplicate key.

    A plain ``dict`` cannot represent duplicate keys, so this is the only
    way to exercise ``_validate_chunk_markers``'s defensive duplicate-ID
    check with a real Mapping-shaped input.
    """

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self._pairs = pairs

    def __getitem__(self, key: str) -> str:
        for pair_key, value in self._pairs:
            if pair_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(key for key, _value in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def items(self) -> list[tuple[str, str]]:  # type: ignore[override]
        return self._pairs


def test_open_or_create_creates_durable_store_with_owner_only_permissions(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        assert session.recovered is False
        assert session.completed_chunk_ids == ()
        store_mode = stat.S_IMODE(store_path.lstat().st_mode)
        assert store_mode == 0o700
        metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
        metadata_mode = stat.S_IMODE(metadata_path.lstat().st_mode)
        assert metadata_mode == 0o600
        payload = json.loads(metadata_path.read_bytes())
        assert payload["format"] == workstore_module.WORK_STORE_FORMAT
        assert payload["schema_version"] == workstore_module.WORK_STORE_SCHEMA_VERSION


def test_create_store_fsyncs_the_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new store directory's own dentry in its parent must be fsynced.

    Without this, the new directory's existence is not durable, even
    though everything written inside it is.
    """
    store_path = tmp_path / "nested" / "store"
    fsynced_paths: list[Path] = []
    original_fsync_directory = workstore_module._fsync_directory

    def spying_fsync_directory(path: Path) -> None:
        fsynced_paths.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(workstore_module, "_fsync_directory", spying_fsync_directory)
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass

    assert store_path.parent in fsynced_paths


def test_open_or_create_second_claim_is_concurrently_owned(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as first:
        assert first.recovered is False
        with pytest.raises(WorkStoreConflictError, match="owned by another process"):
            WorkStoreSession.open_or_create(store_path, _identity())
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass


def test_inspect_work_store_reports_concurrently_owned_without_reading(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})
        inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "concurrently_owned"
    assert inspection.completed_chunk_ids == ()


def test_inspect_work_store_reports_absent_for_missing_path(tmp_path: Path) -> None:
    inspection = inspect_work_store(tmp_path / "never-created", _identity())
    assert inspection.status == "absent"
    assert inspection.completed_chunk_ids == ()


def test_commit_then_close_then_resume_reports_resumable_with_completed_chunks(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0", "chunk-1": "marker-1"})
        session.commit({"chunk-2": "marker-2"})
        assert session.completed_chunk_ids == ("chunk-0", "chunk-1", "chunk-2")

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "resumable"
    assert inspection.completed_chunk_ids == ("chunk-0", "chunk-1", "chunk-2")

    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        assert session.recovered is True
        assert session.completed_chunk_ids == ("chunk-0", "chunk-1", "chunk-2")
        session.commit({"chunk-3": "marker-3"})
        assert session.completed_chunk_ids == (
            "chunk-0",
            "chunk-1",
            "chunk-2",
            "chunk-3",
        )


def test_resume_after_interrupted_commit_truncates_torn_tail_and_accepts_new_commits(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    valid_bytes = journal_path.stat().st_size
    with journal_path.open("ab") as handle:
        # Simulate a process killed mid-append: a syntactically incomplete
        # record with no terminating newline.
        handle.write(b'{"sequence": 1, "previous_sha256": "no-newline-tail"')
    assert journal_path.stat().st_size > valid_bytes

    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        assert session.recovered is True
        assert session.completed_chunk_ids == ("chunk-0",)
        # The torn tail must have been discarded before further appends.
        assert journal_path.stat().st_size == valid_bytes
        session.commit({"chunk-1": "marker-1"})
        assert session.completed_chunk_ids == ("chunk-0", "chunk-1")

    # The journal must now be fully well-formed JSONL with an intact chain.
    lines = journal_path.read_bytes().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert isinstance(record["sha256"], str)


def test_commit_rejects_chunk_ids_already_completed(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})
        with pytest.raises(WorkStoreValidationError, match="already be completed"):
            session.commit({"chunk-0": "marker-0-again"})
        # A rejected commit must not have appended a partial or duplicate record.
        assert session.completed_chunk_ids == ("chunk-0",)
        journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
        assert len(journal_path.read_bytes().splitlines()) == 1


def test_commit_requires_nonempty_mapping_and_valid_markers(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        with pytest.raises(WorkStoreValidationError, match="non-empty mapping"):
            session.commit({})
        with pytest.raises(WorkStoreValidationError, match="chunk ID"):
            session.commit({"": "marker"})
        with pytest.raises(WorkStoreValidationError, match="chunk marker"):
            session.commit({"chunk-0": ""})
        assert session.completed_chunk_ids == ()


def test_validate_chunk_markers_rejects_duplicate_ids_within_one_call() -> None:
    """Duplicate chunk IDs in one commit batch must be rejected.

    A plain ``dict`` cannot represent a literal duplicate key, so this
    directly exercises the private validator with a Mapping-shaped input
    whose ``items()`` legitimately yields one.
    """
    duplicated = _DuplicateKeyMapping([("chunk-0", "marker-a"), ("chunk-0", "marker-b")])
    with pytest.raises(WorkStoreValidationError, match="unique"):
        workstore_module._validate_chunk_markers(duplicated)


def test_commit_rejects_a_closed_session(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    session.close()
    with pytest.raises(WorkStoreValidationError, match="closed"):
        session.commit({"chunk-0": "marker-0"})
    with pytest.raises(WorkStoreValidationError, match="closed"):
        session.verify()


def test_commit_poisons_the_session_and_never_adopts_an_unsynced_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit whose fsync fails must not be resurrected by a later reopen.

    write()+flush() already place the fully formed record where a naive
    reader could see it; the fix must truncate that write back off, poison
    the session against further commit()/verify(), and keep a subsequent
    open_or_create from adopting the never-synced chunk.
    """
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())

    def failing_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(workstore_module.os, "fsync", failing_fsync)
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.commit({"chunk-0": "marker-0"})

    assert session.completed_chunk_ids == ()
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.commit({"chunk-1": "marker-1"})
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.verify()

    session.close()

    with WorkStoreSession.open_or_create(store_path, _identity()) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ()


def test_commit_poisons_and_does_not_adopt_a_record_on_a_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    _patch_journal_append_failure(monkeypatch, fail_at="write")

    with pytest.raises(WorkStoreDamagedError, match="poisoned") as excinfo:
        session.commit({"chunk-0": "marker-0"})
    assert isinstance(excinfo.value, workstore_module.WorkStoreError)

    assert session.completed_chunk_ids == ()
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.commit({"chunk-1": "marker-1"})
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.verify()
    session.close()

    with WorkStoreSession.open_or_create(store_path, _identity()) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ()


def test_commit_poisons_and_does_not_adopt_a_record_visible_before_a_flush_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flush failure after bytes are already visible must still roll back.

    Unlike the write-failure case, the record's bytes are delivered to the
    underlying descriptor before ``flush()`` raises, mirroring a deferred
    ENOSPC/EIO. The already-visible bytes must be truncated back off, not
    just future writes rejected.
    """
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    pre_commit_size = journal_path.stat().st_size

    _patch_journal_append_failure(monkeypatch, fail_at="flush")
    with pytest.raises(WorkStoreDamagedError, match="poisoned") as excinfo:
        session.commit({"chunk-0": "marker-0"})
    assert isinstance(excinfo.value, workstore_module.WorkStoreError)

    # The record's bytes must have been truncated back off the journal.
    assert journal_path.stat().st_size == pre_commit_size
    assert session.completed_chunk_ids == ()
    session.close()

    with WorkStoreSession.open_or_create(store_path, _identity()) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ()


def test_reopen_after_a_poisoned_write_failure_can_retry_without_damage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-chunk retry after reopening must not corrupt the journal.

    Before the fix, a retry after an unguarded write/flush failure could
    append a second, competing sequence-0 record and permanently damage the
    whole store on the next open. After the fix, the rollback leaves a
    clean journal that a fresh session can extend normally.
    """
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    _patch_journal_append_failure(monkeypatch, fail_at="write")
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.commit({"chunk-0": "marker-0"})
    session.close()

    with WorkStoreSession.open_or_create(store_path, _identity()) as retried:
        assert retried.recovered is True
        assert retried.completed_chunk_ids == ()
        retried.commit({"chunk-0": "marker-0"})
        assert retried.completed_chunk_ids == ("chunk-0",)

    final = inspect_work_store(store_path, _identity())
    assert final.status == "resumable"
    assert final.completed_chunk_ids == ("chunk-0",)


def test_commit_poisons_but_keeps_a_durable_record_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory-fsync failure after a durable record must not truncate.

    The record's own fsync already succeeded, so the completed chunk is
    genuinely durable; only the (load-bearing only for a brand-new journal
    file) trailing directory fsync failed. The session must still be
    poisoned and raise a WorkStoreError, but nothing is rolled back, and a
    fresh reopen must see the chunk as completed.
    """
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())

    def failing_fsync_directory(path: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(workstore_module, "_fsync_directory", failing_fsync_directory)
    with pytest.raises(WorkStoreDamagedError, match="poisoned") as excinfo:
        session.commit({"chunk-0": "marker-0"})
    assert isinstance(excinfo.value, workstore_module.WorkStoreError)

    # The record's own fsync already succeeded before the directory fsync
    # failed, so the session's cached state legitimately reflects it.
    assert session.completed_chunk_ids == ("chunk-0",)
    with pytest.raises(WorkStoreDamagedError, match="poisoned"):
        session.verify()
    session.close()

    with WorkStoreSession.open_or_create(store_path, _identity()) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ("chunk-0",)


def test_verify_detects_a_tampered_journal_hash_chain(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})
        session.verify()

        journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
        record = json.loads(journal_path.read_bytes())
        record["sha256"] = "0" * 64
        journal_path.write_bytes((json.dumps(record) + "\n").encode("utf-8"))

        with pytest.raises(WorkStoreDamagedError, match="integrity"):
            session.verify()


def test_open_or_create_detects_a_tampered_journal_on_resume(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    record = json.loads(journal_path.read_bytes())
    record["chunks"][0]["marker"] = "tampered-marker"
    journal_path.write_bytes((json.dumps(record) + "\n").encode("utf-8"))

    with pytest.raises(WorkStoreDamagedError, match="integrity"):
        WorkStoreSession.open_or_create(store_path, _identity())

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"

    # The failed open must not have left the store leased.
    with claim_work_store_lease(store_path):
        pass


def test_open_or_create_rejects_a_correctly_hashed_chunk_id_reuse(tmp_path: Path) -> None:
    """A forged record with a *valid* hash chain must still be rejected.

    This is the sharper tamper-detection case: recomputing the hash chain
    correctly is not enough to resurrect an already-completed chunk ID, so
    integrity depends on more than the SHA-256 chain alone.
    """
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    first_record = json.loads(journal_path.read_bytes())
    forged_payload = workstore_module._journal_record_payload(
        1, first_record["sha256"], (("chunk-0", "marker-again"),)
    )
    forged_record = {**forged_payload, "sha256": workstore_module._sha256_payload(forged_payload)}
    with journal_path.open("ab") as handle:
        handle.write((json.dumps(forged_record) + "\n").encode("utf-8"))

    with pytest.raises(WorkStoreDamagedError, match="repeats a chunk ID"):
        WorkStoreSession.open_or_create(store_path, _identity())


@pytest.mark.parametrize(
    "corrupt_line",
    [
        pytest.param(b"not even json\n", id="not-json"),
        pytest.param(b'"just a json string"\n', id="not-an-object"),
        pytest.param(
            b'{"sequence": 0, "previous_sha256": "' + b"0" * 64 + b'", '
            b'"chunks": "not-a-list", "sha256": "x"}\n',
            id="chunks-not-a-list",
        ),
        pytest.param(
            b'{"sequence": 0, "previous_sha256": "' + b"0" * 64 + b'", '
            b'"chunks": [], "sha256": "x"}\n',
            id="chunks-empty",
        ),
        pytest.param(
            b'{"sequence": 0, "previous_sha256": "' + b"0" * 64 + b'", '
            b'"chunks": ["not-a-mapping"], "sha256": "x"}\n',
            id="chunk-not-a-mapping",
        ),
        pytest.param(
            b'{"sequence": 0, "previous_sha256": "' + b"0" * 64 + b'", '
            b'"chunks": [{"chunk_id": "", "marker": "m"}], "sha256": "x"}\n',
            id="chunk-id-empty",
        ),
        pytest.param(
            b'{"sequence": 5, "previous_sha256": "' + b"0" * 64 + b'", '
            b'"chunks": [{"chunk_id": "c", "marker": "m"}], "sha256": "x"}\n',
            id="sequence-mismatch",
        ),
        pytest.param(
            b'{"sequence": 0, "previous_sha256": "not-empty-chain-root", '
            b'"chunks": [{"chunk_id": "c", "marker": "m"}], "sha256": "x"}\n',
            id="previous-sha256-mismatch",
        ),
    ],
)
def test_inspect_and_open_classify_structurally_malformed_journal_lines(
    tmp_path: Path,
    corrupt_line: bytes,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass
    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    journal_path.write_bytes(corrupt_line)

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_inspect_and_open_classify_a_legacy_store_without_mammoth_metadata(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    store_path.mkdir(mode=0o700)
    (store_path / "some-consumer-file.bin").write_bytes(b"pre-existing consumer content")

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "legacy"

    with pytest.raises(WorkStoreIncompatibleError) as excinfo:
        WorkStoreSession.open_or_create(store_path, _identity())
    assert excinfo.value.status == "legacy"
    # A rejected legacy path must be preserved untouched.
    assert (store_path / "some-consumer-file.bin").exists()


def test_inspect_and_open_classify_an_incompatible_identity(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity(consumer="widget-exporter")):
        pass

    other_identity = _identity(consumer="a-different-consumer")
    inspection = inspect_work_store(store_path, other_identity)
    assert inspection.status == "incompatible"

    with pytest.raises(WorkStoreIncompatibleError) as excinfo:
        WorkStoreSession.open_or_create(store_path, other_identity)
    assert excinfo.value.status == "incompatible"


def test_inspect_and_open_classify_a_copied_store_as_relocated(tmp_path: Path) -> None:
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    with WorkStoreSession.open_or_create(store_a, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    # Same identity, same bytes, but copied to a path the metadata was never
    # durably bound to: this must never be silently adopted as a resume of
    # store_a's completed work at a location store_a's outputs never used.
    shutil.copytree(store_a, store_b)

    inspection = inspect_work_store(store_b, _identity())
    assert inspection.status == "relocated"
    assert inspection.completed_chunk_ids == ()

    with pytest.raises(WorkStoreIncompatibleError) as excinfo:
        WorkStoreSession.open_or_create(store_b, _identity())
    assert excinfo.value.status == "relocated"
    # The copy must be preserved untouched, exactly like a legacy store.
    assert store_b.exists()

    # The original path is unaffected and remains genuinely resumable.
    original_inspection = inspect_work_store(store_a, _identity())
    assert original_inspection.status == "resumable"
    assert original_inspection.completed_chunk_ids == ("chunk-0",)
    with WorkStoreSession.open_or_create(store_a, _identity()) as session:
        assert session.recovered is True
        assert session.completed_chunk_ids == ("chunk-0",)


def test_open_or_create_rejects_a_journal_grafted_from_a_different_store(
    tmp_path: Path,
) -> None:
    """A journal copied between two same-identity stores must not graft.

    The chain is seeded from (format, identity, store's own recorded path),
    so a journal that validates perfectly at its own store fails on its very
    first record when read under a different store's seed, even though the
    identity and every byte of the journal are otherwise untouched.
    """
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    with WorkStoreSession.open_or_create(store_a, _identity()) as session:
        session.commit({"chunk-from-a": "marker-a"})
    with WorkStoreSession.open_or_create(store_b, _identity()) as session:
        session.commit({"chunk-from-b": "marker-b"})

    journal_a = store_a / workstore_module.WORK_STORE_JOURNAL_NAME
    journal_b = store_b / workstore_module.WORK_STORE_JOURNAL_NAME
    shutil.copyfile(journal_a, journal_b)

    inspection = inspect_work_store(store_b, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_b, _identity())

    # store_a's own genuine journal remains fully resumable.
    original = inspect_work_store(store_a, _identity())
    assert original.status == "resumable"
    assert original.completed_chunk_ids == ("chunk-from-a",)


def test_open_or_create_rejects_a_store_path_edit_in_metadata(tmp_path: Path) -> None:
    """Editing metadata's plaintext store_path field alone must not defeat relocation.

    The field itself is unauthenticated, but the journal chain was seeded
    from the *original* store_path when the store was created; editing only
    the metadata field desynchronizes the chain seed recomputed at load time
    from what is actually baked into the untouched journal, so the edit
    degrades detection to damaged instead of silently curing relocation.
    """
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    with WorkStoreSession.open_or_create(store_a, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    shutil.copytree(store_a, store_b)
    metadata_path = store_b / workstore_module.WORK_STORE_METADATA_NAME
    payload = json.loads(metadata_path.read_bytes())
    payload["store_path"] = str(store_b.resolve())
    metadata_path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))

    inspection = inspect_work_store(store_b, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_b, _identity())


def test_inspect_and_open_classify_damaged_metadata_json(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass
    metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
    metadata_path.write_bytes(b"{not valid json")

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_inspect_and_open_classify_damaged_unsupported_schema_version(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass
    metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
    payload = json.loads(metadata_path.read_bytes())
    payload["schema_version"] = 999
    metadata_path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"


def _merge_metadata_corruption(payload: dict[str, object], corrupt: dict[str, object]) -> None:
    """Shallow-merge a corruption override, merging nested ``identity_kdf`` in place."""
    for key, value in corrupt.items():
        existing = payload.get(key)
        if key == "identity_kdf" and isinstance(value, dict) and isinstance(existing, dict):
            existing.update(value)
        else:
            payload[key] = value


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param({"format": "not-the-right-format"}, id="format-invalid"),
        pytest.param({"format": None}, id="format-missing"),
        pytest.param({"created_at": None}, id="created-at-missing"),
        pytest.param({"created_at": ""}, id="created-at-empty"),
        pytest.param({"store_path": None}, id="store-path-missing"),
        pytest.param({"store_path": ""}, id="store-path-empty"),
        pytest.param({"identity_kdf": {"n": 2**20}}, id="identity-kdf-n-out-of-range"),
        pytest.param({"identity_kdf": {"salt": "abc"}}, id="identity-kdf-salt-odd-length"),
        pytest.param({"identity_kdf": {"salt": "aa" * 8}}, id="identity-kdf-salt-wrong-length"),
        pytest.param({"identity_kdf": {"salt": "zz" * 16}}, id="identity-kdf-salt-not-hex"),
    ],
)
def test_inspect_and_open_classify_damaged_metadata_field_corruption(
    tmp_path: Path,
    corrupt: dict[str, object],
) -> None:
    """A single corrupted field must classify damaged, never crash with a raw error.

    In particular, an out-of-range scrypt cost parameter (which could
    otherwise exhaust memory) or a malformed hex salt (which would
    otherwise crash ``bytes.fromhex``) must both be rejected in
    ``_read_metadata`` before ever reaching the KDF.
    """
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass
    metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
    payload = json.loads(metadata_path.read_bytes())
    _merge_metadata_corruption(payload, corrupt)
    metadata_path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_inspect_and_open_classify_damaged_when_store_path_is_not_a_directory(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    store_path.write_bytes(b"not a directory")

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_inspect_and_open_classify_damaged_when_journal_file_is_missing(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()):
        pass

    # A validly created store's journal always exists (possibly empty).
    # Deleting only the journal file must never be mistaken for a fresh
    # store with zero completed chunks.
    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    journal_path.unlink()

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError, match="journal is missing"):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_remove_after_publication_deletes_store_and_releases_lease_for_reuse(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    session.commit({"chunk-0": "marker-0"})
    session.remove_after_publication()

    assert not store_path.exists()
    # Cleanup must release ownership before returning, and in that order:
    # a fresh claim on the same path succeeds immediately afterward.
    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "absent"
    with claim_work_store_lease(store_path):
        pass


def test_remove_after_publication_quarantines_before_removing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store must be durably renamed aside before its final removal.

    This proves the crash-safety property directly: a rename to a
    deterministic retired sibling, a parent-directory fsync, and only then
    the actual removal — so a crash right after the rename never leaves a
    half-deleted store at the live path.
    """
    store_path = tmp_path / "store"
    session = WorkStoreSession.open_or_create(store_path, _identity())
    session.commit({"chunk-0": "marker-0"})

    retired_path = workstore_module._retired_store_path(store_path)
    observed: list[Path] = []
    original_rmtree = shutil.rmtree

    def spying_rmtree(path: object, *args: object, **kwargs: object) -> None:
        observed.append(Path(str(path)))
        assert not store_path.exists(), "store must be renamed aside before removal"
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(workstore_module.shutil, "rmtree", spying_rmtree)
    session.remove_after_publication()

    assert observed == [retired_path]
    assert not store_path.exists()
    assert not retired_path.exists()


def test_cleanup_reclaims_a_stale_retired_remnant_left_by_an_earlier_interruption(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    # Simulate a crash between the quarantine rename and the final removal:
    # the live path is gone, only the deterministic retired sibling remains.
    retired_path = workstore_module._retired_store_path(store_path)
    os.rename(store_path, retired_path)
    assert retired_path.exists()
    assert not store_path.exists()

    # The live path must classify as absent, never adopt the remnant.
    assert inspect_work_store(store_path, _identity()).status == "absent"

    # A fresh store can be created and cleaned up at the same path; cleanup
    # must reclaim the stale remnant deterministically, not choke on it.
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        assert session.recovered is False
        session.commit({"chunk-1": "marker-1"})
    cleanup_work_store(store_path, _identity(), validate_publication=lambda: None)

    assert not store_path.exists()
    assert not retired_path.exists()


def test_cleanup_work_store_removes_after_validation_from_a_fresh_lease(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})

    validated: list[bool] = []
    cleanup_work_store(store_path, _identity(), validate_publication=lambda: validated.append(True))

    assert validated == [True]
    assert not store_path.exists()


def test_cleanup_work_store_is_a_noop_when_absent(tmp_path: Path) -> None:
    store_path = tmp_path / "never-created"
    called = False

    def validate() -> None:
        nonlocal called
        called = True

    cleanup_work_store(store_path, _identity(), validate_publication=validate)
    assert called is False


def test_cleanup_work_store_fails_closed_for_an_incompatible_identity(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity(consumer="widget-exporter")):
        pass

    with pytest.raises(WorkStoreIncompatibleError):
        cleanup_work_store(
            store_path,
            _identity(consumer="someone-else"),
            validate_publication=lambda: pytest.fail("must not validate an incompatible store"),
        )
    assert store_path.exists()


def test_identity_payload_is_sanitized_and_redacted_before_persisting(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    identity = _identity(api_token="super-secret-value")
    with WorkStoreSession.open_or_create(store_path, identity) as session:
        assert session.identity_payload["api_token"] == "<redacted>"
    metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
    payload = json.loads(metadata_path.read_bytes())
    assert payload["identity_payload"]["api_token"] == "<redacted>"
    assert "super-secret-value" not in metadata_path.read_text()


def test_open_or_create_distinguishes_identities_that_redact_to_the_same_value(
    tmp_path: Path,
) -> None:
    """Two payloads that redact identically must never collide as one identity.

    ``sanitize_metadata_fields`` redacts any field whose NAME matches its
    sensitive-name heuristics (like ``session_id``) regardless of its
    value, so identity_sha256 must be computed over the raw payload, not
    the sanitized one: hashing the sanitized copy would make two genuinely
    different sessions indistinguishable and let one silently adopt the
    other's completed chunks.
    """
    store_path = tmp_path / "store"
    identity_a = _identity(session_id="run-alpha-2026")
    identity_b = _identity(session_id="run-beta-2026")

    with WorkStoreSession.open_or_create(store_path, identity_a) as session:
        session.commit({"chunk-0": "marker-0"})

    inspection = inspect_work_store(store_path, identity_b)
    assert inspection.status == "incompatible"
    with pytest.raises(WorkStoreIncompatibleError) as excinfo:
        WorkStoreSession.open_or_create(store_path, identity_b)
    assert excinfo.value.status == "incompatible"

    # identity_a itself remains genuinely, exclusively resumable.
    original = inspect_work_store(store_path, identity_a)
    assert original.status == "resumable"
    assert original.completed_chunk_ids == ("chunk-0",)


def test_identity_with_a_secret_named_field_round_trips_while_persisted_redacted(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    identity = _identity(session_id="run-alpha-2026")

    with WorkStoreSession.open_or_create(store_path, identity) as session:
        assert session.identity_payload["session_id"] == "<redacted>"
        session.commit({"chunk-0": "marker-0"})

    metadata_path = store_path / workstore_module.WORK_STORE_METADATA_NAME
    payload = json.loads(metadata_path.read_bytes())
    assert payload["identity_payload"]["session_id"] == "<redacted>"
    assert "run-alpha-2026" not in metadata_path.read_text()

    # The exact same raw identity still resumes, matched by its raw hash.
    with WorkStoreSession.open_or_create(store_path, identity) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ("chunk-0",)
        assert reopened.identity_payload["session_id"] == "<redacted>"


def test_open_or_create_accepts_pathlike_values_in_a_command_field(tmp_path: Path) -> None:
    """A payload sanitize_metadata_fields accepts must not crash identity hashing.

    sanitize_metadata_fields accepts os.PathLike items inside a
    command-named field (converting them via sanitize_command); the raw
    canonicalization used for identity hashing must handle the same
    PathLike values directly, or a payload that passes sanitization would
    crash json.dumps instead of hashing cleanly.
    """
    store_path = tmp_path / "store"
    identity = _identity(command=[Path("foo"), Path("bar")])

    with WorkStoreSession.open_or_create(store_path, identity) as session:
        session.commit({"chunk-0": "marker-0"})

    with WorkStoreSession.open_or_create(store_path, identity) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ("chunk-0",)


def test_open_or_create_accepts_a_custom_sequence_subtype_in_identity_payload(
    tmp_path: Path,
) -> None:
    """A custom Sequence subtype sanitize_metadata_fields accepts must not crash hashing.

    sanitize_metadata_fields recognizes any ``collections.abc.Sequence``
    implementation, not just ``list``/``tuple``, and always sanitizes it
    down to a plain ``list``. The raw canonicalization used for identity
    hashing must do the same normalization, or a payload that passes
    sanitization would crash json.dumps.
    """
    store_path = tmp_path / "store"
    identity = _identity(tags=_CustomSequence(["a", "b", "c"]))

    with WorkStoreSession.open_or_create(store_path, identity) as session:
        session.commit({"chunk-0": "marker-0"})

    with WorkStoreSession.open_or_create(store_path, identity) as reopened:
        assert reopened.recovered is True
        assert reopened.completed_chunk_ids == ("chunk-0",)


def test_identical_raw_identities_get_different_salts_and_each_round_trips(
    tmp_path: Path,
) -> None:
    """Two stores with the same raw identity must not share a salt or digest.

    A fresh random salt per store means an attacker who recovers one
    store's metadata cannot reuse precomputed work against another store
    that happens to share the same underlying identity payload.
    """
    store_a = tmp_path / "store-a"
    store_b = tmp_path / "store-b"
    identity = _identity()

    with WorkStoreSession.open_or_create(store_a, identity) as session:
        session.commit({"chunk-a": "marker-a"})
    with WorkStoreSession.open_or_create(store_b, identity) as session:
        session.commit({"chunk-b": "marker-b"})

    metadata_a = json.loads((store_a / workstore_module.WORK_STORE_METADATA_NAME).read_bytes())
    metadata_b = json.loads((store_b / workstore_module.WORK_STORE_METADATA_NAME).read_bytes())
    assert metadata_a["identity_kdf"]["salt"] != metadata_b["identity_kdf"]["salt"]
    assert metadata_a["identity_digest"] != metadata_b["identity_digest"]

    with WorkStoreSession.open_or_create(store_a, identity) as reopened_a:
        assert reopened_a.recovered is True
        assert reopened_a.completed_chunk_ids == ("chunk-a",)
    with WorkStoreSession.open_or_create(store_b, identity) as reopened_b:
        assert reopened_b.recovered is True
        assert reopened_b.completed_chunk_ids == ("chunk-b",)


def test_open_or_create_rolls_back_new_store_dir_on_a_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "store"

    def broken_write_metadata(*args: object, **kwargs: object) -> None:
        raise OSError("simulated durability failure")

    monkeypatch.setattr(workstore_module, "_write_metadata", broken_write_metadata)
    with pytest.raises(OSError, match="simulated durability failure"):
        WorkStoreSession.open_or_create(store_path, _identity())

    assert not store_path.exists()
    # The lease must also have been released so a retry is not blocked.
    with claim_work_store_lease(store_path):
        pass


def test_open_or_create_rollback_quarantines_before_removing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creation rollback must use the same quarantine-rename removal path.

    Failing after metadata is durably written but before the journal exists
    is exactly the on-disk shape an interrupted *bare* rmtree rollback could
    leave behind; using _retire_and_remove_store here means a crash mid
    rollback can only ever leave debris under the retired sibling name, not
    directly at the live store path.
    """
    store_path = tmp_path / "store"
    retired_path = workstore_module._retired_store_path(store_path)
    calls: list[Path] = []
    original_retire_and_remove = workstore_module._retire_and_remove_store

    def spying_retire_and_remove(path: Path) -> None:
        calls.append(path)
        original_retire_and_remove(path)

    def broken_create_empty_journal(*args: object, **kwargs: object) -> None:
        raise OSError("simulated failure after metadata was written")

    monkeypatch.setattr(workstore_module, "_create_empty_journal", broken_create_empty_journal)
    monkeypatch.setattr(workstore_module, "_retire_and_remove_store", spying_retire_and_remove)

    with pytest.raises(OSError, match="simulated failure after metadata was written"):
        WorkStoreSession.open_or_create(store_path, _identity())

    assert calls == [store_path]
    assert not store_path.exists()
    assert not retired_path.exists()
    with claim_work_store_lease(store_path):
        pass


def test_open_or_create_rollback_falls_back_to_direct_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing quarantine rename must not permanently block a retry.

    When the quarantine-rename-then-remove path cannot even be attempted,
    rollback best-effort falls back to a direct removal so the live path is
    not left blocked forever, even though the on-disk outcome remains
    best-effort rather than guaranteed.
    """
    store_path = tmp_path / "store"

    def broken_create_empty_journal(*args: object, **kwargs: object) -> None:
        raise OSError("simulated failure after metadata was written")

    def broken_retire_and_remove(path: Path) -> None:
        raise OSError("simulated quarantine-rename failure")

    monkeypatch.setattr(workstore_module, "_create_empty_journal", broken_create_empty_journal)
    monkeypatch.setattr(workstore_module, "_retire_and_remove_store", broken_retire_and_remove)

    with pytest.raises(OSError, match="simulated failure after metadata was written"):
        WorkStoreSession.open_or_create(store_path, _identity())

    assert not store_path.exists()
    with claim_work_store_lease(store_path):
        pass


def test_inspect_and_open_classify_damaged_when_metadata_absent_but_journal_present(
    tmp_path: Path,
) -> None:
    """Mammoth's own journal without Mammoth's own metadata is not legacy.

    This is the signature of Mammoth's own debris (an interrupted creation
    or rollback, or a partial manual edit), not a foreign pre-Mammoth store,
    and must fail closed as damaged rather than being treated as
    adoptable-or-preservable legacy content.
    """
    store_path = tmp_path / "store"
    store_path.mkdir(mode=0o700)
    journal_path = store_path / workstore_module.WORK_STORE_JOURNAL_NAME
    descriptor = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)

    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "damaged"
    with pytest.raises(WorkStoreDamagedError):
        WorkStoreSession.open_or_create(store_path, _identity())


def test_context_manager_close_preserves_the_store_for_later_resume(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    with WorkStoreSession.open_or_create(store_path, _identity()) as session:
        session.commit({"chunk-0": "marker-0"})
    assert store_path.exists()
    inspection = inspect_work_store(store_path, _identity())
    assert inspection.status == "resumable"


def test_open_or_create_rejects_a_non_mapping_identity_payload(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    not_a_mapping: object = ["not", "a", "mapping"]
    with pytest.raises(WorkStoreValidationError):
        WorkStoreSession.open_or_create(store_path, not_a_mapping)  # type: ignore[arg-type]


def test_lease_file_is_a_safe_owned_regular_file(tmp_path: Path) -> None:
    store_path = tmp_path / "nested" / "store"
    with claim_work_store_lease(store_path) as lease:
        metadata = os.stat(lease.path)
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_uid == os.geteuid()


def test_claim_work_store_lease_rejects_an_unsafe_preexisting_lease_file(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store"
    lease_path = workstore_module._lease_path(store_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_bytes(b"")
    # World-writable: a pre-existing lease file with unsafe permissions must
    # be rejected on the already-opened descriptor, not silently claimed.
    os.chmod(lease_path, 0o666)

    with pytest.raises(WorkStoreDamagedError, match="unsafe writable permissions"):
        claim_work_store_lease(store_path)


def test_claim_work_store_lease_rejects_a_symlinked_lease_path(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    lease_path = workstore_module._lease_path(store_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere.lock"
    target.write_bytes(b"")
    lease_path.symlink_to(target)

    # O_NOFOLLOW on a final-component symlink raises a raw OSError (ELOOP);
    # this must translate into the module's own error family instead of
    # escaping unwrapped.
    with pytest.raises(workstore_module.WorkStoreError):
        claim_work_store_lease(store_path)


def test_write_metadata_removes_its_temporary_file_on_a_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "store"
    store_path.mkdir(mode=0o700)

    def failing_fsync(descriptor: int) -> None:
        raise OSError("simulated metadata fsync failure")

    monkeypatch.setattr(workstore_module.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="simulated metadata fsync failure"):
        workstore_module._write_metadata(
            store_path,
            sanitized={"consumer": "widget-exporter"},
            identity_digest="a" * 64,
            kdf=workstore_module._generate_identity_kdf(),
            resolved_store_path=str(store_path.resolve()),
        )

    # The failed write's temporary file must not survive, and the real
    # target must never have been created either.
    leftovers = list(store_path.glob(f".{workstore_module.WORK_STORE_METADATA_NAME}.*.tmp"))
    assert leftovers == []
    assert not (store_path / workstore_module.WORK_STORE_METADATA_NAME).exists()
