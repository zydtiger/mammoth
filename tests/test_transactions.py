"""Focused coverage for durable core multi-artifact publication transactions."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from ctypes import CDLL
from pathlib import Path
from typing import cast

import pytest

from mammoth.core import (
    ArtifactTransactionConflictError,
    ArtifactTransactionPlan,
    ArtifactTransactionRecoveryError,
    ArtifactTransactionRecoveryRequired,
    ArtifactTransactionValidationError,
    TransactionArtifact,
    TransactionArtifactSpec,
    build_artifact_transaction_plan,
    claim_artifact_transaction_leases,
    move_directory_into_transaction_stage,
    publish_artifact_transaction,
    recover_artifact_transaction,
    seal_artifact_transaction,
    stage_transaction_file,
    transaction_journal_exists,
    transaction_journal_path,
    transaction_stage_path,
    transactions,
)


class InjectedInterruption(RuntimeError):
    """Synthetic process loss used to leave a journal at one durable boundary."""


def create_plan(
    tmp_path: Path,
    *,
    transaction_id: str = "generation-1",
    mode: str = "create_only",
) -> ArtifactTransactionPlan:
    """Build one mixed file/directory plan with caller-written sibling stages."""
    root = tmp_path / "publication-root"
    root.mkdir(parents=True)
    file_target = root / "report.json"
    directory_target = root / "payload"
    if mode == "replace":
        file_target.write_text("old report")
        directory_target.mkdir()
        (directory_target / "old.txt").write_text("old payload")
    file_stage = root / f".mammoth-txn-{transaction_id}-report.stage"
    file_stage.write_text("new report")
    directory_stage = root / f".mammoth-txn-{transaction_id}-payload.stage"
    directory_stage.mkdir()
    (directory_stage / "nested").mkdir()
    (directory_stage / "nested" / "part.txt").write_text("new payload")
    return ArtifactTransactionPlan(
        transaction_id=transaction_id,
        lease_root=root,
        artifacts=(
            TransactionArtifact("report", file_stage, file_target, "file"),
            TransactionArtifact("payload", directory_stage, directory_target, "directory"),
        ),
        mode=mode,  # type: ignore[arg-type]
        recovery_policy=("roll_forward" if mode == "create_only" else "rollback_before_commit"),
    )


def test_consumer_planner_provisions_parents_and_preserves_stable_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "new" / "publication-root"
    report_target = root / "report.json"
    payload_target = root / "payload"
    artifacts = (
        TransactionArtifactSpec("report", report_target, "file"),
        TransactionArtifactSpec("payload", payload_target, "directory"),
    )

    plan = build_artifact_transaction_plan(
        namespace="consumer-plan",
        artifacts=artifacts,
        replace=False,
    )
    replacement = build_artifact_transaction_plan(
        namespace="consumer-plan",
        artifacts=artifacts,
        replace=True,
    )

    topology = "\0".join(
        f"{artifact.key}\0{artifact.kind}\0{artifact.target}"
        for artifact in sorted(artifacts, key=lambda artifact: artifact.key)
    )
    expected_id = f"consumer-plan-{hashlib.sha256(topology.encode()).hexdigest()[:16]}"
    assert root.is_dir()
    assert not report_target.exists()
    assert not payload_target.exists()
    assert plan.transaction_id == expected_id
    assert plan.lease_root == root
    assert plan.artifact_roots == (root,)
    assert tuple(artifact.key for artifact in plan.artifacts) == ("report", "payload")
    assert plan.artifacts[0].stage == root / f".mammoth-txn-{expected_id}-report.stage"
    assert plan.mode == "create_only"
    assert plan.recovery_policy == "roll_forward"
    assert replacement.transaction_id == expected_id
    assert replacement.mode == "replace"
    assert replacement.recovery_policy == "rollback_before_commit"


def test_consumer_planner_derives_multiple_roots_without_reordering_artifacts(
    tmp_path: Path,
) -> None:
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("the host has no separate shared-memory filesystem")
    memory_root = Path(tempfile.mkdtemp(prefix="mammoth-consumer-plan-", dir=shared_memory))
    local_root = tmp_path / "local-root"
    try:
        artifacts = (
            TransactionArtifactSpec("payload", memory_root / "payload", "directory"),
            TransactionArtifactSpec("report", local_root / "report.json", "file"),
        )

        plan = build_artifact_transaction_plan(
            namespace="consumer-multi-root",
            artifacts=artifacts,
            replace=False,
        )

        assert plan.artifact_roots == tuple(sorted((local_root, memory_root), key=str))
        assert plan.lease_root == plan.artifact_roots[0]
        assert tuple(artifact.key for artifact in plan.artifacts) == ("payload", "report")
        assert all(
            artifact.stage
            == artifact.target.parent / f".mammoth-txn-{plan.transaction_id}-{artifact.key}.stage"
            for artifact in plan.artifacts
        )
    finally:
        shutil.rmtree(memory_root)


def test_consumer_planner_rejects_invalid_specs_and_unsafe_topology(tmp_path: Path) -> None:
    root = tmp_path / "publication-root"
    report = root / "report.json"
    payload = root / "payload"
    valid = (
        TransactionArtifactSpec("report", report, "file"),
        TransactionArtifactSpec("payload", payload, "directory"),
    )

    with pytest.raises(ArtifactTransactionValidationError, match="namespace"):
        build_artifact_transaction_plan(namespace="Unsafe", artifacts=valid, replace=False)
    with pytest.raises(ArtifactTransactionValidationError, match="keys must be unique"):
        build_artifact_transaction_plan(
            namespace="consumer-invalid",
            artifacts=(
                TransactionArtifactSpec("report", report, "file"),
                TransactionArtifactSpec("report", payload, "directory"),
            ),
            replace=False,
        )
    with pytest.raises(ArtifactTransactionValidationError, match="targets must be unique"):
        build_artifact_transaction_plan(
            namespace="consumer-invalid",
            artifacts=(
                TransactionArtifactSpec("report", report, "file"),
                TransactionArtifactSpec("payload", report, "file"),
            ),
            replace=False,
        )
    with pytest.raises(ArtifactTransactionValidationError, match="must not overlap"):
        build_artifact_transaction_plan(
            namespace="consumer-invalid",
            artifacts=(
                TransactionArtifactSpec("payload", root / "payload", "directory"),
                TransactionArtifactSpec("report", root / "payload" / "report.json", "file"),
            ),
            replace=False,
        )
    assert not (root / "payload").exists()

    external = tmp_path / "external"
    external.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(ArtifactTransactionValidationError, match="symlink"):
        build_artifact_transaction_plan(
            namespace="consumer-invalid",
            artifacts=(
                TransactionArtifactSpec("report", alias / "report.json", "file"),
                TransactionArtifactSpec("payload", payload, "directory"),
            ),
            replace=False,
        )

    root.mkdir()
    special = root / "special"
    os.mkfifo(special)
    with pytest.raises(ArtifactTransactionValidationError, match="kind does not match"):
        build_artifact_transaction_plan(
            namespace="consumer-invalid",
            artifacts=(
                TransactionArtifactSpec("report", special, "file"),
                TransactionArtifactSpec("payload", payload, "directory"),
            ),
            replace=False,
        )


@pytest.mark.parametrize(
    ("reserved_name", "kind", "message"),
    (
        (".mammoth-txn-foreign-payload.stage", "file", "reserved Mammoth object"),
        (".mammoth-txn-foreign-payload.backup", "directory", "reserved Mammoth object"),
        (".mammoth-transactions/foreign", "file", "Mammoth metadata"),
    ),
)
def test_consumer_planner_rejects_reserved_targets_before_parent_provisioning(
    tmp_path: Path,
    reserved_name: str,
    kind: str,
    message: str,
) -> None:
    root = tmp_path / "unprovisioned-root"
    with pytest.raises(ArtifactTransactionValidationError, match=message):
        build_artifact_transaction_plan(
            namespace="consumer-reserved",
            artifacts=(
                TransactionArtifactSpec("reserved", root / reserved_name, kind),  # type: ignore[arg-type]
                TransactionArtifactSpec("normal", root / "normal.txt", "file"),
            ),
            replace=False,
        )

    assert not root.exists()


def create_consumer_plan(tmp_path: Path, *, replace: bool = False) -> ArtifactTransactionPlan:
    """Build one high-level mixed transaction plan for staging tests."""
    root = tmp_path / "consumer-root"
    return build_artifact_transaction_plan(
        namespace="consumer-stage",
        artifacts=(
            TransactionArtifactSpec("report", root / "report.json", "file"),
            TransactionArtifactSpec("payload", root / "payload", "directory"),
        ),
        replace=replace,
    )


def test_stage_transaction_file_is_exclusive_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    synchronized: list[int] = []
    original_fsync = transactions.os.fsync

    def record_fsync(descriptor: int) -> None:
        synchronized.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(transactions.os, "fsync", record_fsync)
    stage = stage_transaction_file(plan, "report", b'{"complete":true}\n')

    assert stage == transaction_stage_path(plan, "report")
    assert stage.read_bytes() == b'{"complete":true}\n'
    assert len(synchronized) >= 2
    with pytest.raises(FileExistsError, match="already exists"):
        stage_transaction_file(plan, "report", b"replacement")
    with pytest.raises(ArtifactTransactionValidationError, match="not a file"):
        stage_transaction_file(plan, "payload", b"not a directory")


def test_stage_transaction_file_removes_only_its_incomplete_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)

    def fail_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(transactions.os, "write", fail_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        stage_transaction_file(plan, "report", b"incomplete")
    assert not transaction_stage_path(plan, "report").exists()

    occupied = transaction_stage_path(plan, "report")
    occupied.write_text("other publisher")
    with pytest.raises(FileExistsError, match="already exists"):
        stage_transaction_file(plan, "report", b"must not remove occupied stage")
    assert occupied.read_text() == "other publisher"


def test_stage_transaction_file_rejects_a_post_sync_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    stage = transaction_stage_path(plan, "report")
    replacement = stage.parent / "replacement.txt"
    replacement.write_text("substituted")
    original_sync = transactions.sync_directory_strict
    replaced = False

    def sync_then_replace(path: Path, *, lease_root: Path | None = None) -> None:
        nonlocal replaced
        original_sync(path, lease_root=lease_root)
        if path == stage.parent and not replaced:
            replaced = True
            os.replace(replacement, stage)

    monkeypatch.setattr(transactions, "sync_directory_strict", sync_then_replace)
    with pytest.raises(ArtifactTransactionRecoveryError, match="changed after synchronization"):
        stage_transaction_file(plan, "report", b"original")

    assert stage.read_text() == "substituted"


def test_move_directory_into_transaction_stage_adopts_one_safe_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    source = plan.lease_root / "rendered"
    source.mkdir()
    (source / "part.txt").write_text("payload")
    synchronized: list[int] = []
    original_fsync = transactions.os.fsync

    def record_fsync(descriptor: int) -> None:
        synchronized.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(transactions.os, "fsync", record_fsync)
    stage = move_directory_into_transaction_stage(plan, "payload", source)

    assert stage == transaction_stage_path(plan, "payload")
    assert (stage / "part.txt").read_text() == "payload"
    assert not source.exists()
    assert len(synchronized) >= 4


def test_directory_staging_preserves_evidence_after_an_ambiguous_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    source = plan.lease_root / "rendered"
    source.mkdir()
    (source / "part.txt").write_text("payload")
    original_exchange = transactions.rename_exchange

    def exchange_then_interrupt(
        first: Path, second: Path, *, lease_root: Path | None = None
    ) -> None:
        original_exchange(first, second, lease_root=lease_root)
        raise InjectedInterruption("after directory exchange")

    monkeypatch.setattr(transactions, "rename_exchange", exchange_then_interrupt)
    with pytest.raises(InjectedInterruption, match="after directory exchange"):
        move_directory_into_transaction_stage(plan, "payload", source)

    stage = transaction_stage_path(plan, "payload")
    assert (stage / "part.txt").read_text() == "payload"
    assert source.is_dir()
    assert not tuple(source.iterdir())


def test_directory_staging_rejects_source_replaced_during_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    source = plan.lease_root / "rendered"
    source.mkdir()
    (source / "original.txt").write_text("original payload")
    replacement = plan.lease_root / "replacement"
    replacement.mkdir()
    (replacement / "replacement.txt").write_text("replacement payload")
    original_exchange = transactions.rename_exchange

    def exchange_then_replace(
        first: Path, second: Path, *, lease_root: Path | None = None
    ) -> None:
        original_exchange(first, second, lease_root=lease_root)
        original_exchange(replacement, first, lease_root=lease_root)

    monkeypatch.setattr(transactions, "rename_exchange", exchange_then_replace)
    with pytest.raises(ArtifactTransactionRecoveryError, match="identity-mismatched"):
        move_directory_into_transaction_stage(plan, "payload", source)

    stage = transaction_stage_path(plan, "payload")
    assert (stage / "replacement.txt").read_text() == "replacement payload"
    assert (replacement / "original.txt").read_text() == "original payload"
    assert source.is_dir()
    assert not tuple(source.iterdir())


def test_directory_staging_rejects_unsafe_sources_and_occupied_stage(tmp_path: Path) -> None:
    plan = create_consumer_plan(tmp_path)
    source = plan.lease_root / "rendered"
    source.mkdir()
    (source / "part.txt").write_text("payload")
    alias = plan.lease_root / "rendered-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ArtifactTransactionValidationError, match="symlink"):
        move_directory_into_transaction_stage(plan, "payload", alias)

    stage = transaction_stage_path(plan, "payload")
    stage.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        move_directory_into_transaction_stage(plan, "payload", source)
    assert source.exists()
    assert stage.exists()

    unsafe_plan = create_consumer_plan(tmp_path / "unsafe")
    unsafe_source = unsafe_plan.lease_root / "unsafe-rendered"
    unsafe_source.mkdir()
    os.mkfifo(unsafe_source / "stream")
    with pytest.raises(ArtifactTransactionValidationError, match="special"):
        move_directory_into_transaction_stage(unsafe_plan, "payload", unsafe_source)


def test_directory_staging_rejects_transaction_metadata_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "consumer-root"
    plan = create_consumer_plan(tmp_path)
    other_plan = build_artifact_transaction_plan(
        namespace="other-journal",
        artifacts=(
            TransactionArtifactSpec("other-report", root / "other-report.json", "file"),
            TransactionArtifactSpec("other-payload", root / "other-payload", "directory"),
        ),
        replace=False,
    )
    stage_transaction_file(other_plan, "other-report", b"other report")
    rendered = root / "other-rendered"
    rendered.mkdir()
    (rendered / "part.txt").write_text("other payload")
    move_directory_into_transaction_stage(other_plan, "other-payload", rendered)
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(other_plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)

    metadata = root / ".mammoth-transactions"
    stage = transaction_stage_path(plan, "payload")
    with pytest.raises(ArtifactTransactionValidationError, match="metadata"):
        move_directory_into_transaction_stage(plan, "payload", metadata)

    assert transaction_journal_path(other_plan).is_file()
    assert metadata.is_dir()
    assert not stage.exists()


def test_directory_staging_rejects_nested_transaction_metadata_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_consumer_plan(tmp_path)
    source = plan.lease_root / "rendered"
    other_plan = build_artifact_transaction_plan(
        namespace="nested-journal",
        artifacts=(
            TransactionArtifactSpec("other-report", source / "other-report.json", "file"),
            TransactionArtifactSpec("other-payload", source / "other-payload", "directory"),
        ),
        replace=False,
    )
    stage_transaction_file(other_plan, "other-report", b"other report")
    nested_rendered = source / "other-rendered"
    nested_rendered.mkdir()
    (nested_rendered / "part.txt").write_text("other payload")
    move_directory_into_transaction_stage(other_plan, "other-payload", nested_rendered)
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(other_plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)

    stage = transaction_stage_path(plan, "payload")
    with pytest.raises(ArtifactTransactionValidationError, match="contains Mammoth metadata"):
        move_directory_into_transaction_stage(plan, "payload", source)

    assert transaction_journal_path(other_plan).is_file()
    assert (source / ".mammoth-transactions").is_dir()
    assert not stage.exists()


def test_directory_staging_rejects_foreign_reserved_stage_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "consumer-root"
    plan = create_consumer_plan(tmp_path)
    other_plan = build_artifact_transaction_plan(
        namespace="foreign-transaction",
        artifacts=(
            TransactionArtifactSpec("other-report", root / "other-report.json", "file"),
            TransactionArtifactSpec("other-payload", root / "other-payload", "directory"),
        ),
        replace=False,
    )
    stage_transaction_file(other_plan, "other-report", b"other report")
    rendered = root / "other-rendered"
    rendered.mkdir()
    (rendered / "part.txt").write_text("other payload")
    foreign_stage = move_directory_into_transaction_stage(other_plan, "other-payload", rendered)
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(other_plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)
    foreign_backup = transactions.transaction_backup_path(other_plan, other_plan.artifacts[1])
    foreign_backup.mkdir()
    (foreign_backup / "original.txt").write_text("foreign original")

    stage = transaction_stage_path(plan, "payload")
    with pytest.raises(ArtifactTransactionValidationError, match="reserved Mammoth object"):
        move_directory_into_transaction_stage(plan, "payload", foreign_stage)
    with pytest.raises(ArtifactTransactionValidationError, match="reserved Mammoth object"):
        move_directory_into_transaction_stage(plan, "payload", foreign_backup)

    assert transaction_journal_path(other_plan).is_file()
    assert (foreign_stage / "part.txt").read_text() == "other payload"
    assert (foreign_backup / "original.txt").read_text() == "foreign original"
    assert not stage.exists()


def test_create_only_transaction_publishes_mixed_artifacts_and_cleans_state(
    tmp_path: Path,
) -> None:
    validated_paths: list[Path] = []

    def validate_report(path: Path) -> None:
        validated_paths.append(path)
        assert path.read_text() == "new report"

    plan = create_plan(tmp_path)
    report = plan.artifacts[0]
    plan = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifacts=(
            TransactionArtifact(
                report.key, report.stage, report.target, report.kind, validate_report
            ),
            plan.artifacts[1],
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )

    result = publish_artifact_transaction(plan)

    assert result.committed_targets == (plan.artifacts[0].target, plan.artifacts[1].target)
    assert result.restored_targets == ()
    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"
    assert not plan.artifacts[0].stage.exists()
    assert not plan.artifacts[1].stage.exists()
    assert not transaction_journal_path(plan).exists()
    assert validated_paths[0] == plan.artifacts[0].stage
    assert validated_paths[1:] == [plan.artifacts[0].target] * 3


def test_replacement_transaction_retains_original_until_commit_then_cleans_backup(
    tmp_path: Path,
) -> None:
    plan = create_plan(tmp_path, mode="replace")

    result = publish_artifact_transaction(plan)

    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"
    for artifact in plan.artifacts:
        assert not transactions.transaction_backup_path(plan, artifact).exists()
    assert not transaction_journal_path(plan).exists()


def test_existing_journal_requires_explicit_recovery_and_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first durable target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    with pytest.raises(ArtifactTransactionRecoveryRequired):
        publish_artifact_transaction(plan)
    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"
    assert not transaction_journal_path(plan).exists()


def test_transaction_journal_exists_reflects_missing_and_present_journal(
    tmp_path: Path,
) -> None:
    plan = create_plan(tmp_path)

    assert transaction_journal_exists(plan) is False

    publish_artifact_transaction(plan)

    assert transaction_journal_exists(plan) is False


def test_recover_artifact_transaction_missing_ok_returns_none_without_journal(
    tmp_path: Path,
) -> None:
    plan = create_plan(tmp_path)

    assert transaction_journal_exists(plan) is False
    assert recover_artifact_transaction(plan, missing_ok=True) is None
    with pytest.raises(FileNotFoundError):
        recover_artifact_transaction(plan)


def test_recover_artifact_transaction_missing_ok_still_recovers_present_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first durable target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    assert transaction_journal_exists(plan) is True
    result = recover_artifact_transaction(plan, missing_ok=True)

    assert result is not None
    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"
    assert not transaction_journal_path(plan).exists()
    assert transaction_journal_exists(plan) is False


def create_validated_three_artifact_plan(
    tmp_path: Path,
    expected: dict[str, str],
    *,
    replace: bool = False,
    observed_validation_paths: list[Path] | None = None,
) -> ArtifactTransactionPlan:
    """Build and stage a three-file plan whose validators read mutable expectations."""
    root = tmp_path / "three-artifact-root"

    def validator(key: str) -> transactions.ArtifactValidator:
        def validate(path: Path) -> None:
            if observed_validation_paths is not None:
                observed_validation_paths.append(path)
            assert path.read_text() == expected[key]

        return validate

    plan = build_artifact_transaction_plan(
        namespace="recovery-preflight",
        artifacts=tuple(
            TransactionArtifactSpec(key, root / f"{key}.txt", "file", validator(key))
            for key in ("first", "second", "third")
        ),
        replace=replace,
    )
    if replace:
        for artifact in plan.artifacts:
            artifact.target.write_text(f"old-{artifact.key}")
    for artifact in plan.artifacts:
        stage_transaction_file(plan, artifact.key, expected[artifact.key].encode())
    return plan


def test_recovery_preflights_every_visible_stage_before_first_target_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"first": "first", "second": "second", "third": "third"}
    plan = create_validated_three_artifact_plan(tmp_path, expected)
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)
    expected["third"] = "semantic-stale"

    with pytest.raises(ArtifactTransactionValidationError, match="third"):
        recover_artifact_transaction(plan)

    assert all(not artifact.target.exists() for artifact in plan.artifacts)
    assert all(artifact.stage.exists() for artifact in plan.artifacts)
    assert transaction_journal_path(plan).exists()


def test_recovery_preflight_blocks_later_publication_after_a_partial_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"first": "first", "second": "second", "third": "third"}
    plan = create_validated_three_artifact_plan(tmp_path, expected)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def publish_first_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", publish_first_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)
    expected["third"] = "semantic-stale"

    with pytest.raises(ArtifactTransactionValidationError, match="third"):
        recover_artifact_transaction(plan)

    assert plan.artifacts[0].target.read_text() == "first"
    assert not plan.artifacts[1].target.exists()
    assert not plan.artifacts[2].target.exists()
    assert plan.artifacts[1].stage.exists()
    assert plan.artifacts[2].stage.exists()
    assert transaction_journal_path(plan).exists()


def test_recovery_reassociates_journal_records_when_specs_are_reordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reordered-root"
    specs = (
        TransactionArtifactSpec("first", root / "first.txt", "file"),
        TransactionArtifactSpec("second", root / "second.txt", "file"),
        TransactionArtifactSpec("third", root / "third.txt", "file"),
    )
    plan = build_artifact_transaction_plan(
        namespace="reordered-recovery",
        artifacts=specs,
        replace=False,
    )
    for artifact in plan.artifacts:
        stage_transaction_file(plan, artifact.key, artifact.key.encode())
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)
    rebuilt_plan = build_artifact_transaction_plan(
        namespace="reordered-recovery",
        artifacts=tuple(reversed(specs)),
        replace=False,
    )
    published_targets: list[Path] = []
    original_rename = transactions.rename_without_overwrite

    def record_publication_order(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        original_rename(source, destination, lease_root=lease_root)
        if destination in {artifact.target for artifact in plan.artifacts}:
            published_targets.append(destination)

    monkeypatch.setattr(transactions, "rename_without_overwrite", record_publication_order)
    result = recover_artifact_transaction(rebuilt_plan)

    assert rebuilt_plan.transaction_id == plan.transaction_id
    assert published_targets == [artifact.target for artifact in plan.artifacts]
    assert result.committed_targets == tuple(artifact.target for artifact in rebuilt_plan.artifacts)
    assert all(artifact.target.read_text() == artifact.key for artifact in plan.artifacts)
    assert not transaction_journal_path(plan).exists()


def test_replacement_recovery_never_preflights_a_visible_old_target_as_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"first": "first", "second": "second", "third": "third"}
    observed: list[Path] = []
    plan = create_validated_three_artifact_plan(
        tmp_path,
        expected,
        replace=True,
        observed_validation_paths=observed,
    )
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(
        path: Path, journal: dict[str, object]
    ) -> transactions._JournalHandle:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert all(artifact.target.read_text() == f"old-{artifact.key}" for artifact in plan.artifacts)
    assert all(path.name.endswith(".stage") for path in observed)


def test_replacement_recovery_restores_prior_generation_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path, mode="replace")
    original_rename = transactions.rename_without_overwrite
    interrupted = False
    first_backup = transactions.transaction_backup_path(plan, plan.artifacts[0])

    def backup_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == first_backup and not interrupted:
            interrupted = True
            raise InjectedInterruption("after original backup rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", backup_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    result = recover_artifact_transaction(plan)

    assert plan.artifacts[0].target.read_text() == "old report"
    assert (plan.artifacts[1].target / "old.txt").read_text() == "old payload"
    assert result.restored_targets == (plan.artifacts[0].target,)
    assert not transaction_journal_path(plan).exists()
    assert not first_backup.exists()


def test_replacement_recovery_removes_new_target_then_restores_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path, mode="replace")
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def publish_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after replacement target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", publish_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    result = recover_artifact_transaction(plan)

    assert result.restored_targets == (plan.artifacts[0].target,)
    assert plan.artifacts[0].target.read_text() == "old report"
    assert (plan.artifacts[1].target / "old.txt").read_text() == "old payload"
    assert not transaction_journal_path(plan).exists()


def test_recovery_rejects_journal_bound_to_a_different_expected_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target:
            raise InjectedInterruption("after first target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)
    other_target = plan.lease_root / "substituted-report.json"
    expected_plan = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifacts=(
            TransactionArtifact("report", plan.artifacts[0].stage, other_target, "file"),
            plan.artifacts[1],
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )

    with pytest.raises(ArtifactTransactionRecoveryError, match="does not match"):
        recover_artifact_transaction(expected_plan)
    assert transaction_journal_path(plan).exists()


def test_recovery_preserves_journal_when_visible_target_identity_is_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target:
            raise InjectedInterruption("after first target rename")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)
    plan.artifacts[0].target.write_text("substituted")

    with pytest.raises(ArtifactTransactionRecoveryError, match="substituted"):
        recover_artifact_transaction(plan)
    assert transaction_journal_path(plan).exists()
    assert plan.artifacts[0].target.read_text() == "substituted"


def test_recovery_rolls_forward_from_durable_journal_before_any_target_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_create = transactions.create_transaction_journal

    def create_then_interrupt(path: Path, journal: dict[str, object]) -> None:
        original_create(path, journal)
        raise InjectedInterruption("after durable journal creation")

    monkeypatch.setattr(transactions, "create_transaction_journal", create_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"


def test_committed_recovery_retries_cleanup_without_republishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    interrupted = False

    def delete_then_interrupt(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise InjectedInterruption("before journal cleanup")
        raise AssertionError(f"unexpected repeated journal cleanup: {args}")

    monkeypatch.setattr(transactions, "delete_transaction_journal", delete_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.undo()

    result = recover_artifact_transaction(plan)

    assert result.committed_targets == (plan.artifacts[0].target, plan.artifacts[1].target)
    assert plan.artifacts[0].target.read_text() == "new report"
    assert not transaction_journal_path(plan).exists()


def test_recovery_finds_a_journal_interrupted_after_deterministic_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    retired_journal = transactions.transaction_retired_journal_path(plan)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def retire_journal_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == retired_journal and not interrupted:
            interrupted = True
            raise InjectedInterruption("after durable journal retirement")

    monkeypatch.setattr(transactions, "rename_without_overwrite", retire_journal_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert not transaction_journal_path(plan).exists()
    assert not retired_journal.exists()


def test_lease_contention_for_overlapping_target_is_deterministic(tmp_path: Path) -> None:
    plan = create_plan(tmp_path)
    leases = claim_artifact_transaction_leases(plan)
    try:
        with pytest.raises(ArtifactTransactionConflictError, match="overlapping"):
            claim_artifact_transaction_leases(plan)
    finally:
        leases.__exit__(None, None, None)
    with claim_artifact_transaction_leases(plan):
        pass


def test_nested_lease_roots_contend_for_the_same_stable_targets(tmp_path: Path) -> None:
    parent_root = tmp_path / "parent-root"
    child_root = parent_root / "child-root"
    child_root.mkdir(parents=True)

    def plan_for(lease_root: Path, transaction_id: str) -> ArtifactTransactionPlan:
        report_target = child_root / "report.json"
        payload_target = child_root / "payload"
        report_stage = child_root / f".mammoth-txn-{transaction_id}-report.stage"
        report_stage.write_text(transaction_id)
        payload_stage = child_root / f".mammoth-txn-{transaction_id}-payload.stage"
        payload_stage.mkdir()
        (payload_stage / "payload.txt").write_text(transaction_id)
        return ArtifactTransactionPlan(
            transaction_id=transaction_id,
            lease_root=lease_root,
            artifacts=(
                TransactionArtifact("report", report_stage, report_target, "file"),
                TransactionArtifact("payload", payload_stage, payload_target, "directory"),
            ),
            mode="create_only",
            recovery_policy="roll_forward",
        )

    parent_plan = plan_for(parent_root, "parent-generation")
    child_plan = plan_for(child_root, "child-generation")
    with (
        claim_artifact_transaction_leases(parent_plan),
        pytest.raises(ArtifactTransactionConflictError, match="overlapping"),
    ):
        claim_artifact_transaction_leases(child_plan)


def test_no_replace_rename_preserves_a_destination_created_after_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("transaction")
    libc = CDLL(None, use_errno=True)
    actual_renameat2 = libc.renameat2

    class DestinationRace:
        """Create an unrelated destination immediately before the kernel rename."""

        def renameat2(self, *arguments: object) -> int:
            destination.write_text("unrelated")
            return cast(int, actual_renameat2(*arguments))

    monkeypatch.setattr("ctypes.CDLL", lambda *_args, **_kwargs: DestinationRace())

    with pytest.raises(ArtifactTransactionRecoveryError, match="refusing to overwrite"):
        transactions.rename_without_overwrite(source, destination)

    assert source.read_text() == "transaction"
    assert destination.read_text() == "unrelated"


def test_sealing_syncs_stage_and_journal_parent_directories_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    synchronized: list[Path] = []
    original_sync = transactions.sync_directory_strict

    def record_sync(path: Path, *, lease_root: Path | None = None) -> None:
        synchronized.append(path)
        original_sync(path, lease_root=lease_root)

    monkeypatch.setattr(transactions, "sync_directory_strict", record_sync)
    publish_artifact_transaction(plan)

    metadata_directory = plan.lease_root / ".mammoth-transactions"
    journal_directory = metadata_directory / "journals"
    assert plan.lease_root in synchronized
    assert metadata_directory in synchronized
    assert journal_directory in synchronized


def test_plan_rejects_dotdot_escape_from_lease_root(tmp_path: Path) -> None:
    plan = create_plan(tmp_path)
    escaped_root = plan.lease_root / "child" / ".." / ".." / "outside"
    escaped_root.mkdir(parents=True)
    escaped_report = escaped_root / "report.json"
    escaped_stage = escaped_root / ".mammoth-txn-generation-1-report.stage"
    escaped_stage.write_text("outside")
    escaped = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifacts=(
            TransactionArtifact("report", escaped_stage, escaped_report, "file"),
            plan.artifacts[1],
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )

    with pytest.raises(ArtifactTransactionValidationError, match="beneath lease_root"):
        seal_artifact_transaction(escaped)


def test_plan_rejects_cross_artifact_target_stage_overlap(tmp_path: Path) -> None:
    plan = create_plan(tmp_path, mode="replace")
    overlapping_target = plan.artifacts[0].stage
    second_stage = plan.lease_root / ".mammoth-txn-generation-1-other.stage"
    second_stage.write_text("other")
    overlapping = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifacts=(
            plan.artifacts[0],
            TransactionArtifact("other", second_stage, overlapping_target, "file"),
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )

    with pytest.raises(ArtifactTransactionValidationError, match="must not overlap"):
        seal_artifact_transaction(overlapping)


def test_cleanup_retirement_preserves_an_occupied_backup_retirement_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path, mode="replace")
    backup = transactions.transaction_backup_path(plan, plan.artifacts[0])
    retired = transactions.transaction_retired_object_path(plan, plan.artifacts[0], "backup")
    original_rename = transactions.rename_without_overwrite
    raced = False

    def occupy_retirement_before_rename(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal raced
        if source == backup and destination == retired and not raced:
            raced = True
            retired.write_text("unrelated")
        original_rename(source, destination, lease_root=lease_root)

    monkeypatch.setattr(transactions, "rename_without_overwrite", occupy_retirement_before_rename)

    with pytest.raises(ArtifactTransactionRecoveryError, match="refusing to overwrite"):
        publish_artifact_transaction(plan)

    assert backup.read_text() == "old report"
    assert retired.read_text() == "unrelated"
    assert transaction_journal_path(plan).exists()


def test_interrupted_backup_retirement_is_recovered_from_its_deterministic_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path, mode="replace")
    backup = transactions.transaction_backup_path(plan, plan.artifacts[0])
    retired = transactions.transaction_retired_object_path(plan, plan.artifacts[0], "backup")
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def retire_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if source == backup and destination == retired and not interrupted:
            interrupted = True
            raise InjectedInterruption("after durable backup retirement")

    monkeypatch.setattr(transactions, "rename_without_overwrite", retire_then_interrupt)

    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert not backup.exists()
    assert not retired.exists()
    assert not transaction_journal_path(plan).exists()


def test_journal_update_preserves_an_unknown_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    journal_path = transaction_journal_path(plan)
    original_exchange = transactions.rename_exchange
    raced = False

    def replace_journal_before_exchange(
        first: Path, second: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal raced
        if first == journal_path and not raced:
            raced = True
            unrelated = journal_path.with_name("unrelated-journal")
            unrelated.write_text('{"unexpected":true}\n')
            os.replace(unrelated, journal_path)
        original_exchange(first, second, lease_root=lease_root)

    monkeypatch.setattr(transactions, "rename_exchange", replace_journal_before_exchange)

    with pytest.raises(
        ArtifactTransactionRecoveryError, match="journal changed before replacement"
    ):
        publish_artifact_transaction(plan)

    assert journal_path.read_text() == '{"unexpected":true}\n'


def test_recovery_reclaims_a_deterministic_journal_update_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    journal_path = transaction_journal_path(plan)
    swap_path = transactions.transaction_journal_swap_path(journal_path)
    original_exchange = transactions.rename_exchange
    interrupted = False

    def exchange_then_interrupt(
        first: Path, second: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_exchange(first, second, lease_root=lease_root)
        if first == journal_path and not interrupted:
            interrupted = True
            raise InjectedInterruption("after durable journal update exchange")

    monkeypatch.setattr(transactions, "rename_exchange", exchange_then_interrupt)
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "rename_exchange", original_exchange)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert not swap_path.exists()
    assert not journal_path.exists()


def test_publication_rejects_an_ancestor_symlink_created_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "publication-root"
    parent = root / "nested"
    parent.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    file_target = parent / "report.json"
    file_stage = parent / ".mammoth-txn-symlink-race-report.stage"
    file_stage.write_text("new report")
    directory_target = parent / "payload"
    directory_stage = parent / ".mammoth-txn-symlink-race-payload.stage"
    directory_stage.mkdir()
    (directory_stage / "part.txt").write_text("new payload")
    plan = ArtifactTransactionPlan(
        transaction_id="symlink-race",
        lease_root=root,
        artifacts=(
            TransactionArtifact("report", file_stage, file_target, "file"),
            TransactionArtifact("payload", directory_stage, directory_target, "directory"),
        ),
        mode="create_only",
        recovery_policy="roll_forward",
    )
    original_open_parent = transactions.open_confined_parent
    redirected = False

    def redirect_before_open(
        path: Path, *, lease_root: Path | None = None
    ) -> tuple[int, str]:
        nonlocal redirected
        if path == file_stage and not redirected:
            redirected = True
            os.rename(parent, external / "moved-artifacts")
            parent.symlink_to(external, target_is_directory=True)
        return original_open_parent(path, lease_root=lease_root)

    monkeypatch.setattr(transactions, "open_confined_parent", redirect_before_open)

    with pytest.raises(ArtifactTransactionRecoveryError, match="path is unsafe"):
        publish_artifact_transaction(plan)

    assert not (external / "report.json").exists()
    assert transaction_journal_path(plan).exists()


def test_confined_root_open_rejects_a_replaced_lease_root_ancestor(tmp_path: Path) -> None:
    ancestor = tmp_path / "ancestor"
    root = ancestor / "publication-root"
    root.mkdir(parents=True)
    external = tmp_path / "external"
    (external / "publication-root").mkdir(parents=True)
    moved = tmp_path / "moved-ancestor"
    os.rename(ancestor, moved)
    ancestor.symlink_to(external, target_is_directory=True)

    with pytest.raises(OSError):
        transactions.open_confined_directory(root, lease_root=root)


def test_plan_rejects_nested_mount_artifacts_outside_journal_filesystem(tmp_path: Path) -> None:
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("the host has no separate shared-memory filesystem")
    temporary_root = shared_memory / f"mammoth-transaction-{os.getpid()}"
    temporary_root.mkdir()
    try:
        file_target = temporary_root / "report.json"
        file_stage = temporary_root / ".mammoth-txn-cross-device-report.stage"
        file_stage.write_text("report")
        directory_target = temporary_root / "payload"
        directory_stage = temporary_root / ".mammoth-txn-cross-device-payload.stage"
        directory_stage.mkdir()
        (directory_stage / "payload.txt").write_text("payload")
        plan = ArtifactTransactionPlan(
            transaction_id="cross-device",
            lease_root=Path("/"),
            artifacts=(
                TransactionArtifact("report", file_stage, file_target, "file"),
                TransactionArtifact("payload", directory_stage, directory_target, "directory"),
            ),
            mode="create_only",
            recovery_policy="roll_forward",
        )

        with pytest.raises(ArtifactTransactionValidationError, match="artifact_root's filesystem"):
            seal_artifact_transaction(plan)
    finally:
        shutil.rmtree(temporary_root)


def create_multi_root_plan(
    tmp_path: Path, *, mode: str = "create_only"
) -> tuple[ArtifactTransactionPlan, Path]:
    """Build a plan whose journal and payload directory use separate local devices."""
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or shared_memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("the host has no separate shared-memory filesystem")
    coordinator_root = tmp_path / "coordinator-root"
    coordinator_root.mkdir()
    payload_root = Path(tempfile.mkdtemp(prefix="mammoth-multi-root-", dir=shared_memory))
    report_target = coordinator_root / "report.json"
    payload_target = payload_root / "payload"
    if mode == "replace":
        report_target.write_text("old report")
        payload_target.mkdir()
        (payload_target / "old.txt").write_text("old payload")
    transaction_id = "multi-root-generation"
    report_stage = coordinator_root / f".mammoth-txn-{transaction_id}-report.stage"
    report_stage.write_text("new report")
    payload_stage = payload_root / f".mammoth-txn-{transaction_id}-payload.stage"
    payload_stage.mkdir()
    (payload_stage / "part.txt").write_text("new payload")
    return (
        ArtifactTransactionPlan(
            transaction_id=transaction_id,
            lease_root=coordinator_root,
            artifact_roots=(coordinator_root, payload_root),
            artifacts=(
                TransactionArtifact("report", report_stage, report_target, "file"),
                TransactionArtifact("payload", payload_stage, payload_target, "directory"),
            ),
            mode=mode,  # type: ignore[arg-type]
            recovery_policy=(
                "roll_forward" if mode == "create_only" else "rollback_before_commit"
            ),
        ),
        payload_root,
    )


def test_multi_root_transaction_publishes_local_artifacts_without_cross_device_rename(
    tmp_path: Path,
) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path)
    try:
        result = publish_artifact_transaction(plan)

        assert result.cleanup_complete
        assert plan.artifacts[0].target.read_text() == "new report"
        assert (plan.artifacts[1].target / "part.txt").read_text() == "new payload"
        assert not transaction_journal_path(plan).exists()
    finally:
        shutil.rmtree(payload_root)


def test_multi_root_recovery_rolls_forward_after_a_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first local publication")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    try:
        with pytest.raises(InjectedInterruption):
            publish_artifact_transaction(plan)
        monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

        result = recover_artifact_transaction(plan)

        assert result.cleanup_complete
        assert plan.artifacts[0].target.read_text() == "new report"
        assert (plan.artifacts[1].target / "part.txt").read_text() == "new payload"
    finally:
        shutil.rmtree(payload_root)


def test_multi_root_recovery_accepts_reordered_declared_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path)
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def rename_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first local publication")

    monkeypatch.setattr(transactions, "rename_without_overwrite", rename_then_interrupt)
    try:
        with pytest.raises(InjectedInterruption):
            publish_artifact_transaction(plan)
        monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)
        reordered = ArtifactTransactionPlan(
            transaction_id=plan.transaction_id,
            lease_root=plan.lease_root,
            artifact_roots=tuple(reversed(plan.artifact_roots)),
            artifacts=plan.artifacts,
            mode=plan.mode,
            recovery_policy=plan.recovery_policy,
        )

        result = recover_artifact_transaction(reordered)

        assert result.cleanup_complete
        assert reordered.artifacts[0].target.read_text() == "new report"
        assert (reordered.artifacts[1].target / "part.txt").read_text() == "new payload"
    finally:
        shutil.rmtree(payload_root)


def test_multi_root_replacement_recovery_restores_prior_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path, mode="replace")
    original_rename = transactions.rename_without_overwrite
    interrupted = False

    def publish_then_interrupt(
        source: Path, destination: Path, *, lease_root: Path | None = None
    ) -> None:
        nonlocal interrupted
        original_rename(source, destination, lease_root=lease_root)
        if destination == plan.artifacts[0].target and not interrupted:
            interrupted = True
            raise InjectedInterruption("after first local replacement publication")

    monkeypatch.setattr(transactions, "rename_without_overwrite", publish_then_interrupt)
    try:
        with pytest.raises(InjectedInterruption):
            publish_artifact_transaction(plan)
        monkeypatch.setattr(transactions, "rename_without_overwrite", original_rename)

        result = recover_artifact_transaction(plan)

        assert result.cleanup_complete
        assert result.restored_targets == (plan.artifacts[0].target,)
        assert plan.artifacts[0].target.read_text() == "old report"
        assert (plan.artifacts[1].target / "old.txt").read_text() == "old payload"
        for artifact in plan.artifacts:
            assert not transactions.transaction_backup_path(plan, artifact).exists()
    finally:
        shutil.rmtree(payload_root)


def test_multi_root_lease_failure_closes_previously_opened_root_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path)
    original_open = transactions.open_absolute_directory_without_symlinks
    descriptors: list[int] = []

    def open_then_fail(path: Path) -> int:
        if path == payload_root:
            raise OSError("synthetic second-root failure")
        descriptor = original_open(path)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(transactions, "open_absolute_directory_without_symlinks", open_then_fail)
    try:
        with pytest.raises(OSError, match="second-root failure"):
            claim_artifact_transaction_leases(plan)
        assert descriptors
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        shutil.rmtree(payload_root)


def test_schema_v1_journal_recovers_a_one_root_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(tmp_path)
    original_create = transactions.create_transaction_journal

    def create_legacy_journal_then_interrupt(
        path: Path, payload: dict[str, object]
    ) -> transactions._JournalHandle:
        legacy_payload = dict(payload)
        legacy_payload["version"] = 1
        del legacy_payload["lease_roots"]
        original_create(path, legacy_payload)
        raise InjectedInterruption("after schema-v1 journal creation")

    monkeypatch.setattr(
        transactions, "create_transaction_journal", create_legacy_journal_then_interrupt
    )
    with pytest.raises(InjectedInterruption):
        publish_artifact_transaction(plan)
    monkeypatch.setattr(transactions, "create_transaction_journal", original_create)

    result = recover_artifact_transaction(plan)

    assert result.cleanup_complete
    assert plan.artifacts[0].target.read_text() == "new report"
    assert (plan.artifacts[1].target / "nested" / "part.txt").read_text() == "new payload"


def test_multi_root_plan_rejects_an_undeclared_artifact_root(tmp_path: Path) -> None:
    plan, payload_root = create_multi_root_plan(tmp_path)
    outside_root = tmp_path / "outside-root"
    outside_root.mkdir()
    outside_stage = outside_root / ".mammoth-txn-multi-root-generation-report.stage"
    outside_stage.write_text("outside")
    invalid = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifact_roots=plan.artifact_roots,
        artifacts=(
            TransactionArtifact("report", outside_stage, outside_root / "report.json", "file"),
            plan.artifacts[1],
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )
    try:
        with pytest.raises(ArtifactTransactionValidationError, match="declared artifact root"):
            seal_artifact_transaction(invalid)
    finally:
        shutil.rmtree(payload_root)


def test_public_lease_api_rejects_a_target_outside_its_lease_root(tmp_path: Path) -> None:
    plan = create_plan(tmp_path)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    invalid = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=plan.lease_root,
        artifacts=(
            TransactionArtifact(
                "report",
                outside_root / ".mammoth-txn-generation-1-report.stage",
                outside_root / "report.json",
                "file",
            ),
            plan.artifacts[1],
        ),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
    )

    with pytest.raises(ArtifactTransactionValidationError, match="beneath lease_root"):
        claim_artifact_transaction_leases(invalid)


def test_plan_rejects_existing_create_only_target_without_overwriting(tmp_path: Path) -> None:
    plan = create_plan(tmp_path)
    plan.artifacts[0].target.write_text("unrelated")

    with pytest.raises(FileExistsError, match="already exists"):
        publish_artifact_transaction(plan)

    assert plan.artifacts[0].target.read_text() == "unrelated"
    assert plan.artifacts[0].stage.read_text() == "new report"
    assert not transaction_journal_path(plan).exists()


def test_plan_rejects_symlink_special_tree_and_nested_targets(tmp_path: Path) -> None:
    plan = create_plan(tmp_path)
    plan.artifacts[0].stage.unlink()
    plan.artifacts[0].stage.symlink_to(plan.lease_root / "elsewhere")
    with pytest.raises(ArtifactTransactionValidationError, match="symlink"):
        seal_artifact_transaction(plan)

    tree_plan = create_plan(tmp_path / "tree")
    fifo = tree_plan.artifacts[1].stage / "stream"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactTransactionValidationError, match="special"):
        seal_artifact_transaction(tree_plan)

    replacement = create_plan(tmp_path / "other")
    parent_target = replacement.lease_root / "nested-parent"
    parent_target.mkdir()
    parent_stage = replacement.lease_root / ".mammoth-txn-generation-1-parent.stage"
    parent_stage.mkdir()
    child_target = parent_target / "child.json"
    child_stage = parent_target / ".mammoth-txn-generation-1-child.stage"
    child_stage.write_text("child")
    nested = ArtifactTransactionPlan(
        transaction_id=replacement.transaction_id,
        lease_root=replacement.lease_root,
        artifacts=(
            TransactionArtifact(
                "parent",
                parent_stage,
                parent_target,
                "directory",
            ),
            TransactionArtifact(
                "child",
                child_stage,
                child_target,
                "file",
            ),
        ),
        mode=replacement.mode,
        recovery_policy=replacement.recovery_policy,
    )
    with pytest.raises(ArtifactTransactionValidationError, match="must not overlap"):
        seal_artifact_transaction(nested)
