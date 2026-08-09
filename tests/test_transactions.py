"""Focused coverage for durable core multi-artifact publication transactions."""

from __future__ import annotations

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
    claim_artifact_transaction_leases,
    publish_artifact_transaction,
    recover_artifact_transaction,
    seal_artifact_transaction,
    transaction_journal_path,
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
