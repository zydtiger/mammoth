"""Tests for framework-neutral retireable lease namespaces."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

import mammoth.core.leases as lease_module
from mammoth.core import (
    LeaseNamespaceConflictError,
    LeaseNamespaceError,
    LeaseNamespaceRecoveryError,
    claim_lease_namespace,
    reconcile_lease_namespace,
)


def _paused_claim(path: str, connection: Any) -> None:
    """Pause a contender after opening the old lock inode but before flock."""
    original_open = lease_module._open_lock

    def pause_after_open(directory_descriptor: int, namespace_path: Path) -> int:
        descriptor = original_open(directory_descriptor, namespace_path)
        connection.send("opened")
        connection.recv()
        return descriptor

    lease_module._open_lock = pause_after_open
    try:
        with claim_lease_namespace(Path(path)):
            connection.send("acquired")
    except BaseException as error:
        connection.send(type(error).__name__)
    finally:
        connection.close()


def _simultaneous_fresh_claim(path: str, start: Any, release: Any, outcomes: Any) -> None:
    """Race first creation while keeping the winner held for conflict classification."""
    start.wait()
    try:
        lease = claim_lease_namespace(Path(path))
    except BaseException as error:
        outcomes.put(type(error).__name__)
        return
    outcomes.put("acquired")
    release.wait()
    lease.close()


def test_close_preserves_generation_and_retire_removes_everything(tmp_path: Path) -> None:
    namespace = tmp_path / ".staging" / "leases"
    parent = namespace.parent
    first = claim_lease_namespace(namespace, owned_parents=(parent,))
    generation = first.generation
    first.close()

    assert namespace.is_dir()
    second = claim_lease_namespace(namespace, owned_parents=(parent,))
    assert second.generation == generation
    second.retire()

    assert not namespace.exists()
    assert not lease_module._retired_path(namespace).exists()
    assert not parent.exists()


def test_nonblocking_contention_rejects_a_second_owner(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    first = claim_lease_namespace(namespace)
    try:
        with pytest.raises(LeaseNamespaceConflictError):
            claim_lease_namespace(namespace)
    finally:
        first.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_simultaneous_first_claim_classifies_creation_race_as_conflict(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / ".shared" / "leases"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_simultaneous_fresh_claim,
            args=(str(namespace), start, release, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    observed = sorted(outcomes.get(timeout=10) for _ in processes)
    release.set()
    for process in processes:
        process.join(timeout=10)

    assert observed == ["LeaseNamespaceConflictError", "acquired"]
    assert all(process.exitcode == 0 for process in processes)


def test_creation_handoff_disappearance_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    original_create = lease_module._create_namespace_if_absent
    attempts = 0

    def disappear_once(path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("winner promoted creation scratch")
        original_create(path)

    monkeypatch.setattr(lease_module, "_create_namespace_if_absent", disappear_once)
    lease = claim_lease_namespace(namespace)
    lease.retire()

    assert attempts == 2


def test_retired_cleanup_handoff_disappearance_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    original_reconcile = lease_module._reconcile_retired_only
    attempts = 0

    def disappear_once(path: Path, parents: tuple[Path, ...]) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("another cleaner completed retirement")
        return original_reconcile(path, parents)

    monkeypatch.setattr(lease_module, "_reconcile_retired_only", disappear_once)
    assert reconcile_lease_namespace(namespace) == "absent"
    assert attempts == 2


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_paused_open_cannot_adopt_retired_generation(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    owner = claim_lease_namespace(namespace)
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_paused_claim, args=(str(namespace), child_connection))
    process.start()
    assert parent_connection.recv() == "opened"

    owner.retire()
    replacement = claim_lease_namespace(namespace)
    parent_connection.send("continue")
    outcome = parent_connection.recv()
    process.join(timeout=10)
    replacement.retire()

    assert outcome == "LeaseNamespaceRecoveryError"
    assert process.exitcode == 0


def test_terminal_canonical_generation_is_reclaimed_before_reacquisition(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "leases"
    first = claim_lease_namespace(namespace)
    generation = first.generation
    lease_module._write_terminal_marker(first)
    first.close()

    second = claim_lease_namespace(namespace)
    try:
        assert second.generation != generation
    finally:
        second.retire()


def test_reconcile_terminal_canonical_does_not_create_replacement(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(lease)
    lease.close()

    assert reconcile_lease_namespace(namespace) == "reclaimed"
    assert not namespace.exists()
    assert not lease_module._retired_path(namespace).exists()
    assert not lease_module._retired_proof_path(namespace).exists()


def test_retired_crash_remnant_is_reclaimed_idempotently(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(lease)
    retired = lease_module._retired_path(namespace)
    os.rename(namespace, retired)
    lease.close()

    assert reconcile_lease_namespace(namespace) == "reclaimed"
    assert reconcile_lease_namespace(namespace) == "absent"


def test_retired_generation_is_not_reclaimed_until_owner_releases(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    owner = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(owner)
    retired = lease_module._retired_path(namespace)
    os.rename(namespace, retired)

    with pytest.raises(LeaseNamespaceConflictError, match="still owned"):
        claim_lease_namespace(namespace)

    owner.close()
    replacement = claim_lease_namespace(namespace)
    replacement.retire()


def test_retirement_proof_stays_locked_until_physical_cleanup_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    owner = claim_lease_namespace(namespace)
    proof = lease_module._retired_proof_path(namespace)
    paused = threading.Event()
    resume = threading.Event()
    failure: list[BaseException] = []
    original_unlink = lease_module.os.unlink

    def pause_before_proof_unlink(path: str | bytes, *args: Any, **kwargs: Any) -> None:
        if path == proof.name and threading.current_thread().name == "terminalizer":
            paused.set()
            assert resume.wait(timeout=10)
        original_unlink(path, *args, **kwargs)

    def terminalize() -> None:
        try:
            owner.retire()
        except BaseException as error:
            failure.append(error)

    monkeypatch.setattr(lease_module.os, "unlink", pause_before_proof_unlink)
    thread = threading.Thread(target=terminalize, name="terminalizer")
    thread.start()
    assert paused.wait(timeout=10)

    with pytest.raises(LeaseNamespaceConflictError, match="proof is still owned"):
        claim_lease_namespace(namespace)

    resume.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not failure
    replacement = claim_lease_namespace(namespace)
    replacement.retire()


def test_rename_to_release_handoff_has_no_unlocked_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    owner = claim_lease_namespace(namespace)
    paused = threading.Event()
    resume = threading.Event()
    failure: list[BaseException] = []
    original_close = lease_module.RetireableLeaseNamespace.close

    def pause_after_release(value: Any) -> None:
        original_close(value)
        if threading.current_thread().name == "terminalizer":
            paused.set()
            assert resume.wait(timeout=10)

    def terminalize() -> None:
        try:
            owner.retire()
        except BaseException as error:
            failure.append(error)

    monkeypatch.setattr(lease_module.RetireableLeaseNamespace, "close", pause_after_release)
    thread = threading.Thread(target=terminalize, name="terminalizer")
    thread.start()
    assert paused.wait(timeout=10)

    with pytest.raises(LeaseNamespaceConflictError, match="proof is still owned"):
        claim_lease_namespace(namespace)

    resume.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not failure
    replacement = claim_lease_namespace(namespace)
    replacement.retire()


def test_active_plus_retired_state_is_preserved_as_ambiguous(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(lease)
    retired = lease_module._retired_path(namespace)
    os.rename(namespace, retired)
    lease.close()
    namespace.mkdir(mode=0o700)

    with pytest.raises(LeaseNamespaceRecoveryError, match="ambiguous"):
        reconcile_lease_namespace(namespace)
    assert namespace.exists()
    assert retired.exists()


def test_unknown_retired_child_is_preserved(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(lease)
    retired = lease_module._retired_path(namespace)
    os.rename(namespace, retired)
    lease.close()
    (retired / "unknown").write_text("evidence")

    with pytest.raises(LeaseNamespaceRecoveryError, match="unknown children"):
        reconcile_lease_namespace(namespace)
    assert (retired / "unknown").read_text() == "evidence"


def test_symlinked_namespace_is_rejected_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    namespace = tmp_path / "leases"
    namespace.symlink_to(target, target_is_directory=True)

    with pytest.raises(LeaseNamespaceError):
        claim_lease_namespace(namespace)
    assert target.is_dir()


def test_intermediate_symlink_is_rejected_without_creating_below_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        claim_lease_namespace(linked_parent / "nested" / "leases")

    assert not (outside / "nested").exists()


@pytest.mark.parametrize("remaining", ["terminal-metadata", "terminal", "proof", "proof-only"])
def test_each_partial_retirement_state_is_reclaimed(tmp_path: Path, remaining: str) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lease_module._write_terminal_marker(lease)
    retired = lease_module._retired_path(namespace)
    proof = lease_module._retired_proof_path(namespace)
    os.rename(namespace, retired)
    lease.close()

    (retired / lease_module._LOCK_NAME).unlink()
    if remaining in {"terminal", "proof", "proof-only"}:
        (retired / lease_module._METADATA_NAME).unlink()
    if remaining in {"proof", "proof-only"}:
        os.rename(retired / lease_module._TERMINAL_NAME, proof)
    if remaining == "proof-only":
        retired.rmdir()

    assert reconcile_lease_namespace(namespace) == "reclaimed"
    assert not retired.exists()
    assert not proof.exists()


def test_unauthenticated_empty_retired_directory_is_preserved(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    retired = lease_module._retired_path(namespace)
    retired.mkdir(mode=0o700)

    with pytest.raises(LeaseNamespaceRecoveryError, match="terminal authenticator"):
        reconcile_lease_namespace(namespace)

    assert retired.is_dir()


@pytest.mark.parametrize("child", [None, lease_module._METADATA_NAME, lease_module._LOCK_NAME])
def test_partial_creation_scratch_is_reinitialized(tmp_path: Path, child: str | None) -> None:
    namespace = tmp_path / "leases"
    creating = lease_module._creating_path(namespace)
    creating.mkdir(mode=0o700)
    if child is not None:
        (creating / child).write_text("partial")
        (creating / child).chmod(0o600)

    lease = claim_lease_namespace(namespace)
    try:
        assert namespace.is_dir()
        assert not creating.exists()
    finally:
        lease.retire()


def test_platform_without_no_follow_support_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW")

    with pytest.raises(NotImplementedError, match="O_NOFOLLOW"):
        claim_lease_namespace(tmp_path / "leases")


def test_absent_reconciliation_finishes_owned_parent_cleanup(tmp_path: Path) -> None:
    parent = tmp_path / ".staging"
    parent.mkdir(mode=0o700)
    namespace = parent / "leases"

    assert reconcile_lease_namespace(namespace, owned_parents=(parent,)) == "absent"
    assert not parent.exists()


def test_creation_interruption_reinitializes_authenticated_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    original_write = lease_module._write_new_json_at

    def interrupt_after_metadata(descriptor: int, name: str, payload: dict[str, Any]) -> None:
        original_write(descriptor, name, payload)
        if name == lease_module._METADATA_NAME:
            raise RuntimeError("creation interrupted")

    monkeypatch.setattr(lease_module, "_write_new_json_at", interrupt_after_metadata)
    with pytest.raises(RuntimeError, match="creation interrupted"):
        claim_lease_namespace(namespace)
    monkeypatch.setattr(lease_module, "_write_new_json_at", original_write)

    lease = claim_lease_namespace(namespace)
    lease.retire()
    assert not lease_module._creating_path(namespace).exists()


def test_terminal_marker_interruption_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    original_marker = lease_module._write_terminal_marker

    def interrupt_after_marker(value: Any) -> None:
        original_marker(value)
        raise RuntimeError("marker interrupted")

    monkeypatch.setattr(lease_module, "_write_terminal_marker", interrupt_after_marker)
    with pytest.raises(RuntimeError, match="marker interrupted"):
        lease.terminalize()
    lease.close()
    monkeypatch.setattr(lease_module, "_write_terminal_marker", original_marker)

    assert reconcile_lease_namespace(namespace) == "reclaimed"


def test_retirement_rename_interruption_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    original_rename = lease_module._rename_no_replace_at

    def interrupt_after_rename(*arguments: Any) -> None:
        original_rename(*arguments)
        if arguments[-1] == lease_module._retired_path(namespace).name:
            raise OSError("rename interrupted")

    monkeypatch.setattr(lease_module, "_rename_no_replace_at", interrupt_after_rename)
    with pytest.raises(LeaseNamespaceRecoveryError, match="atomically retired"):
        lease.terminalize()
    lease.close()
    monkeypatch.setattr(lease_module, "_rename_no_replace_at", original_rename)

    assert reconcile_lease_namespace(namespace) == "reclaimed"


def test_descriptor_release_interruption_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    original_close = lease_module.RetireableLeaseNamespace.close

    def interrupt_after_close(value: Any) -> None:
        original_close(value)
        raise RuntimeError("release interrupted")

    monkeypatch.setattr(lease_module.RetireableLeaseNamespace, "close", interrupt_after_close)
    with pytest.raises(RuntimeError, match="release interrupted"):
        lease.terminalize()
    monkeypatch.setattr(lease_module.RetireableLeaseNamespace, "close", original_close)

    assert reconcile_lease_namespace(namespace) == "reclaimed"


def test_parent_cleanup_interruption_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / ".staging"
    namespace = parent / "leases"
    lease = claim_lease_namespace(namespace, owned_parents=(parent,))
    original_cleanup = lease_module._remove_empty_owned_parents

    def interrupt_cleanup(_parents: tuple[Path, ...]) -> None:
        raise RuntimeError("parent cleanup interrupted")

    monkeypatch.setattr(lease_module, "_remove_empty_owned_parents", interrupt_cleanup)
    with pytest.raises(RuntimeError, match="parent cleanup interrupted"):
        lease.terminalize()
    monkeypatch.setattr(lease_module, "_remove_empty_owned_parents", original_cleanup)

    assert reconcile_lease_namespace(namespace, owned_parents=(parent,)) == "absent"
    assert not parent.exists()


def test_unsafe_namespace_permissions_are_rejected(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    namespace.mkdir(mode=0o777)
    namespace.chmod(0o777)

    with pytest.raises(LeaseNamespaceError, match="safely accessible"):
        claim_lease_namespace(namespace)


def test_lock_inode_substitution_is_detected(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    lock_path = namespace / lease_module._LOCK_NAME
    lock_path.rename(namespace / "displaced-lock")
    lock_path.touch(mode=0o600)

    with pytest.raises(LeaseNamespaceRecoveryError, match="Lease inode changed"):
        lease.terminalize()
    lease.close()


def test_generation_substitution_is_detected(tmp_path: Path) -> None:
    namespace = tmp_path / "leases"
    lease = claim_lease_namespace(namespace)
    metadata_path = namespace / lease_module._METADATA_NAME
    payload = json.loads(metadata_path.read_text())
    payload["generation"] = "0" * 32
    metadata_path.write_text(json.dumps(payload))
    metadata_path.chmod(0o600)

    with pytest.raises(LeaseNamespaceRecoveryError, match="generation metadata changed"):
        lease.terminalize()
    lease.close()
