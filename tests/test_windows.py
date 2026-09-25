"""Native handle and crash-recovery checks for Windows direct execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mammoth.core._filesystem import windows as _windows
from mammoth.core._filesystem.leases import WINDOWS_RUN_LOCK
from mammoth.core.artifacts import (
    atomic_write_json,
    discard_prepared_artifact,
    open_artifact_session,
    prepare_artifact,
    publish_prepared_artifact,
)
from mammoth.core.execution import claim_logical_run_lease, is_immutable_log_entry
from mammoth.torch import CheckpointArtifact, CheckpointPlan
from mammoth.torch.checkpoint import (
    anchor_checkpoint_plan,
    publish_anchored_checkpoint_plan,
    validate_checkpoint_plan,
)

pytestmark = [pytest.mark.windows, pytest.mark.skipif(os.name != "nt", reason="native Windows")]


def test_imports_and_cli_do_not_require_fcntl() -> None:
    code = """
import sys
sys.modules['fcntl'] = None
import mammoth.core.artifacts
import mammoth.logging
import mammoth.execution
import mammoth.torch
from mammoth.cli import app
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_parent_is_pinned_until_publication_or_discard(tmp_path: Path) -> None:
    parent = tmp_path / "output"
    parent.mkdir()
    target = parent / "weights.pt"

    def writer(path):
        with pytest.raises(PermissionError):
            parent.rename(tmp_path / "moved")
        path.write_bytes(b"weights")

    prepared = prepare_artifact(target, writer)
    with pytest.raises(PermissionError):
        parent.rename(tmp_path / "moved")
    publish_prepared_artifact(prepared)
    parent.rename(tmp_path / "moved")
    assert (tmp_path / "moved" / "weights.pt").read_bytes() == b"weights"


def test_read_only_staging_is_discardable(tmp_path: Path) -> None:
    target = tmp_path / "weights.pt"
    prepared = prepare_artifact(target, lambda p: p.write_bytes(b"weights"), mode=0o400)
    discard_prepared_artifact(prepared)
    assert not list(tmp_path.iterdir())


def test_junctions_cannot_redirect_prepared_writes(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert created.returncode == 0, created.stderr
    try:
        with pytest.raises(ValueError, match="reparse"):
            prepare_artifact(link / "weights.pt", lambda p: p.write_bytes(b"wrong"))
        assert not list(outside.iterdir())
    finally:
        link.rmdir()


def test_artifact_read_rejects_file_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    link = tmp_path / "link"
    try:
        link.symlink_to(source)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip(
                "Windows account lacks symlink privilege; junction rejection is still tested"
            )
        raise
    with pytest.raises(ValueError, match="symlink"), open_artifact_session(link):
        pass


def test_checkpoint_queue_rejects_a_replaced_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    plan = validate_checkpoint_plan(
        CheckpointPlan(
            root,
            (CheckpointArtifact(root / "weights.pt", lambda p: p.write_bytes(b"weights")),),
        )
    )
    anchored = anchor_checkpoint_plan(plan)
    root.rename(tmp_path / "previous")
    root.mkdir()
    with pytest.raises(RuntimeError, match="root changed"):
        publish_anchored_checkpoint_plan(anchored)
    assert not list(root.iterdir())


def test_file_validation_interruption_closes_exclusive_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lock"

    def fail(descriptor):
        raise KeyboardInterrupt("interrupted validation")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", fail)
        with pytest.raises(KeyboardInterrupt, match="interrupted validation"):
            _windows.open_file(path, write=True, create=True, share=0)
    descriptor = _windows.open_file(path, write=True, share=0)
    os.close(descriptor)


def test_retired_lease_recovery_keeps_the_coordinator(tmp_path: Path) -> None:
    lease = claim_logical_run_lease(tmp_path)
    namespace = tmp_path / "logs" / ".mammoth-leases" / "logical-run"
    metadata = json.loads((namespace / ".mammoth-windows-lease.json").read_text())
    atomic_write_json(namespace / ".mammoth-windows-terminal.json", metadata)
    namespace.rename(namespace.with_name(".logical-run.mammoth-retired"))
    lease.close()  # Simulate exit between retirement and cleanup.
    with claim_logical_run_lease(tmp_path) as recovered:
        recovered.retire()
    coordinator = tmp_path / "logs" / WINDOWS_RUN_LOCK
    assert coordinator.is_file() and coordinator.stat().st_size == 0
    assert is_immutable_log_entry(coordinator.parent, coordinator)
    assert not namespace.parent.exists()


def test_unknown_lease_content_is_preserved_and_ownership_released(tmp_path: Path) -> None:
    lease = claim_logical_run_lease(tmp_path)
    namespace = tmp_path / "logs" / ".mammoth-leases" / "logical-run"
    unknown = namespace / "user-data"
    unknown.write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="Unknown"):
        lease.retire()
    assert unknown.read_bytes() == b"preserve"
    with pytest.raises(RuntimeError, match="not safely acquirable"):
        claim_logical_run_lease(tmp_path)


def test_publication_replaces_read_only_target_and_preserves_mode(tmp_path: Path) -> None:
    target = tmp_path / "weights.pt"
    first = prepare_artifact(target, lambda p: p.write_bytes(b"first"), mode=0o400)
    publish_prepared_artifact(first)
    try:
        second = prepare_artifact(target, lambda p: p.write_bytes(b"second"))
        assert second.temporary.read_bytes() == b"second"
        publish_prepared_artifact(second)
        assert target.read_bytes() == b"second", list(tmp_path.rglob("*"))
        assert target.stat().st_file_attributes & 1
    finally:
        target.chmod(0o600)


def test_failed_read_only_replacement_leaves_previous_target_intact(tmp_path: Path) -> None:
    target = tmp_path / "weights.pt"
    first = prepare_artifact(target, lambda p: p.write_bytes(b"first"), mode=0o400)
    publish_prepared_artifact(first)
    second = prepare_artifact(target, lambda p: p.write_bytes(b"second"))
    descriptor = _windows.open_file(target, share=3)  # A reader that refuses delete sharing.
    try:
        with pytest.raises(PermissionError):
            publish_prepared_artifact(second)
        assert target.read_bytes() == b"first"
        assert target.stat().st_file_attributes & 1
    finally:
        os.close(descriptor)
        discard_prepared_artifact(second)
        target.chmod(0o600)


def test_file_handles_are_not_inheritable(tmp_path: Path) -> None:
    descriptor = _windows.open_file(tmp_path / "file", write=True, create=True)
    try:
        assert not os.get_inheritable(descriptor)
    finally:
        os.close(descriptor)
