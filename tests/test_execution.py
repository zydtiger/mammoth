from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import mammoth.core.execution as execution_module
import mammoth.core.leases as lease_module
from mammoth.core import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    LOGICAL_RUN_LEASE_FILENAME,
    PHASE_ENV,
    RUN_NAME_ENV,
    ExecutionMetadata,
    RunLayout,
    claim_logical_run_lease,
    create_execution_context,
    execution_id_from_environment,
    is_immutable_log_entry,
    join_execution_context,
    latest_execution_id,
)
from mammoth.core.execution import EXECUTION_SCHEMA_VERSION


def create_context(tmp_path: Path, execution_id: str = "attempt-1"):
    layout = RunLayout(tmp_path, "run-one").prepare()
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="unit-test",
        intended_phases=("opaque-phase",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py", "--token=secret", "https://user:pass@example.test/x?q=1"),
        config_reference="https://user:pass@example.test/config.yaml?token=x",
        execution_id=execution_id,
        previous_execution_id="previous-1",
        parent_execution_id="parent-1",
        resume_checkpoint="/safe/checkpoint.pt",
        resume_checkpoint_sha256="a" * 64,
        created_at="2026-01-02T03:04:05Z",
    )


def test_create_and_join_publish_immutable_sanitized_metadata(tmp_path: Path) -> None:
    context = create_context(tmp_path)
    payload = json.loads(context.metadata_path.read_text())

    assert payload["schema_version"] == EXECUTION_SCHEMA_VERSION
    assert payload["execution_id"] == "attempt-1"
    assert payload["previous_execution_id"] == "previous-1"
    assert payload["parent_execution_id"] == "parent-1"
    assert payload["resume_checkpoint_sha256"] == "a" * 64
    assert payload["command"][2] == "--token=<redacted>"
    assert payload["command"][3] == "https://<redacted>@example.test/x"
    assert payload["config_reference"] == "https://<redacted>@example.test/config.yaml"
    assert (
        join_execution_context(
            context.run_dir,
            "attempt-1",
            expected_run_name="run-one",
        )
        == context
    )


def test_runtime_metadata_is_optional_sanitized_schema_v1_data(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "runtime-run").prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
        execution_id="runtime-attempt",
        runtime={
            "framework": "pytorch",
            "strategy": "single",
            "credentials": {"api_token": "secret"},
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        },
    )

    payload = json.loads(context.metadata_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["resume_checkpoint_sha256"] is None
    assert payload["runtime"] == {
        "AWS_ACCESS_KEY_ID": "<redacted>",
        "credentials": "<redacted>",
        "framework": "pytorch",
        "strategy": "single",
    }
    assert (
        join_execution_context(layout.run_dir, "runtime-attempt").metadata.runtime
        == payload["runtime"]
    )


def test_runtime_metadata_is_deeply_immutable_and_to_dict_is_detached(tmp_path: Path) -> None:
    runtime = {"framework": "pytorch", "nested": {"devices": ["cpu"]}}
    layout = RunLayout(tmp_path, "immutable-runtime").prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
        runtime=runtime,
    )
    runtime["framework"] = "changed"
    payload = context.metadata.to_dict()
    payload["runtime"]["nested"]["devices"].append("cuda")

    assert context.metadata.runtime is not None
    assert context.metadata.runtime["framework"] == "pytorch"
    assert context.metadata.runtime["nested"]["devices"] == ("cpu",)
    with pytest.raises(TypeError):
        context.metadata.runtime["framework"] = "changed"  # type: ignore[index]


def test_runtime_metadata_rejects_non_object_payload() -> None:
    payload = {
        "schema_version": 1,
        "run_name": "run",
        "execution_id": "attempt",
        "created_at": "2026-01-02T03:04:05Z",
        "invocation_kind": "test",
        "intended_phases": ["train"],
        "world_size": 1,
        "execution_mode": "single",
        "command": ["python", "job.py"],
        "config_reference": "",
        "runtime": "ddp",
    }

    with pytest.raises(ValueError, match="runtime must be an object"):
        ExecutionMetadata.from_dict(payload)


def test_lineage_is_explicit_and_not_inferred(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run-one").prepare()
    common = {
        "run_name": layout.run_name,
        "invocation_kind": "test",
        "intended_phases": ("phase",),
        "world_size": 1,
        "execution_mode": "single",
        "command": ("python", "job.py"),
    }
    create_execution_context(
        layout.run_dir,
        execution_id="first",
        created_at="2026-01-01T00:00:00Z",
        **common,
    )
    second = create_execution_context(
        layout.run_dir,
        execution_id="second",
        created_at="2026-01-02T00:00:00Z",
        **common,
    )

    assert second.metadata.previous_execution_id is None
    assert second.metadata.parent_execution_id is None
    assert latest_execution_id(layout.run_dir) == "second"


def test_resume_checkpoint_without_explicit_parent_records_no_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume artifact stays independent from execution lineage metadata."""
    layout = RunLayout(tmp_path, "resume-without-parent").prepare()
    checkpoint = layout.run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")

    original_stat = Path.stat
    original_resolve = Path.resolve

    def reject_checkpoint_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == checkpoint:
            pytest.fail("execution lineage must not inspect the resume artifact")
        return original_stat(path, *args, **kwargs)

    def reject_checkpoint_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == checkpoint:
            pytest.fail("execution lineage must not inspect the resume artifact")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", reject_checkpoint_stat)
    monkeypatch.setattr(Path, "resolve", reject_checkpoint_resolve)

    context = create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("python", "job.py"),
        execution_id="resumed",
        resume_checkpoint=checkpoint,
    )

    assert context.metadata.resume_checkpoint == str(checkpoint)
    assert context.metadata.parent_execution_id is None


@pytest.mark.parametrize(
    "digest",
    ("A" * 64, "a" * 63, "a" * 65, "g" * 64, 7),
)
def test_resume_checkpoint_sha256_requires_canonical_lowercase(
    tmp_path: Path,
    digest: object,
) -> None:
    """Persisted resume digests are caller-supplied canonical SHA-256 values."""
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        create_execution_context(
            tmp_path / "invalid-digest",
            run_name="invalid-digest",
            invocation_kind="test",
            intended_phases=("train",),
            world_size=1,
            execution_mode="single",
            command=("python", "train.py"),
            resume_checkpoint="checkpoint.pt",
            resume_checkpoint_sha256=digest,  # type: ignore[arg-type]
        )


def test_metadata_rejects_malformed_resume_checkpoint_sha256(tmp_path: Path) -> None:
    """Deserialization enforces the same digest contract as new publication."""
    payload = create_context(tmp_path).metadata.to_dict()
    payload["resume_checkpoint_sha256"] = "A" * 64

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        ExecutionMetadata.from_dict(payload)


def test_schema_v1_metadata_from_originating_project_remains_readable(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "legacy-run").prepare(project_directories=False)
    execution_dir = layout.execution_dir("legacy-attempt")
    execution_dir.mkdir()
    payload = {
        "schema_version": 1,
        "run_name": "legacy-run",
        "execution_id": "legacy-attempt",
        "created_at": "2025-12-01T12:00:00Z",
        "invocation_kind": "direct",
        "intended_phases": ["train", "validate"],
        "world_size": 2,
        "execution_mode": "distributed",
        "command": ["uv", "run", "tisam", "train"],
        "config_reference": "configs/example.yaml",
        "previous_execution_id": None,
        "resume_checkpoint": None,
        "parent_execution_id": None,
        "starting_epoch": 0,
        "starting_global_step": 0,
    }
    (execution_dir / "execution.json").write_text(json.dumps(payload))

    context = join_execution_context(layout.run_dir, "legacy-attempt")

    assert context.metadata.run_name == "legacy-run"
    assert context.metadata.intended_phases == ("train", "validate")
    assert context.metadata.world_size == 2
    assert context.metadata.resume_checkpoint_sha256 is None


def test_environment_hook_accepts_only_the_canonical_name() -> None:
    assert execution_id_from_environment({"MAMMOTH_EXECUTION_ID": "one"}) == "one"
    assert execution_id_from_environment({"CALLER_EXECUTION_ID": "two"}) is None
    assert execution_id_from_environment({}) is None
    with pytest.raises(ValueError, match="Execution IDs"):
        execution_id_from_environment({"MAMMOTH_EXECUTION_ID": "not/an-id"})


def test_logical_run_lease_rejects_a_second_producer(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "locked").prepare()
    with claim_logical_run_lease(layout.run_dir) as first:
        assert first.path.is_file()
        with pytest.raises(RuntimeError, match="already active"):
            claim_logical_run_lease(layout.run_dir)
    with claim_logical_run_lease(layout.run_dir):
        pass


def test_workflow_child_contract_symbols_are_public_from_core() -> None:
    """`mammoth.core` re-exports the private-module workflow-child contract symbols."""
    assert EXECUTION_ID_ENV == execution_module.EXECUTION_ID_ENV
    assert RUN_NAME_ENV == execution_module.RUN_NAME_ENV
    assert INVOCATION_KIND_ENV == execution_module.INVOCATION_KIND_ENV
    assert PHASE_ENV == execution_module.PHASE_ENV
    assert LOGICAL_RUN_LEASE_FILENAME == execution_module.LOGICAL_RUN_LEASE_FILENAME


def test_logical_run_lease_closes_descriptor_after_lock_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``BaseException`` after flock succeeds cannot retain producer ownership."""
    layout = RunLayout(tmp_path, "interrupted-lock").prepare()
    original_flock = lease_module.fcntl.flock

    def interrupt_after_lock(descriptor: int, operation: int) -> None:
        original_flock(descriptor, operation)
        if operation == lease_module.fcntl.LOCK_EX | lease_module.fcntl.LOCK_NB:
            raise KeyboardInterrupt("lease acquisition interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(lease_module.fcntl, "flock", interrupt_after_lock)
        with pytest.raises(KeyboardInterrupt, match="lease acquisition interrupted"):
            claim_logical_run_lease(layout.run_dir)

    with claim_logical_run_lease(layout.run_dir):
        pass


def test_is_immutable_log_entry_covers_known_immutable_entries(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "immutable-run").prepare()
    log_dir = layout.logs_dir
    lease_container = log_dir / ".mammoth-leases"

    assert is_immutable_log_entry(log_dir, log_dir / "executions") is True
    assert is_immutable_log_entry(log_dir, log_dir / LOGICAL_RUN_LEASE_FILENAME) is True
    assert is_immutable_log_entry(log_dir, lease_container) is True
    assert is_immutable_log_entry(log_dir, lease_container / "logical-run") is True
    assert (
        is_immutable_log_entry(log_dir, lease_container / "logical-run" / "metadata.json") is True
    )
    # Nested entries under the execution-attempt container stay immutable too.
    assert is_immutable_log_entry(log_dir, log_dir / "executions" / "attempt-1") is True
    assert (
        is_immutable_log_entry(log_dir, log_dir / "executions" / "attempt-1" / "execution.json")
        is True
    )


def test_log_reset_preserves_active_logical_run_lease_namespace(tmp_path: Path) -> None:
    """A consumer reset retains the active namespace until terminal retirement."""
    layout = RunLayout(tmp_path, "reset-run").prepare()
    log_dir = layout.logs_dir
    mutable_log = log_dir / "events.out.tfevents.1"
    mutable_log.write_text("tensorboard")
    lease = claim_logical_run_lease(layout.run_dir)

    for child in log_dir.iterdir():
        if not is_immutable_log_entry(log_dir, child):
            child.unlink()

    assert not mutable_log.exists()
    assert (log_dir / ".mammoth-leases" / "logical-run").is_dir()
    lease.retire()

    assert not (log_dir / ".mammoth-leases").exists()


def test_is_immutable_log_entry_rejects_arbitrary_consumer_entries(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "mutable-run").prepare()
    log_dir = layout.logs_dir
    (log_dir / "events.out.tfevents.1").write_text("tensorboard")
    (log_dir / "scratch").mkdir()
    (log_dir / "scratch" / "child.txt").write_text("scratch")
    os.symlink(log_dir / "scratch", log_dir / "scratch-link")

    assert is_immutable_log_entry(log_dir, log_dir / "events.out.tfevents.1") is False
    assert is_immutable_log_entry(log_dir, log_dir / "scratch") is False
    assert is_immutable_log_entry(log_dir, log_dir / "scratch" / "child.txt") is False
    assert is_immutable_log_entry(log_dir, log_dir / "scratch-link") is False
    # An entry named "executions" one level too deep is not the real container.
    assert is_immutable_log_entry(log_dir, log_dir / "scratch" / "executions") is False


def test_is_immutable_log_entry_does_not_require_the_child_to_exist(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "nonexistent-run").prepare()
    log_dir = layout.logs_dir

    assert is_immutable_log_entry(log_dir, log_dir / "executions" / "never-created") is True
    assert is_immutable_log_entry(log_dir, log_dir / "never-created.txt") is False


def test_is_immutable_log_entry_classifies_a_symlinked_immutable_name_by_identity(
    tmp_path: Path,
) -> None:
    """Classification is by path identity within ``log_dir``, not by content."""
    layout = RunLayout(tmp_path, "symlink-run").prepare()
    log_dir = layout.logs_dir
    (log_dir / "executions").rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, log_dir / "executions")

    assert is_immutable_log_entry(log_dir, log_dir / "executions") is True


def test_is_immutable_log_entry_fails_closed_outside_log_dir(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "escape-run").prepare()
    log_dir = layout.logs_dir
    outside = tmp_path / "elsewhere.txt"

    with pytest.raises(ValueError, match="does not lie inside log_dir"):
        is_immutable_log_entry(log_dir, outside)
    with pytest.raises(ValueError, match="does not lie inside log_dir"):
        is_immutable_log_entry(log_dir, log_dir)
    with pytest.raises(ValueError, match="does not lie inside log_dir"):
        is_immutable_log_entry(log_dir, log_dir.parent)
