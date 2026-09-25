"""Shared Linux/Windows acceptance tests for the supported direct-training path."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mammoth.core import claim_logical_run_lease, read_execution_events
from mammoth.core.artifacts import (
    ArtifactChangedError,
    atomic_write_json,
    discard_prepared_artifact,
    inspect_artifact,
    open_artifact_session,
    prepare_artifact,
    publish_prepared_artifact,
)
from mammoth.execution import ExecutionSession, ExecutionSpec
from mammoth.logging.tensorboard import TensorBoardSink
from mammoth.torch import (
    CheckpointArtifact,
    CheckpointInspection,
    CheckpointPlan,
    CheckpointSavePolicy,
    RuntimeConfig,
    StepOutput,
    Trainer,
    TrainerCheckpointRestore,
    TrainerCheckpointWriters,
    TrainerConfig,
    initialize_runtime,
)
from mammoth.torch.checkpoint import publish_checkpoint_plan


class TrainingPolicy:
    """Exercise consumer-owned serialization and descriptor-bound restoration."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def inspect(self, path):
        with open_artifact_session(path) as artifact, artifact.open_reader() as reader:
            torch.load(reader, weights_only=True)
        return CheckpointInspection(frozenset({"model", "optimizer", "trainer"}))

    def capture(self, context):
        payload = deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": context.epoch,
                "global_step": context.global_step,
                "optimizer_step": context.optimizer_step,
            }
        )
        return TrainerCheckpointWriters(resumable=lambda path: torch.save(payload, path))

    def restore(self, path, *, device, options):
        with open_artifact_session(path) as artifact, artifact.open_reader() as reader:
            payload = torch.load(reader, weights_only=True, map_location=device)
        self.model.load_state_dict(payload["model"])
        return TrainerCheckpointRestore(
            epoch=payload["epoch"],
            global_step=payload["global_step"],
            optimizer_step=payload["optimizer_step"],
            optimizer_state_dict=payload["optimizer"],
            restored_components=frozenset({"model"}),
        )


def _step(model, batch, context):
    loss = torch.nn.functional.mse_loss(model(batch[0]), batch[1])
    return StepOutput(loss=loss, metrics={"mse": loss.detach()})


def _validation_step(model, batch, context):
    loss = torch.nn.functional.mse_loss(model(batch[0]), batch[1])
    return StepOutput(metrics={"mse": loss})


def test_direct_training_resume_logs_and_retention(tmp_path: Path) -> None:
    run = tmp_path / "training"
    checkpoints = run / "checkpoints"
    loader = DataLoader(TensorDataset(torch.ones(5, 1), torch.zeros(5, 1)), batch_size=2)
    completed_steps = 0
    for attempt in range(2):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        spec = ExecutionSpec(
            run, "training", "train", ("train",), execution_id=f"attempt-{attempt}"
        )
        tb_path = run / "tensorboard" / spec.execution_id
        with ExecutionSession.create(spec, additional_sinks=(TensorBoardSink(tb_path),)) as session:
            with (
                session.phase_scope("train"),
                Trainer(
                    model=model,
                    optimizer=optimizer,
                    train_loader=loader,
                    train_step=_step,
                    validation_loader=loader,
                    validation_step=_validation_step,
                    observer=session.observer,
                    config=TrainerConfig(
                        epochs=attempt + 1,
                        device="cpu",
                        gradient_accumulation_steps=2,
                        emit_fit_phase_events=False,
                    ),
                    checkpoint_dir=checkpoints,
                    checkpoint_policy=TrainingPolicy(model, optimizer),
                    checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
                ) as trainer,
            ):
                if attempt:
                    trainer.load_checkpoint(checkpoints / "latest_epoch_0.pt")
                    assert trainer.state.global_step == completed_steps
                    assert optimizer.state
                result = trainer.fit()
                completed_steps = result.state.global_step
            assert session.observer.disabled_sink_count == 0
        assert list(tb_path.glob("events.*"))
        records = read_execution_events(session.context.execution_dir / "rank-0.jsonl")
        assert {event.phase for event in records if event.event == "progress"} == {
            "train",
            "validation",
        }
        assert records[0].event == "process_started"
        assert records[-1].event == "process_completed" and records[-1].exit_code == 0
        raw = [
            json.loads(line)
            for line in (session.context.execution_dir / "rank-0.jsonl").read_text().splitlines()
        ]
        receipts = [item for record in raw for item in record.get("checkpoints", [])]
        checkpoint = checkpoints / f"latest_epoch_{attempt}.pt"
        assert receipts[-1]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert receipts[-1]["size_bytes"] == checkpoint.stat().st_size
        if attempt:
            assert not (checkpoints / "latest_epoch_0.pt").exists()
    assert completed_steps == 6  # Three microbatches per epoch, two logical batches.


def test_artifact_binary_receipt_and_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "binary.pt"
    payload = bytes(range(256)) + b"\r\n\x1a\n"
    path.write_bytes(payload)
    with open_artifact_session(path) as artifact:
        assert artifact.receipt.sha256 == hashlib.sha256(payload).hexdigest()
        with artifact.open_reader() as reader:
            assert reader.read() == payload
    with pytest.raises(ArtifactChangedError), open_artifact_session(path):
        path.write_bytes(b"changed")
    assert inspect_artifact(path).size_bytes == 7


def test_prepared_publication_replaces_or_discards_without_truncation(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"old")
    staged = prepare_artifact(path, lambda p: p.write_bytes(b"new"))
    assert path.read_bytes() == b"old"
    publish_prepared_artifact(staged)
    assert path.read_bytes() == b"new"
    staged = prepare_artifact(path, lambda p: p.write_bytes(b"discard"))
    discard_prepared_artifact(staged)
    discard_prepared_artifact(staged)
    assert path.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [path]
    atomic_write_json(tmp_path / "config.json", {"name": "训练"})
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == {"name": "训练"}


def test_checkpoint_serializer_failure_preserves_previous_artifacts(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    previous = tmp_path / "latest_epoch_0.pt"
    best.write_bytes(b"old best")
    previous.write_bytes(b"previous")

    def fail(path):
        path.write_bytes(b"partial")
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError, match="serialization failed"):
        publish_checkpoint_plan(
            CheckpointPlan(
                tmp_path,
                (
                    CheckpointArtifact(best, lambda p: p.write_bytes(b"new best")),
                    CheckpointArtifact(tmp_path / "latest_epoch_1.pt", fail),
                ),
                retire_after_commit=(previous,),
            )
        )
    assert best.read_bytes() == b"old best"
    assert previous.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "latest_epoch_0.pt"]


def test_run_lock_excludes_other_process_and_recovers_after_exit(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from mammoth.core import claim_logical_run_lease
try:
    lease = claim_logical_run_lease(Path(sys.argv[1]))
except RuntimeError:
    sys.exit(23)
sys.exit(0)
"""
    command = [sys.executable, "-c", script, str(tmp_path)]
    with claim_logical_run_lease(tmp_path):
        competing = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert competing.returncode == 23, competing.stderr
    assert subprocess.run(command, capture_output=True, timeout=30).returncode == 0
    with claim_logical_run_lease(tmp_path) as lease:
        lease.retire()


@pytest.mark.parametrize("exit_kind", ["failure", "interrupt", "killed"])
def test_session_failure_and_process_death_release_ownership(
    tmp_path: Path, exit_kind: str
) -> None:
    script = """
import os, sys
from pathlib import Path
from mammoth.execution import ExecutionSession, ExecutionSpec
spec = ExecutionSpec(Path(sys.argv[1]), "test", "test", ("train",))
with ExecutionSession.create(spec) as session, session.phase_scope("train"):
    if sys.argv[2] == "killed":
        os._exit(17)
    if sys.argv[2] == "interrupt":
        raise KeyboardInterrupt
    raise RuntimeError("failed training")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), exit_kind],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    with claim_logical_run_lease(tmp_path) as lease:
        lease.retire()


def test_text_log_excludes_other_writers_and_flushes_unicode(tmp_path: Path) -> None:
    spec = ExecutionSpec(tmp_path, "test", "test", ("work",))
    with ExecutionSession.create(spec) as session, session.phase_scope("work"):
        handler = session.execution_logging.text_handler
        handler.handle(logging.LogRecord("test", logging.INFO, "", 0, "诊断", (), None))
        code = """
import sys
from pathlib import Path
from mammoth.logging.text import claim_process_text_log
try:
    claim_process_text_log(Path(sys.argv[1]))
except RuntimeError:
    sys.exit(23)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(handler.path)],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 23
    assert "诊断" in handler.path.read_text(encoding="utf-8")


@pytest.mark.parametrize("failed", [False, True])
def test_torch_runtime_session_owns_logs_and_releases_run(tmp_path: Path, failed: bool) -> None:
    spec = ExecutionSpec(tmp_path, "runtime", "train", ("train",), execution_id="runtime-attempt")
    with initialize_runtime(RuntimeConfig(device="cpu")) as runtime:
        context = runtime.create_execution(spec)
        runtime.start_execution_logging()
        session = runtime.create_execution_session()
        with pytest.raises(RuntimeError, match="already active"):
            claim_logical_run_lease(tmp_path)
        if failed:
            with pytest.raises(RuntimeError, match="workload failed"), session:
                session.start_phase("train")
                raise RuntimeError("workload failed")
        else:
            with session:
                session.start_phase("train")
                loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)))
                model = torch.nn.Linear(1, 1)
                trainer = session.create_trainer(
                    model=model,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                    train_loader=loader,
                    train_step=_step,
                    runtime=runtime,
                    config=TrainerConfig(epochs=1, device="cpu", emit_fit_phase_events=False),
                    checkpoint_dir=tmp_path / "checkpoints",
                )
                trainer.fit()
                session.complete_phase()
            assert (tmp_path / "checkpoints" / "checkpoint-0000.pt").is_file()
    events = read_execution_events(context.execution_dir / "rank-0.jsonl")
    assert events[0].event == "process_started"
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == (1 if failed else 0)
    assert any(event.event == ("phase_failed" if failed else "phase_completed") for event in events)
    with claim_logical_run_lease(tmp_path) as recovered:
        recovered.retire()
