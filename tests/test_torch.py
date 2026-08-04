from __future__ import annotations

import errno
import multiprocessing
import os
import stat
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from queue import Empty
from typing import Any, cast

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset, TensorDataset

import mammoth.torch.checkpoint as checkpoint_module
import mammoth.torch.runtime as torch_runtime_module
from mammoth.core import (
    PreparedArtifact,
    claim_logical_run_lease,
    create_execution_context,
    publish_prepared_artifact,
    read_execution_events,
)
from mammoth.logging import Observation, RunObserver
from mammoth.torch import (
    AsyncCheckpointPublisher,
    CheckpointArtifact,
    CheckpointPlan,
    CheckpointPublication,
    EarlyStopping,
    MetricAccumulator,
    MetricSpec,
    StateRegistry,
    StepContext,
    StepOutput,
    TorchExecutionRequest,
    TorchRuntimeConfig,
    Trainer,
    TrainerConfig,
    initialize_torch_runtime,
    move_batch_to_device,
    publish_checkpoint_plan,
    restore_checkpoint,
)


class RecordingSink:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def observe(self, observation: Observation) -> None:
        self.observations.append(observation)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class MappingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.inputs = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
        self.targets = 2 * self.inputs + 1

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.inputs[index], "targets": self.targets[index]}


class CounterState:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value = state.get("value")
        if not isinstance(value, int):
            raise ValueError("counter value must be an integer")
        self.value = value


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, value: float | None = None) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        steps = state.get("steps")
        if not isinstance(steps, int):
            raise ValueError("scheduler steps must be an integer")
        self.steps = steps


def classification_step(
    model: torch.nn.Module,
    batch: Any,
    context: StepContext,
) -> StepOutput:
    features, targets = batch
    logits = model(features)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return StepOutput(loss=loss, metrics={"accuracy": accuracy})


def regression_step(
    model: torch.nn.Module,
    batch: Any,
    context: StepContext,
) -> StepOutput:
    prediction = model(batch["features"])
    loss = torch.nn.functional.mse_loss(prediction, batch["targets"])
    return StepOutput(loss=loss, metrics={"mae": (prediction - batch["targets"]).abs().mean()})


def distributed_regression_step(
    model: torch.nn.Module,
    batch: Any,
    context: StepContext,
) -> StepOutput:
    """Return one scalar loss for the CPU DDP runtime integration fixture."""
    features, targets = batch
    prediction = model(features)
    return StepOutput(loss=torch.nn.functional.mse_loss(prediction, targets))


def _torch_runtime_worker(
    rank: int,
    rendezvous: str,
    run_dir: str,
    result_queue: Any,
    fail_logging_rank: int | None,
) -> None:
    """Exercise a real two-process Gloo runtime and report bounded results."""
    try:
        config = TorchRuntimeConfig(
            strategy="ddp",
            device="cpu",
            backend="gloo",
            init_method=rendezvous,
            timeout_seconds=30,
            rank=rank,
            local_rank=rank,
            world_size=2,
        )
        with initialize_torch_runtime(config) as runtime:
            if fail_logging_rank == rank:

                def fail_logging(*args: Any, **kwargs: Any) -> Any:
                    raise PermissionError("rank log unavailable")

                torch_runtime_module.create_execution_logging = fail_logging
            bundle = runtime.start_execution(
                TorchExecutionRequest(
                    run_dir=Path(run_dir),
                    run_name="ddp-run",
                    invocation_kind="test",
                    intended_phases=("train",),
                    command=("python", "train.py"),
                    execution_id="ddp-attempt",
                )
            )
            broadcast = runtime.broadcast_object("ready" if rank == 0 else None)
            gathered = runtime.all_gather_object(rank)
            reduced = runtime.all_reduce_sum(torch.tensor(rank + 1)).item()
            features = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
            targets = 2 * features
            loader = DataLoader(TensorDataset(features, targets), batch_size=2)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            with Trainer(
                model=model,
                optimizer=optimizer,
                train_loader=loader,
                train_step=distributed_regression_step,
                config=TrainerConfig(epochs=1, device="cpu", strategy="ddp"),
                checkpoint_dir=Path(run_dir) / "checkpoints",
                runtime=runtime,
            ) as trainer:
                result = trainer.fit()
            bundle.observer.emit("process_completed", phase="train", exit_code=0)
            runtime.close_process_group()
            result_queue.put(
                (
                    rank,
                    runtime.execution_context.metadata.execution_id,
                    broadcast,
                    gathered,
                    reduced,
                    result.state.global_step,
                    None,
                )
            )
    except BaseException as error:
        result_queue.put((rank, None, None, None, None, None, str(error)))


def run_two_process_runtime(
    tmp_path: Path,
    *,
    fail_logging_rank: int | None = None,
) -> list[tuple[Any, ...]]:
    """Launch the reusable CPU DDP fixture and return one result per rank."""
    process_context = multiprocessing.get_context("spawn")
    result_queue = process_context.Queue()
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    run_dir = str(tmp_path / "ddp-run")
    processes = [
        process_context.Process(
            target=_torch_runtime_worker,
            args=(rank, rendezvous, run_dir, result_queue, fail_logging_rank),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=40)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("CPU DDP runtime worker did not shut down coherently")
        assert process.exitcode == 0
    results: list[tuple[Any, ...]] = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=5))
        except Empty:
            pytest.fail("CPU DDP runtime worker returned no result")
    return sorted(results)


def test_same_trainer_handles_classification_and_mapping_regression(tmp_path: Path) -> None:
    torch.manual_seed(7)
    features = torch.randn(16, 2)
    targets = (features[:, 0] > features[:, 1]).long()
    classification_loader = DataLoader(TensorDataset(features, targets), batch_size=4)
    classifier = torch.nn.Linear(2, 2)
    classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.2)
    sink = RecordingSink()
    classification_config = TrainerConfig(
        epochs=3,
        device="cpu",
        gradient_accumulation_steps=2,
        max_gradient_norm=1.0,
        display_metric_names=("loss", "accuracy"),
    )
    with Trainer(
        model=classifier,
        optimizer=classifier_optimizer,
        train_loader=classification_loader,
        train_step=classification_step,
        config=classification_config,
        observer=RunObserver((sink,)),
        checkpoint_dir=tmp_path / "classification",
    ) as trainer:
        classification_result = trainer.fit()

    regression_loader = DataLoader(MappingDataset(), batch_size=2)
    regressor = torch.nn.Linear(1, 1)
    regression_optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01)
    with Trainer(
        model=regressor,
        optimizer=regression_optimizer,
        train_loader=regression_loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=2, device="cpu", checkpoint_every_epochs=None),
    ) as trainer:
        regression_result = trainer.fit()

    assert classification_result.state.global_step == 12
    assert classification_result.state.optimizer_step == 6
    assert len(classification_result.training_history) == 3
    assert {"loss", "accuracy"}.issubset(classification_result.training_history[-1])
    assert len(list((tmp_path / "classification").glob("checkpoint-*.pt"))) == 3
    assert any(observation.event == "progress" for observation in sink.observations)
    assert len(regression_result.training_history) == 2
    assert {"loss", "mae"}.issubset(regression_result.training_history[-1])


def test_recursive_batch_transfer_preserves_common_container_structure() -> None:
    batch = {
        "tensor": torch.tensor([1]),
        "tuple": (torch.tensor([2]), "opaque"),
        "list": [torch.tensor([3])],
    }

    moved = move_batch_to_device(batch, torch.device("cpu"))

    assert moved["tensor"].device.type == "cpu"
    assert isinstance(moved["tuple"], tuple)
    assert moved["tuple"][1] == "opaque"
    assert isinstance(moved["list"], list)


def test_metric_accumulator_supports_mean_sum_and_last() -> None:
    accumulator = MetricAccumulator(
        {
            "mean": MetricSpec("mean"),
            "sum": MetricSpec("sum"),
            "last": MetricSpec("last"),
        }
    )
    accumulator.update({"mean": 2, "sum": 2, "last": 2}, weight=1)
    accumulator.update({"mean": 4, "sum": 4, "last": 4}, weight=3)

    assert accumulator.compute() == {"last": 4.0, "mean": 3.5, "sum": 14.0}


def test_validation_callback_stops_early_on_project_metric() -> None:
    features = torch.ones(4, 1)
    targets = torch.zeros(4, 1)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def train_step(module: torch.nn.Module, batch: Any, context: StepContext) -> StepOutput:
        prediction = module(batch[0])
        return StepOutput(loss=prediction.sum() * 0)

    def validation_step(module: torch.nn.Module, batch: Any, context: StepContext) -> StepOutput:
        return StepOutput(metrics={"score": 1.0})

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=train_step,
        validation_loader=loader,
        validation_step=validation_step,
        callbacks=(EarlyStopping("score", patience=0),),
        config=TrainerConfig(epochs=5, device="cpu", checkpoint_every_epochs=None),
    ) as trainer:
        result = trainer.fit()

    assert result.state.stopped_early
    assert result.state.epoch == 1
    assert len(result.validation_history) == 2


def test_registered_checkpoint_round_trip_resumes_next_epoch(tmp_path: Path) -> None:
    torch.manual_seed(3)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    counter = CounterState(9)
    checkpoint_dir = tmp_path / "checkpoints"
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=classification_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=checkpoint_dir,
        extra_state={"counter": counter},
    ) as trainer:
        trainer.fit()
    checkpoint = checkpoint_dir / "checkpoint-0000.pt"

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_counter = CounterState()
    with Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        train_loader=loader,
        train_step=classification_step,
        config=TrainerConfig(epochs=2, device="cpu", checkpoint_every_epochs=None),
        extra_state={"counter": restored_counter},
    ) as restored:
        restored.load_checkpoint(checkpoint)
        assert restored.state.epoch == 0
        assert restored_counter.value == 9
        for original, loaded in zip(model.parameters(), restored_model.parameters(), strict=True):
            assert torch.equal(original, loaded)
        result = restored.fit()

    assert result.state.epoch == 1
    assert len(result.training_history) == 1


def test_async_checkpoint_publication_is_atomic_and_bounded(tmp_path: Path) -> None:
    registry = StateRegistry()
    counter = CounterState(4)
    registry.register("counter", counter)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    with AsyncCheckpointPublisher(max_pending=1) as publisher:
        publisher.publish(first, {"schema_version": 1, "state": registry.state_dict()})
        counter.value = 7
        publisher.publish(second, {"schema_version": 1, "state": registry.state_dict()})
        assert publisher.pending_count <= 1

    restored = CounterState()
    restored_registry = StateRegistry()
    restored_registry.register("counter", restored)
    restore_checkpoint(first, restored_registry)
    assert restored.value == 4
    restore_checkpoint(second, restored_registry)
    assert restored.value == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_async_checkpoint_publication_clones_tensors_before_worker_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    real_publish = checkpoint_module.publish_torch_payload

    def blocking_publish(path: Path, payload: Mapping[str, Any]) -> Path:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        return real_publish(path, payload)

    monkeypatch.setattr(checkpoint_module, "publish_torch_payload", blocking_publish)
    live_tensor = torch.tensor([1.0])
    destination = tmp_path / "checkpoint.pt"
    with AsyncCheckpointPublisher() as publisher:
        publisher.publish(destination, {"tensor": live_tensor})
        assert started.wait(timeout=5)
        live_tensor.add_(10)
        release.set()

    saved = torch.load(destination, map_location="cpu", weights_only=False)
    torch.testing.assert_close(saved["tensor"], torch.tensor([1.0]))


def test_checkpoint_plan_prepares_all_artifacts_before_ordered_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    previous = checkpoint_root / "checkpoint-0000.pt"
    previous.write_bytes(b"previous")
    retained = checkpoint_root / "retained.pt"
    retained.write_bytes(b"retained")
    first = checkpoint_root / "best.weights"
    second = checkpoint_root / "checkpoint-0001.pt"
    events: list[str] = []

    def writer(name: str, payload: bytes) -> Callable[[Path], None]:
        def write(temporary: Path) -> None:
            events.append(f"prepare:{name}")
            temporary.write_bytes(payload)

        return write

    def record_publish(artifact: PreparedArtifact) -> Path:
        events.append(f"commit:{artifact.destination.name}")
        return publish_prepared_artifact(artifact)

    monkeypatch.setattr(checkpoint_module, "publish_prepared_artifact", record_publish)
    result = publish_checkpoint_plan(
        CheckpointPlan(
            checkpoint_root=checkpoint_root,
            artifacts=(
                CheckpointArtifact(first, writer("best", b"best")),
                CheckpointArtifact(second, writer("resume", b"resume")),
            ),
            retire_after_commit=(previous,),
        )
    )

    assert events == [
        "prepare:best",
        "prepare:resume",
        "commit:best.weights",
        "commit:checkpoint-0001.pt",
    ]
    assert result == CheckpointPublication(
        published=(first, second),
        retired=(previous,),
    )
    assert first.read_bytes() == b"best"
    assert second.read_bytes() == b"resume"
    assert not previous.exists()
    assert retained.read_bytes() == b"retained"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_plan_preparation_failure_preserves_all_destinations(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    first = checkpoint_root / "best.weights"
    second = checkpoint_root / "latest.pt"
    retired = checkpoint_root / "previous.pt"
    first.write_bytes(b"old-best")
    second.write_bytes(b"old-latest")
    retired.write_bytes(b"old-resume")

    def write_first(temporary: Path) -> None:
        temporary.write_bytes(b"new-best")

    def fail_second(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("serialization failed")

    with pytest.raises(OSError, match="serialization failed"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root=checkpoint_root,
                artifacts=(
                    CheckpointArtifact(first, write_first),
                    CheckpointArtifact(second, fail_second),
                ),
                retire_after_commit=(retired,),
            )
        )

    assert first.read_bytes() == b"old-best"
    assert second.read_bytes() == b"old-latest"
    assert retired.read_bytes() == b"old-resume"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_plan_noop_serializer_preserves_destinations_and_retirement(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    destination = checkpoint_root / "latest.pt"
    retirement = checkpoint_root / "previous.pt"
    destination.write_bytes(b"old-latest")
    retirement.write_bytes(b"old-previous")

    with pytest.raises(FileNotFoundError):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(destination, lambda _temporary: None),),
                retire_after_commit=(retirement,),
            )
        )

    assert destination.read_bytes() == b"old-latest"
    assert retirement.read_bytes() == b"old-previous"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_plan_commit_failure_cleans_unpublished_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    first = checkpoint_root / "best.weights"
    second = checkpoint_root / "latest.pt"
    retired = checkpoint_root / "previous.pt"
    second.write_bytes(b"old-latest")
    retired.write_bytes(b"old-resume")
    publications = 0
    real_replace = os.replace

    def write(payload: bytes) -> Callable[[Path], None]:
        def write_payload(temporary: Path) -> None:
            temporary.write_bytes(payload)

        return write_payload

    def fail_second_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal publications
        publications += 1
        if publications == 2:
            raise OSError("replace failed")
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="replace failed"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root=checkpoint_root,
                artifacts=(
                    CheckpointArtifact(first, write(b"new-best")),
                    CheckpointArtifact(second, write(b"new-latest")),
                ),
                retire_after_commit=(retired,),
            )
        )

    assert first.read_bytes() == b"new-best"
    assert second.read_bytes() == b"old-latest"
    assert retired.read_bytes() == b"old-resume"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_plan_rejects_parent_replacement_before_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    nested = checkpoint_root / "nested"
    nested.mkdir(parents=True)
    moved_nested = checkpoint_root / "moved-nested"
    outside = tmp_path / "outside"
    outside.mkdir()
    retirement = nested / "old.pt"
    retirement.write_bytes(b"inside")
    outside_retirement = outside / "old.pt"
    outside_retirement.write_bytes(b"outside")
    destination = checkpoint_root / "latest.pt"
    real_publish = publish_prepared_artifact

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    def publish_then_replace_parent(artifact: PreparedArtifact) -> Path:
        published = real_publish(artifact)
        nested.rename(moved_nested)
        nested.symlink_to(outside, target_is_directory=True)
        return published

    monkeypatch.setattr(
        checkpoint_module,
        "publish_prepared_artifact",
        publish_then_replace_parent,
    )

    with pytest.raises(ValueError, match="outside checkpoint_root"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(destination, write),),
                retire_after_commit=(retirement,),
            )
        )

    assert destination.read_bytes() == b"latest"
    assert (moved_nested / "old.pt").read_bytes() == b"inside"
    assert outside_retirement.read_bytes() == b"outside"


def test_checkpoint_plan_rejects_same_parent_moved_outside_root(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    nested = checkpoint_root / "nested"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved_nested = outside / "moved-nested"
    destination = nested / "latest.pt"

    def move_parent_and_write(temporary: Path) -> None:
        nested.rename(moved_nested)
        nested.symlink_to(moved_nested, target_is_directory=True)
        temporary.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside checkpoint_root"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(destination, move_parent_and_write),),
            )
        )

    assert not (moved_nested / "latest.pt").exists()
    assert not list(moved_nested.glob(".*.tmp"))


def test_checkpoint_plan_surfaces_directory_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    destination = checkpoint_root / "latest.pt"
    retirement = checkpoint_root / "previous.pt"
    retirement.write_bytes(b"previous")
    real_fsync = os.fsync

    def fail_directory_sync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError(errno.EIO, "directory sync failed")
        real_fsync(file_descriptor)

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    monkeypatch.setattr(os, "fsync", fail_directory_sync)

    with pytest.raises(OSError, match="directory sync failed"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(destination, write),),
                retire_after_commit=(retirement,),
            )
        )

    assert destination.read_bytes() == b"latest"
    assert retirement.read_bytes() == b"previous"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_plan_syncs_each_new_directory_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    nested = checkpoint_root / "nested"
    destination = nested / "latest.pt"
    synced_directories: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def record_sync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            synced_directories.add((descriptor_stat.st_dev, descriptor_stat.st_ino))
        real_fsync(file_descriptor)

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    monkeypatch.setattr(os, "fsync", record_sync)
    publish_checkpoint_plan(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(destination, write),),
        )
    )

    required_directories = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in (tmp_path, checkpoint_root, nested)
    }
    assert required_directories <= synced_directories
    assert destination.read_bytes() == b"latest"


def test_checkpoint_plan_rejects_unconfined_and_ambiguous_targets(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    target = checkpoint_root / "latest.pt"
    outside = tmp_path / "outside.pt"

    def writer(temporary: Path) -> None:
        temporary.write_bytes(b"checkpoint")

    invalid_plans = (
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(outside, writer),),
        ),
        CheckpointPlan(
            checkpoint_root,
            (
                CheckpointArtifact(target, writer),
                CheckpointArtifact(target, writer),
            ),
        ),
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(target, writer),),
            retire_after_commit=(target,),
        ),
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(target, writer),),
            retire_after_commit=(outside,),
        ),
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(target, writer),),
            retire_after_commit=(checkpoint_root / "old.pt", checkpoint_root / "old.pt"),
        ),
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root, writer),),
        ),
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(target, writer),),
            retire_after_commit=(checkpoint_root,),
        ),
    )
    for plan in invalid_plans:
        with pytest.raises(ValueError):
            publish_checkpoint_plan(plan)


def test_checkpoint_plan_rejects_empty_one_shot_artifact_iterable(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    retirement = checkpoint_root / "previous.pt"
    retirement.write_bytes(b"previous")
    empty_artifacts = cast(tuple[CheckpointArtifact, ...], iter(()))

    with pytest.raises(ValueError, match="at least one artifact"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                empty_artifacts,
                retire_after_commit=(retirement,),
            )
        )

    assert retirement.read_bytes() == b"previous"


def test_checkpoint_plan_rejects_directory_retirement_before_publication(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    destination = checkpoint_root / "latest.pt"
    destination.write_bytes(b"old")
    retirement = checkpoint_root / "old-directory"
    retirement.mkdir()
    writer_called = False

    def write(temporary: Path) -> None:
        nonlocal writer_called
        writer_called = True
        temporary.write_bytes(b"new")

    with pytest.raises(ValueError, match="regular file"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(destination, write),),
                retire_after_commit=(retirement,),
            )
        )

    assert not writer_called
    assert destination.read_bytes() == b"old"
    assert retirement.is_dir()


def test_checkpoint_plan_rejects_missing_descriptor_relative_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"checkpoint")

    monkeypatch.setattr(os, "supports_dir_fd", set())

    with pytest.raises(NotImplementedError, match="POSIX descriptor-relative"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(checkpoint_root / "latest.pt", write),),
            )
        )

    assert not checkpoint_root.exists()


def test_async_checkpoint_plan_submission_applies_bounded_backpressure(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    started = threading.Event()
    release = threading.Event()

    def blocking_writer(temporary: Path) -> None:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"first")

    def write_second(temporary: Path) -> None:
        temporary.write_bytes(b"second")

    first_plan = CheckpointPlan(
        checkpoint_root,
        (CheckpointArtifact(checkpoint_root / "first.pt", blocking_writer),),
    )
    second_plan = CheckpointPlan(
        checkpoint_root,
        (
            CheckpointArtifact(
                checkpoint_root / "second.pt",
                write_second,
            ),
        ),
    )
    errors: list[BaseException] = []
    with AsyncCheckpointPublisher(max_pending=1) as publisher:
        publisher.submit(first_plan)
        assert started.wait(timeout=5)

        def submit_second() -> None:
            try:
                publisher.submit(second_plan)
            except BaseException as error:
                errors.append(error)

        submitter = threading.Thread(target=submit_second)
        submitter.start()
        submitter.join(timeout=0.1)
        assert submitter.is_alive()
        release.set()
        submitter.join(timeout=5)
        assert not submitter.is_alive()
        publisher.flush()

    assert errors == []
    assert (checkpoint_root / "first.pt").read_bytes() == b"first"
    assert (checkpoint_root / "second.pt").read_bytes() == b"second"


def test_concurrent_checkpoint_submissions_preserve_pending_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    worker_started = threading.Event()
    release_worker = threading.Event()
    one_returned = threading.Event()
    all_returned = threading.Event()
    return_lock = threading.Lock()
    returned = 0
    errors: list[BaseException] = []

    def blocking_writer(temporary: Path) -> None:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"checkpoint")

    publisher = AsyncCheckpointPublisher(max_pending=1)
    real_await_submission_slot = publisher._await_submission_slot
    simultaneous_slot_checks = threading.Barrier(2)

    def synchronized_slot_check() -> None:
        real_await_submission_slot()
        with suppress(threading.BrokenBarrierError):
            simultaneous_slot_checks.wait(timeout=0.2)

    monkeypatch.setattr(publisher, "_await_submission_slot", synchronized_slot_check)

    def submit(name: str) -> None:
        nonlocal returned
        try:
            publisher.submit(
                CheckpointPlan(
                    checkpoint_root,
                    (
                        CheckpointArtifact(
                            checkpoint_root / f"{name}.pt",
                            blocking_writer,
                        ),
                    ),
                )
            )
        except BaseException as error:
            errors.append(error)
        finally:
            with return_lock:
                returned += 1
                one_returned.set()
                if returned == 2:
                    all_returned.set()

    submitters = [threading.Thread(target=submit, args=(name,)) for name in ("one", "two")]
    for submitter in submitters:
        submitter.start()

    assert worker_started.wait(timeout=5)
    assert one_returned.wait(timeout=5)
    assert not all_returned.wait(timeout=0.1)
    assert len(publisher._pending) == 1

    release_worker.set()
    for submitter in submitters:
        submitter.join(timeout=5)
        assert not submitter.is_alive()
    publisher.flush()
    publisher.close()

    assert errors == []
    assert all_returned.is_set()
    assert (checkpoint_root / "one.pt").read_bytes() == b"checkpoint"
    assert (checkpoint_root / "two.pt").read_bytes() == b"checkpoint"


def test_async_checkpoint_plan_freezes_relative_paths_before_worker_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_root = Path("checkpoints")
    checkpoint_root.mkdir()
    previous = checkpoint_root / "previous.pt"
    previous.write_bytes(b"previous")
    blocker_root = tmp_path / "blocker"
    blocker_root.mkdir()
    started = threading.Event()
    release = threading.Event()

    def block(temporary: Path) -> None:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"blocker")

    def write_relative(temporary: Path) -> None:
        temporary.write_bytes(b"relative")

    with AsyncCheckpointPublisher(max_pending=2) as publisher:
        publisher.submit(
            CheckpointPlan(
                blocker_root,
                (CheckpointArtifact(blocker_root / "first.pt", block),),
            )
        )
        assert started.wait(timeout=5)
        future = publisher.submit(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(checkpoint_root / "latest.pt", write_relative),),
                retire_after_commit=(previous,),
            )
        )
        other_directory = tmp_path / "other"
        other_directory.mkdir()
        monkeypatch.chdir(other_directory)
        release.set()
        publisher.flush()

    publication = future.result()
    resolved_root = (tmp_path / "checkpoints").resolve()
    assert publication.published == (resolved_root / "latest.pt",)
    assert publication.retired == (resolved_root / "previous.pt",)
    assert (resolved_root / "latest.pt").read_bytes() == b"relative"
    assert not (other_directory / "checkpoints").exists()


def test_async_checkpoint_plan_rejects_root_replacement_while_queued(
    tmp_path: Path,
) -> None:
    blocker_root = tmp_path / "blocker"
    blocker_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    previous = checkpoint_root / "previous.pt"
    previous.write_bytes(b"inside")
    moved_root = tmp_path / "moved-checkpoints"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_previous = outside / "previous.pt"
    outside_previous.write_bytes(b"outside")
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block(temporary: Path) -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"blocker")

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    publisher.submit(
        CheckpointPlan(
            blocker_root,
            (CheckpointArtifact(blocker_root / "blocker.pt", block),),
        )
    )
    assert blocker_started.wait(timeout=5)
    redirected = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "latest.pt", write),),
            retire_after_commit=(previous,),
        )
    )
    checkpoint_root.rename(moved_root)
    checkpoint_root.symlink_to(outside, target_is_directory=True)
    release_blocker.set()

    with pytest.raises((OSError, ValueError)):
        redirected.result()
    with pytest.raises((OSError, ValueError)):
        publisher.flush()
    publisher.close()

    assert (moved_root / "previous.pt").read_bytes() == b"inside"
    assert not (moved_root / "latest.pt").exists()
    assert outside_previous.read_bytes() == b"outside"
    assert not (outside / "latest.pt").exists()


def test_async_checkpoint_plan_rejects_destination_symlink_while_queued(
    tmp_path: Path,
) -> None:
    blocker_root = tmp_path / "blocker"
    blocker_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    destination = checkpoint_root / "latest.pt"
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block(temporary: Path) -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"blocker")

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    publisher.submit(
        CheckpointPlan(
            blocker_root,
            (CheckpointArtifact(blocker_root / "blocker.pt", block),),
        )
    )
    assert blocker_started.wait(timeout=5)
    redirected = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(destination, write),),
        )
    )
    destination.symlink_to(outside)
    release_blocker.set()

    with pytest.raises(ValueError, match="regular file"):
        redirected.result()
    with pytest.raises(ValueError, match="regular file"):
        publisher.flush()
    publisher.close()

    assert destination.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_async_checkpoint_plan_rejects_ordinary_root_replacement_while_queued(
    tmp_path: Path,
) -> None:
    blocker_root = tmp_path / "blocker"
    blocker_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    previous = checkpoint_root / "previous.pt"
    previous.write_bytes(b"inside")
    moved_root = tmp_path / "moved-checkpoints"
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block(temporary: Path) -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"blocker")

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    publisher.submit(
        CheckpointPlan(
            blocker_root,
            (CheckpointArtifact(blocker_root / "blocker.pt", block),),
        )
    )
    assert blocker_started.wait(timeout=5)
    redirected = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "latest.pt", write),),
            retire_after_commit=(previous,),
        )
    )
    checkpoint_root.rename(moved_root)
    checkpoint_root.mkdir()
    replacement_previous = checkpoint_root / "previous.pt"
    replacement_previous.write_bytes(b"replacement")
    release_blocker.set()

    with pytest.raises(RuntimeError, match="checkpoint root changed"):
        redirected.result()
    with pytest.raises(RuntimeError, match="checkpoint root changed"):
        publisher.flush()
    publisher.close()

    assert (moved_root / "previous.pt").read_bytes() == b"inside"
    assert not (moved_root / "latest.pt").exists()
    assert replacement_previous.read_bytes() == b"replacement"
    assert not (checkpoint_root / "latest.pt").exists()


def test_async_checkpoint_plan_anchors_root_before_bounded_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocker_root = tmp_path / "blocker"
    blocker_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    previous = checkpoint_root / "old.pt"
    previous.write_bytes(b"inside")
    moved_root = tmp_path / "moved-checkpoints"
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    waiting_for_slot = threading.Event()

    def block(temporary: Path) -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=5):
            raise TimeoutError("test did not release checkpoint writer")
        temporary.write_bytes(b"blocker")

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"latest")

    publisher = AsyncCheckpointPublisher(max_pending=1)
    publisher.submit(
        CheckpointPlan(
            blocker_root,
            (CheckpointArtifact(blocker_root / "blocker.pt", block),),
        )
    )
    assert blocker_started.wait(timeout=5)
    real_await_submission_slot = publisher._await_submission_slot

    def observe_bounded_wait() -> None:
        waiting_for_slot.set()
        real_await_submission_slot()

    monkeypatch.setattr(publisher, "_await_submission_slot", observe_bounded_wait)
    submitted: list[Future[CheckpointPublication]] = []
    errors: list[BaseException] = []

    def submit_redirected_plan() -> None:
        try:
            submitted.append(
                publisher.submit(
                    CheckpointPlan(
                        checkpoint_root,
                        (CheckpointArtifact(checkpoint_root / "latest.pt", write),),
                        retire_after_commit=(previous,),
                    )
                )
            )
        except BaseException as error:
            errors.append(error)

    submitter = threading.Thread(target=submit_redirected_plan)
    submitter.start()
    assert waiting_for_slot.wait(timeout=5)
    checkpoint_root.rename(moved_root)
    checkpoint_root.mkdir()
    replacement_previous = checkpoint_root / "old.pt"
    replacement_previous.write_bytes(b"replacement")
    release_blocker.set()
    submitter.join(timeout=5)
    assert not submitter.is_alive()
    assert errors == []
    assert len(submitted) == 1

    with pytest.raises(RuntimeError, match="checkpoint root changed"):
        submitted[0].result()
    with pytest.raises(RuntimeError, match="checkpoint root changed"):
        publisher.flush()
    publisher.close()

    assert (moved_root / "old.pt").read_bytes() == b"inside"
    assert not (moved_root / "latest.pt").exists()
    assert replacement_previous.read_bytes() == b"replacement"
    assert not (checkpoint_root / "latest.pt").exists()


def test_async_checkpoint_plan_failure_surfaces_at_next_bounded_submission(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    def fail(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("plan failed")

    def succeed(temporary: Path) -> None:
        temporary.write_bytes(b"complete")

    publisher = AsyncCheckpointPublisher(max_pending=1)
    failed = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "failed.pt", fail),),
        )
    )
    with pytest.raises(OSError, match="plan failed"):
        failed.result()
    with pytest.raises(OSError, match="plan failed"):
        publisher.submit(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(checkpoint_root / "blocked.pt", succeed),),
            )
        )
    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "complete.pt", succeed),),
        )
    )
    publisher.flush()
    publisher.close()
    publisher.close()

    assert (checkpoint_root / "complete.pt").read_bytes() == b"complete"
    assert not list(checkpoint_root.glob(".*.tmp"))


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_async_checkpoint_serializer_interrupt_is_dequeued_after_propagation(
    tmp_path: Path,
    interrupt_type: type[BaseException],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    def interrupt(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise interrupt_type("worker interrupted")

    def succeed(temporary: Path) -> None:
        temporary.write_bytes(b"complete")

    publisher = AsyncCheckpointPublisher()
    interrupted = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "interrupted.pt", interrupt),),
        )
    )
    with pytest.raises(interrupt_type, match="worker interrupted"):
        interrupted.result()
    with pytest.raises(interrupt_type, match="worker interrupted"):
        publisher.submit(
            CheckpointPlan(
                checkpoint_root,
                (CheckpointArtifact(checkpoint_root / "not-submitted.pt", succeed),),
            )
        )

    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "complete.pt", succeed),),
        )
    )
    publisher.flush()
    publisher.close()

    assert not (checkpoint_root / "interrupted.pt").exists()
    assert not (checkpoint_root / "not-submitted.pt").exists()
    assert (checkpoint_root / "complete.pt").read_bytes() == b"complete"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_async_checkpoint_plan_failure_surfaces_from_flush_close_and_context(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    def fail(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("plan failed")

    plan = CheckpointPlan(
        checkpoint_root,
        (CheckpointArtifact(checkpoint_root / "failed.pt", fail),),
    )

    flushed = AsyncCheckpointPublisher()
    flushed.submit(plan)
    with pytest.raises(OSError, match="plan failed"):
        flushed.flush()
    flushed.close()

    closed = AsyncCheckpointPublisher()
    closed.submit(plan)
    with pytest.raises(OSError, match="plan failed"):
        closed.close()
    closed.close()

    with pytest.raises(OSError, match="plan failed"), AsyncCheckpointPublisher() as contextual:
        contextual.submit(plan)

    assert not list(checkpoint_root.glob(".*.tmp"))


def test_async_checkpoint_flush_waits_for_later_work_before_raising(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    second_started = threading.Event()
    release_second = threading.Event()

    def fail_first(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("first plan failed")

    def block_second(temporary: Path) -> None:
        second_started.set()
        if not release_second.wait(timeout=5):
            raise TimeoutError("test did not release second checkpoint writer")
        temporary.write_bytes(b"second")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "first.pt", fail_first),),
        )
    )
    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "second.pt", block_second),),
        )
    )
    errors: list[BaseException] = []

    def flush() -> None:
        try:
            publisher.flush()
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=flush)
    waiter.start()
    assert second_started.wait(timeout=5)
    waiter.join(timeout=0.1)
    assert waiter.is_alive()
    release_second.set()
    waiter.join(timeout=5)
    publisher.close()

    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)
    assert str(errors[0]) == "first plan failed"
    assert (checkpoint_root / "second.pt").read_bytes() == b"second"


def test_precision_and_ddp_configuration_guards(tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="fp16"):
        Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_step=regression_step,
            config=TrainerConfig(epochs=1, device="cpu", precision="fp16"),
        )
    with pytest.raises(RuntimeError, match="initialized"):
        Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_step=regression_step,
            config=TrainerConfig(epochs=1, device="cpu", strategy="ddp"),
        )


def test_bf16_precision_scheduler_and_batch_mover_are_generic() -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()
    moved_batches = 0

    def mover(batch: Any, device: torch.device) -> Any:
        nonlocal moved_batches
        moved_batches += 1
        return move_batch_to_device(batch, device)

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        scheduler=scheduler,
        batch_mover=mover,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            precision="bf16",
            scheduler_interval="optimizer",
            checkpoint_every_epochs=None,
        ),
    ) as trainer:
        result = trainer.fit()

    assert moved_batches == len(loader)
    assert scheduler.steps == result.state.optimizer_step == len(loader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fp16_precision_uses_cuda_scaling() -> None:
    features = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=classification_step,
        config=TrainerConfig(
            epochs=1,
            device="cuda:0",
            precision="fp16",
            checkpoint_every_epochs=None,
        ),
    ) as trainer:
        result = trainer.fit()

    assert result.state.optimizer_step == len(loader)


def test_world_size_one_cpu_ddp_uses_same_project_step_contract(tmp_path: Path) -> None:
    rendezvous = tmp_path / "rendezvous"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        loader = DataLoader(MappingDataset(), batch_size=2)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_step=regression_step,
            config=TrainerConfig(
                epochs=1,
                device="cpu",
                strategy="ddp",
                checkpoint_every_epochs=None,
            ),
        ) as trainer:
            result = trainer.fit()
        assert result.state.global_step == len(loader)
    finally:
        torch.distributed.destroy_process_group()


def test_single_runtime_establishes_execution_and_supplies_trainer_observer(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "single-run"
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with initialize_torch_runtime(TorchRuntimeConfig(device="cpu")) as runtime:
        bundle = runtime.start_execution(
            TorchExecutionRequest(
                run_dir=run_dir,
                run_name="single-run",
                invocation_kind="test",
                intended_phases=("train",),
                command=("python", "train.py"),
                execution_id="single-attempt",
                runtime={"credentials": {"api_token": "secret"}},
            )
        )
        bundle.observer.emit("process_started", phase="train")
        with Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_step=regression_step,
            config=TrainerConfig(epochs=1, device="cpu"),
            checkpoint_dir=run_dir / "checkpoints",
            runtime=runtime,
        ) as trainer:
            result = trainer.fit()
        bundle.observer.emit("process_completed", phase="train", exit_code=0)

        assert result.state.global_step == len(loader)
        assert runtime.execution_context is not None
        assert runtime.execution_context.metadata.runtime == {
            "backend": None,
            "credentials": "<redacted>",
            "device_type": "cpu",
            "framework": "pytorch",
            "framework_version": str(torch.__version__),
            "strategy": "single",
        }

    events = read_execution_events(
        run_dir / "logs" / "executions" / "single-attempt" / "rank-0.jsonl"
    )
    assert {event.event for event in events} >= {
        "process_started",
        "phase_started",
        "progress",
        "phase_completed",
        "process_completed",
    }
    assert (run_dir / "checkpoints" / "checkpoint-0000.pt").is_file()


def test_single_runtime_joins_runner_execution_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "joined-run"
    create_execution_context(
        run_dir,
        run_name="joined-run",
        invocation_kind="workflow",
        intended_phases=("train", "validate"),
        world_size=1,
        execution_mode="single",
        command=("mammoth", "workflow", "run"),
        execution_id="runner-attempt",
    )
    monkeypatch.setenv("MAMMOTH_EXECUTION_ID", "runner-attempt")

    with initialize_torch_runtime(TorchRuntimeConfig(device="cpu")) as runtime:
        bundle = runtime.start_execution(
            TorchExecutionRequest(
                run_dir=run_dir,
                run_name="joined-run",
                invocation_kind="train",
                intended_phases=("train",),
                command=("python", "train.py"),
            )
        )
        bundle.observer.emit("process_completed", phase="train", exit_code=0)
        assert runtime.execution_context is not None
        assert runtime.execution_context.metadata.execution_id == "runner-attempt"


def test_execution_establishment_can_be_used_without_mammoth_logging(tmp_path: Path) -> None:
    run_dir = tmp_path / "adapter-run"
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    context = runtime.establish_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="adapter-run",
            invocation_kind="test",
            intended_phases=("custom",),
            command=("python", "custom.py"),
            execution_id="adapter-attempt",
        )
    )

    assert context.metadata.execution_id == "adapter-attempt"
    with pytest.raises(RuntimeError, match="already active"):
        claim_logical_run_lease(run_dir)
    runtime.close()
    with claim_logical_run_lease(run_dir):
        pass


def test_two_process_runtime_owns_ddp_execution_collectives_and_rank_streams(
    tmp_path: Path,
) -> None:
    results = run_two_process_runtime(tmp_path)

    assert [result[0] for result in results] == [0, 1]
    assert {result[1] for result in results} == {"ddp-attempt"}
    assert {result[2] for result in results} == {"ready"}
    assert {result[3] for result in results} == {(0, 1)}
    assert {result[4] for result in results} == {3}
    assert {result[5] for result in results} == {2}
    assert {result[6] for result in results} == {None}
    execution_dir = tmp_path / "ddp-run" / "logs" / "executions" / "ddp-attempt"
    for rank in range(2):
        assert (execution_dir / f"rank-{rank}.log").is_file()
        events = read_execution_events(execution_dir / f"rank-{rank}.jsonl")
        assert any(event.event == "progress" for event in events)
    assert len(list((tmp_path / "ddp-run" / "checkpoints").glob("*.pt"))) == 1


def test_two_process_runtime_propagates_rank_logging_startup_failure(
    tmp_path: Path,
) -> None:
    results = run_two_process_runtime(tmp_path, fail_logging_rank=1)

    errors = {result[6] for result in results}
    assert len(errors) == 1
    error = errors.pop()
    assert error is not None
    assert "execution logging" in error
    assert "rank 1" in error
    assert "rank log unavailable" in error
