from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset, TensorDataset

from mammoth.logging import Observation, RunObserver
from mammoth.torch import (
    AsyncCheckpointPublisher,
    EarlyStopping,
    MetricAccumulator,
    MetricSpec,
    StateRegistry,
    StepContext,
    StepOutput,
    Trainer,
    TrainerConfig,
    move_batch_to_device,
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

    def train_step(
        module: torch.nn.Module, batch: Any, context: StepContext
    ) -> StepOutput:
        prediction = module(batch[0])
        return StepOutput(loss=prediction.sum() * 0)

    def validation_step(
        module: torch.nn.Module, batch: Any, context: StepContext
    ) -> StepOutput:
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
