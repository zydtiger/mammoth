from __future__ import annotations

import errno
import hashlib
import multiprocessing
import os
import stat
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import nullcontext, suppress
from pathlib import Path
from queue import Empty
from typing import Any, Literal, cast

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

import mammoth.torch.checkpoint as checkpoint_module
import mammoth.torch.runtime as torch_runtime_module
import mammoth.torch.trainer as trainer_module
from mammoth.core import (
    BoundedBackgroundPipeline,
    PreparedArtifact,
    claim_logical_run_lease,
    create_execution_context,
    publish_prepared_artifact,
    read_execution_events,
)
from mammoth.core.events import ExecutionEventWriter
from mammoth.logging import JsonlEventSink, Observation, RunObserver
from mammoth.torch import (
    AccumulationPlan,
    AsyncCheckpointPublisher,
    Callback,
    CheckpointArtifact,
    CheckpointInspection,
    CheckpointPlan,
    CheckpointPublication,
    CheckpointSavePolicy,
    EarlyStopping,
    MetricAccumulator,
    MetricRoute,
    MetricSpec,
    PublishedCheckpoint,
    RestoreOptions,
    StateRegistry,
    StepContext,
    StepOutput,
    TorchCompileConfig,
    TorchExecutionRequest,
    TorchRuntimeConfig,
    Trainer,
    TrainerCheckpointContext,
    TrainerCheckpointRestore,
    TrainerCheckpointWriters,
    TrainerConfig,
    TrainerState,
    UniformAccumulationPolicy,
    WarmupLinearLR,
    WeightedAccumulationPolicy,
    WeightedDistributedBatchSampler,
    allocate_weighted_tasks,
    checkpoint_payload,
    initialize_torch_runtime,
    move_batch_to_device,
    publish_checkpoint_plan,
    restore_checkpoint,
    weighted_partition_counts,
    weighted_partition_indices,
)
from mammoth.torch.metrics import compute_stateful_metrics


def build_warmup_linear_stack(
    *,
    total_steps: int,
    warmup_ratio: float = 0.0,
) -> tuple[torch.optim.Optimizer, WarmupLinearLR]:
    """Build a two-group optimizer and reusable warmup-linear scheduler."""
    parameters = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [
            {"params": [parameters[0]], "lr": 1.0},
            {"params": [parameters[1]], "lr": 0.1},
        ]
    )
    return optimizer, WarmupLinearLR(
        optimizer,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
    )


def advance_optimizer_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupLinearLR,
    count: int,
) -> None:
    """Advance a scheduler using PyTorch's optimizer-before-scheduler order."""
    for _ in range(count):
        optimizer.step()
        scheduler.step()


def test_warmup_linear_lr_validates_configuration_and_boundaries() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    for total_steps in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="total_steps must be a positive integer"):
            WarmupLinearLR(optimizer, warmup_ratio=0.5, total_steps=cast(Any, total_steps))
    for warmup_ratio in (-0.1, 1.0, float("inf"), True):
        with pytest.raises(ValueError, match="warmup_ratio must be finite"):
            WarmupLinearLR(
                optimizer,
                warmup_ratio=cast(Any, warmup_ratio),
                total_steps=4,
            )

    scheduler = WarmupLinearLR(optimizer, warmup_ratio=0.5, total_steps=4)
    observed = [scheduler.get_last_lr()[0]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])
    assert observed == pytest.approx([0.0, 0.05, 0.1, 0.05, 0.0])


def test_warmup_linear_lr_preserves_multiple_parameter_group_ratios() -> None:
    optimizer, scheduler = build_warmup_linear_stack(total_steps=4, warmup_ratio=0.5)
    observed = [scheduler.get_last_lr()]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr())
    torch.testing.assert_close(
        torch.tensor(observed),
        torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.05],
                [1.0, 0.1],
                [0.5, 0.05],
                [0.0, 0.0],
            ]
        ),
    )


def test_warmup_linear_lr_checkpoint_round_trip_preserves_same_horizon(
    tmp_path: Path,
) -> None:
    optimizer, scheduler = build_warmup_linear_stack(total_steps=8)
    advance_optimizer_scheduler(optimizer, scheduler, 2)
    registry = StateRegistry()
    registry.register("optimizer", optimizer)
    registry.register("scheduler", scheduler)
    checkpoint = tmp_path / "warmup-linear.pt"
    torch.save(checkpoint_payload(registry), checkpoint)
    scheduler_state = scheduler.state_dict()

    restored_optimizer, restored_scheduler = build_warmup_linear_stack(total_steps=8)
    restored_registry = StateRegistry()
    restored_registry.register("optimizer", restored_optimizer)
    restored_registry.register("scheduler", restored_scheduler)
    restore_checkpoint(checkpoint, restored_registry)

    assert restored_scheduler.state_dict() == scheduler_state
    assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx(
        [0.75, 0.075]
    )


def test_warmup_linear_lr_rebases_extended_horizon() -> None:
    optimizer, scheduler = build_warmup_linear_stack(total_steps=4)
    advance_optimizer_scheduler(optimizer, scheduler, 4)
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    restored_optimizer, restored_scheduler = build_warmup_linear_stack(total_steps=8)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)

    assert restored_scheduler.last_epoch == 4
    assert restored_scheduler.total_steps == 8
    assert restored_scheduler.get_last_lr() == pytest.approx([0.5, 0.05])
    assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx(
        [0.5, 0.05]
    )


def test_warmup_linear_lr_rejects_shorter_resume_horizon() -> None:
    optimizer, scheduler = build_warmup_linear_stack(total_steps=8)
    advance_optimizer_scheduler(optimizer, scheduler, 2)
    scheduler_state = scheduler.state_dict()
    _, shorter_scheduler = build_warmup_linear_stack(total_steps=4)

    with pytest.raises(
        ValueError,
        match=r"checkpoint total_steps=8, configured total_steps=4",
    ):
        shorter_scheduler.load_state_dict(scheduler_state)


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


class UnevenAccumulationPolicy:
    """Give two test ranks a 3:1 local microbatch split."""

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        assert world_size == 2
        assert local_batch_count in {2, 6}
        return AccumulationPlan(
            local_microbatches_per_step=3 if rank == 0 else 1,
            loss_scale=world_size / 4,
            incomplete_window="error",
        )


class UnevenPartialAccumulationPolicy:
    """Give two ranks correct scales for a partial two-microbatch window."""

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        assert world_size == 2
        assert local_batch_count in {2, 4}
        return AccumulationPlan(
            local_microbatches_per_step=3 if rank == 0 else 1,
            loss_scale=world_size / 4,
            incomplete_window="step",
            window_loss_scales=(world_size / 4, world_size / 2),
        )


class RankConditionalInvalidPolicy:
    """Reject only rank one's local final window for consensus coverage."""

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        del world_size, local_batch_count
        return AccumulationPlan(
            local_microbatches_per_step=1 if rank == 0 else 2,
            loss_scale=1.0,
            incomplete_window="error",
        )


class AdditiveCountMetric:
    """Count opaque update values through additive tensor state."""

    def __init__(self) -> None:
        self.count = torch.tensor(0.0)

    def reset(self) -> None:
        self.count.zero_()

    def update(self, value: Any) -> None:
        self.count += float(value)

    def state_tensors(self) -> Mapping[str, torch.Tensor]:
        return {"count": self.count}

    def compute(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, float | torch.Tensor]:
        return {"count": state["count"]}


class RaisingResetMetric(AdditiveCountMetric):
    """Fail during metric lifecycle setup for terminal-event coverage."""

    def reset(self) -> None:
        raise RuntimeError("reset failed")


class RankShapedMetric(AdditiveCountMetric):
    """Return rank-dependent state shapes for pre-collective validation."""

    def __init__(self, rank: int) -> None:
        self.count = torch.zeros(rank + 1)


class NamedCountMetric(AdditiveCountMetric):
    """Return additive count state under one configured scalar name."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def compute(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, float | torch.Tensor]:
        return {self.name: state["count"]}


class RankFailingComputeMetric(NamedCountMetric):
    """Fail project metric computation only on one rank."""

    def __init__(self, rank: int) -> None:
        super().__init__("first")
        self.rank = rank

    def compute(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, float | torch.Tensor]:
        if self.rank == 1:
            raise ValueError("rank-one metric compute failed")
        return super().compute(state)


class RankFailingSampler(Sampler[int]):
    """Fail epoch setup on one selected rank for DDP consensus coverage."""

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __iter__(self) -> Any:
        return iter((0,))

    def __len__(self) -> int:
        return 1

    def set_epoch(self, epoch: int) -> None:
        del epoch
        if self.rank == 1:
            raise ValueError("rank-one sampler failed")


class EpochRecordingBatchSampler(Sampler[list[int]]):
    """Record epoch advancement for a project-owned batch sampler."""

    def __init__(self) -> None:
        self.epochs: list[int] = []

    def __iter__(self) -> Any:
        return iter(([0],))

    def __len__(self) -> int:
        return 1

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class RaisingCallback(Callback):
    """Fail at one selected outer trainer lifecycle hook."""

    def __init__(self, hook: str) -> None:
        self.hook = hook

    def on_train_start(self, state: TrainerState) -> None:
        del state
        if self.hook == "start":
            raise RuntimeError("start callback failed")

    def on_train_end(self, state: TrainerState) -> None:
        del state
        if self.hook == "end":
            raise RuntimeError("end callback failed")


class RecordingCheckpointPolicy:
    """Record trainer checkpoint contexts in one project-owned text artifact."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.contexts: list[TrainerCheckpointContext] = []

    def inspect(
        self,
        path: Path,
    ) -> CheckpointInspection:
        del path
        return CheckpointInspection(
            available_components=frozenset({"model", "trainer", "stopped_early"}),
        )

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(epoch=2, global_step=7, optimizer_step=3)

    def capture(self, context: TrainerCheckpointContext) -> TrainerCheckpointWriters:
        self.contexts.append(context)
        return TrainerCheckpointWriters(
            resumable=lambda path: path.write_text(
                f"epoch={context.epoch}\n",
                encoding="utf-8",
            ),
            best=lambda path: path.write_text(
                f"best_epoch={context.epoch}\n",
                encoding="utf-8",
            ),
        )


class CursorlessCheckpointPolicy(RecordingCheckpointPolicy):
    """Restore only an epoch so Mammoth must infer ordinary loop cursors."""

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(epoch=1)


class InitialCheckpointPolicy(RecordingCheckpointPolicy):
    """Restore Mammoth's valid pre-training coordinate."""

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(epoch=-1)


class SlowRecordingCheckpointPolicy(RecordingCheckpointPolicy):
    """Delay checkpoint planning long enough to exercise periodic heartbeats."""

    def capture(self, context: TrainerCheckpointContext) -> TrainerCheckpointWriters:
        time.sleep(0.05)
        return super().capture(context)


class RankFailingRestorePolicy(RecordingCheckpointPolicy):
    """Fail project checkpoint restore only on one DDP rank."""

    def __init__(self, checkpoint_dir: Path, rank: int) -> None:
        super().__init__(checkpoint_dir)
        self.rank = rank

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        if self.rank == 1:
            raise ValueError("rank-one restore failed")
        return super().restore(path, device=device, options=options)


class RankDivergentRestorePolicy(RecordingCheckpointPolicy):
    """Return rank-local generic state that Mammoth must synchronize from rank zero."""

    def __init__(self, checkpoint_dir: Path, rank: int) -> None:
        super().__init__(checkpoint_dir)
        self.rank = rank

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD((parameter,), lr=0.1 + self.rank)
        return TrainerCheckpointRestore(
            epoch=0,
            global_step=1,
            optimizer_step=1,
            optimizer_state_dict=optimizer.state_dict(),
        )


class UnpickleableInspectionPolicy(RecordingCheckpointPolicy):
    """Return metadata that rank zero cannot serialize for DDP inspection."""

    def inspect(self, path: Path) -> CheckpointInspection:
        del path
        return CheckpointInspection(
            available_components=frozenset({"model"}),
            metadata={"unpickleable": lambda: None},
        )


class FalseResetReportingPolicy(RecordingCheckpointPolicy):
    """Falsely claim a Mammoth-managed reset for contract validation."""

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(
            epoch=0,
            reset_components=frozenset({"optimizer"}),
        )


class CallbackRestorePolicy(RecordingCheckpointPolicy):
    """Return callback state to exercise rank-local application failures."""

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(
            epoch=0,
            callback_state_dicts={0: {"best": 0.5, "bad_checks": 1}},
        )


class TensorMetadataRestorePolicy(RecordingCheckpointPolicy):
    """Return identical tensor-valued opaque metadata on every rank."""

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(
            epoch=0,
            metadata={
                "tensor": torch.tensor([1, 2]),
                "components": frozenset({"model", "scaler", "project"}),
            },
        )


class RankInvalidRestorePolicy(RecordingCheckpointPolicy):
    """Return an invalid checkpoint result on only one rank."""

    def __init__(self, checkpoint_dir: Path, rank: int) -> None:
        super().__init__(checkpoint_dir)
        self.rank = rank

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> Any:
        if self.rank == 1:
            return {"epoch": 0}
        return super().restore(path, device=device, options=options)


class TypedStateCheckpointPolicy(RecordingCheckpointPolicy):
    """Return normalized generic states with an inspection-selected callback reset."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any],
        callback_state: Mapping[str, Any],
    ) -> None:
        super().__init__(checkpoint_dir)
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state
        self.callback_state = callback_state

    def inspect(
        self,
        path: Path,
    ) -> CheckpointInspection:
        del path
        return CheckpointInspection(
            available_components=frozenset(
                {"model", "optimizer", "scheduler", "callbacks", "trainer", "stopped_early"}
            ),
            restore_options=RestoreOptions(
                callbacks="reset",
                stopped_early="reset",
            ),
            metadata={"objective_changed": True},
        )

    def restore(
        self,
        path: Path,
        *,
        device: torch.device,
        options: RestoreOptions,
    ) -> TrainerCheckpointRestore:
        del path, device, options
        return TrainerCheckpointRestore(
            epoch=1,
            optimizer_step=4,
            global_step=8,
            stopped_early=True,
            optimizer_state_dict=self.optimizer_state,
            scheduler_state_dict=self.scheduler_state,
            callback_state_dicts={0: self.callback_state},
            metadata={"objective_changed": True},
            restored_components=frozenset({"model"}),
        )


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


def _distributed_interrupt_worker(
    rank: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_queue: Any,
) -> None:
    """Report whether rank-wide interrupt consensus reaches checkpoint policy."""
    try:
        with initialize_torch_runtime(
            TorchRuntimeConfig(
                strategy="ddp",
                device="cpu",
                backend="gloo",
                init_method=rendezvous,
                timeout_seconds=30,
                rank=rank,
                local_rank=rank,
                world_size=2,
            )
        ) as runtime:
            loader = DataLoader(MappingDataset(), batch_size=2)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            policy = RecordingCheckpointPolicy(Path(checkpoint_dir))

            def interrupt_on_peer(
                module: torch.nn.Module,
                batch: Any,
                context: StepContext,
            ) -> StepOutput:
                if rank == 1:
                    raise KeyboardInterrupt("stop distributed training")
                return regression_step(module, batch, context)

            error_type = ""
            error_message = ""
            with Trainer(
                model=model,
                optimizer=optimizer,
                train_loader=loader,
                train_step=interrupt_on_peer,
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    strategy="ddp",
                    checkpoint_every_epochs=None,
                ),
                checkpoint_dir=Path(checkpoint_dir),
                checkpoint_policy=policy,
                checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
                runtime=runtime,
            ) as trainer:
                try:
                    trainer.fit()
                except BaseException as error:
                    error_type = type(error).__name__
                    error_message = str(error)
            result_queue.put(
                (
                    rank,
                    error_type,
                    error_message,
                    [context.reason for context in policy.contexts],
                )
            )
    except BaseException as error:
        result_queue.put((rank, type(error).__name__, str(error), []))


def run_distributed_interrupt(tmp_path: Path) -> list[tuple[Any, ...]]:
    """Launch the two-rank interrupted-checkpoint regression fixture."""
    process_context = multiprocessing.get_context("spawn")
    result_queue = process_context.Queue()
    rendezvous = f"file://{tmp_path / 'interrupt-rendezvous'}"
    checkpoint_dir = str(tmp_path / "interrupted-checkpoint")
    processes = [
        process_context.Process(
            target=_distributed_interrupt_worker,
            args=(rank, rendezvous, checkpoint_dir, result_queue),
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
            pytest.fail("Distributed interrupt worker did not shut down coherently")
        assert process.exitcode == 0
    return sorted(result_queue.get(timeout=5) for _ in processes)


def _uneven_accumulation_worker(
    rank: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_queue: Any,
) -> None:
    """Exercise unequal rank-local windows with global logical-batch metrics."""
    try:
        with initialize_torch_runtime(
            TorchRuntimeConfig(
                strategy="ddp",
                device="cpu",
                backend="gloo",
                init_method=rendezvous,
                timeout_seconds=30,
                rank=rank,
                local_rank=rank,
                world_size=2,
            )
        ) as runtime:
            local_count = 6 if rank == 0 else 2
            local_value = 1.0 if rank == 0 else 3.0
            features = torch.full((local_count, 1), local_value)
            loader = DataLoader(TensorDataset(features), batch_size=1)
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            sink = RecordingSink()

            def step(
                module: torch.nn.Module,
                batch: Any,
                context: StepContext,
            ) -> StepOutput:
                del context
                values = batch[0]
                loss = module(values).sum() * 0 + values.mean()
                return StepOutput(
                    loss=loss,
                    metrics={"sample": values.mean()},
                    metric_updates={"count": len(values)},
                )

            with Trainer(
                model=model,
                optimizer=optimizer,
                train_loader=loader,
                train_step=step,
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    strategy="ddp",
                ),
                accumulation_policy=UnevenAccumulationPolicy(),
                train_stateful_metrics={"count": AdditiveCountMetric()},
                train_metric_routes={
                    "count": MetricRoute(
                        batch_name="batch/count",
                        epoch_name="epoch/count",
                    )
                },
                observer=RunObserver((sink,)),
                checkpoint_dir=Path(checkpoint_dir) if rank == 0 else None,
                runtime=runtime,
            ) as trainer:
                result = trainer.fit()
            progress = [
                observation
                for observation in sink.observations
                if observation.event == "progress"
            ]
            partial_count = 4 if rank == 0 else 2
            partial_value = 1.0 if rank == 0 else 3.0
            partial_loader = DataLoader(
                TensorDataset(torch.full((partial_count, 1), partial_value)),
                batch_size=1,
            )
            partial_model = torch.nn.Linear(1, 1, bias=False)
            partial_optimizer = torch.optim.SGD(partial_model.parameters(), lr=1.0)

            def partial_step(
                module: torch.nn.Module,
                batch: Any,
                context: StepContext,
            ) -> StepOutput:
                del context
                return StepOutput(loss=module(batch[0]).mean())

            with Trainer(
                model=partial_model,
                optimizer=partial_optimizer,
                train_loader=partial_loader,
                train_step=partial_step,
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    strategy="ddp",
                    checkpoint_every_epochs=None,
                ),
                accumulation_policy=UnevenPartialAccumulationPolicy(),
                runtime=runtime,
            ) as partial_trainer:
                partial_initial_weight = partial_model.weight.detach().item()
                partial_trainer.fit()
            partial_weight_delta = (
                partial_model.weight.detach().item() - partial_initial_weight
            )
            last_accumulator = MetricAccumulator({"rank1": MetricSpec("last")})
            if rank == 1:
                last_accumulator.update({"rank1": 7.0})
            try:
                last_accumulator.compute(device=runtime.device, distributed=True)
            except ValueError as error:
                last_error = str(error)
            else:
                last_error = None
            primary_last_accumulator = MetricAccumulator(
                {"primary": MetricSpec("last")}
            )
            if rank == 0:
                primary_last_accumulator.update({"primary": 9.0})
            primary_last_metrics = primary_last_accumulator.compute(
                device=runtime.device,
                distributed=True,
            )
            local_accumulator = MetricAccumulator(
                {"local": MetricSpec("mean", distributed=False)}
            )
            if rank == 1:
                local_accumulator.update({"local": 7.0})
            local_metrics = local_accumulator.compute(
                device=runtime.device,
                distributed=True,
            )
            try:
                compute_stateful_metrics(
                    {"shape": RankShapedMetric(rank)},
                    device=runtime.device,
                    distributed=True,
                )
            except ValueError as error:
                shape_error = str(error)
            else:
                shape_error = None
            try:
                compute_stateful_metrics(
                    {
                        "first": RankFailingComputeMetric(rank),
                        "second": NamedCountMetric("second"),
                    },
                    device=runtime.device,
                    distributed=True,
                )
            except RuntimeError as error:
                metric_compute_error = str(error)
            else:
                metric_compute_error = None
            failing_model = torch.nn.Linear(1, 1, bias=False)
            failing_optimizer = torch.optim.SGD(failing_model.parameters(), lr=0.0)
            failing_loader = DataLoader(
                TensorDataset(torch.ones(1, 1)),
                batch_size=1,
            )
            try:
                with Trainer(
                    model=failing_model,
                    optimizer=failing_optimizer,
                    train_loader=failing_loader,
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    accumulation_policy=RankConditionalInvalidPolicy(),
                    runtime=runtime,
                ) as failing_trainer:
                    failing_trainer.fit()
            except RuntimeError as error:
                planning_error = str(error)
            else:
                planning_error = None
            step_failure_model = torch.nn.Linear(1, 1, bias=False)
            step_failure_optimizer = torch.optim.SGD(
                step_failure_model.parameters(),
                lr=0.0,
            )
            step_failure_count = 6 if rank == 0 else 2
            step_failure_loader = DataLoader(
                TensorDataset(torch.ones(step_failure_count, 1)),
                batch_size=1,
            )

            def rank_failing_step(
                module: torch.nn.Module,
                batch: Any,
                context: StepContext,
            ) -> StepOutput:
                del context
                if rank == 1:
                    raise ValueError("rank-one step failed")
                prediction = module(batch[0])
                return StepOutput(loss=prediction.sum())

            try:
                with Trainer(
                    model=step_failure_model,
                    optimizer=step_failure_optimizer,
                    train_loader=step_failure_loader,
                    train_step=rank_failing_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    accumulation_policy=UnevenAccumulationPolicy(),
                    runtime=runtime,
                ) as step_failure_trainer:
                    step_failure_trainer.fit()
            except RuntimeError as error:
                step_error = str(error)
            else:
                step_error = None
            sampler_failure_model = torch.nn.Linear(1, 1, bias=False)
            sampler_failure_optimizer = torch.optim.SGD(
                sampler_failure_model.parameters(),
                lr=0.0,
            )
            sampler_failure_loader = DataLoader(
                TensorDataset(torch.ones(1, 1)),
                batch_size=1,
                sampler=RankFailingSampler(rank),
            )
            try:
                with Trainer(
                    model=sampler_failure_model,
                    optimizer=sampler_failure_optimizer,
                    train_loader=sampler_failure_loader,
                    train_step=step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    runtime=runtime,
                ) as sampler_failure_trainer:
                    sampler_failure_trainer.fit()
            except RuntimeError as error:
                sampler_error = str(error)
            else:
                sampler_error = None
            restore_model = torch.nn.Linear(1, 1, bias=False)
            restore_optimizer = torch.optim.SGD(restore_model.parameters(), lr=0.0)
            try:
                with Trainer(
                    model=restore_model,
                    optimizer=restore_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    checkpoint_policy=RankFailingRestorePolicy(
                        Path(checkpoint_dir),
                        rank,
                    ),
                    runtime=runtime,
                ) as restore_trainer:
                    restore_trainer.load_checkpoint(Path("unused.pt"))
            except RuntimeError as error:
                restore_error = str(error)
            else:
                restore_error = None
            divergent_model = torch.nn.Linear(1, 1, bias=False)
            divergent_optimizer = torch.optim.SGD(
                divergent_model.parameters(),
                lr=0.0,
            )
            try:
                with Trainer(
                    model=divergent_model,
                    optimizer=divergent_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    checkpoint_policy=RankDivergentRestorePolicy(
                        Path(checkpoint_dir),
                        rank,
                    ),
                    runtime=runtime,
                ) as divergent_trainer:
                    divergent_trainer.load_checkpoint(Path("unused.pt"))
                    divergent_restore_result = divergent_optimizer.param_groups[0]["lr"]
            except RuntimeError as error:
                divergent_restore_result = str(error)
            inspection_model = torch.nn.Linear(1, 1, bias=False)
            inspection_optimizer = torch.optim.SGD(
                inspection_model.parameters(),
                lr=0.0,
            )
            try:
                with Trainer(
                    model=inspection_model,
                    optimizer=inspection_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    checkpoint_policy=UnpickleableInspectionPolicy(
                        Path(checkpoint_dir),
                    ),
                    runtime=runtime,
                ) as inspection_trainer:
                    inspection_trainer.inspect_checkpoint(Path("unused.pt"))
            except RuntimeError as error:
                inspection_error = str(error)
            else:
                inspection_error = None
            callback_model = torch.nn.Linear(1, 1, bias=False)
            callback_optimizer = torch.optim.SGD(callback_model.parameters(), lr=0.0)
            try:
                with Trainer(
                    model=callback_model,
                    optimizer=callback_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    callbacks=(EarlyStopping("loss", patience=2),) if rank == 0 else (),
                    checkpoint_policy=CallbackRestorePolicy(Path(checkpoint_dir)),
                    runtime=runtime,
                ) as callback_trainer:
                    callback_trainer.load_checkpoint(Path("unused.pt"))
            except RuntimeError as error:
                callback_restore_error = str(error)
            else:
                callback_restore_error = None
            metadata_model = torch.nn.Linear(1, 1, bias=False)
            metadata_optimizer = torch.optim.SGD(metadata_model.parameters(), lr=0.0)
            with Trainer(
                model=metadata_model,
                optimizer=metadata_optimizer,
                train_loader=DataLoader(
                    TensorDataset(torch.ones(1, 1)),
                    batch_size=1,
                ),
                train_step=distributed_regression_step,
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    strategy="ddp",
                    checkpoint_every_epochs=None,
                ),
                checkpoint_policy=TensorMetadataRestorePolicy(Path(checkpoint_dir)),
                runtime=runtime,
            ) as metadata_trainer:
                metadata_restore = metadata_trainer.load_checkpoint(Path("unused.pt"))
                metadata_value = metadata_restore.metadata["tensor"].tolist()
            invalid_model = torch.nn.Linear(1, 1, bias=False)
            invalid_optimizer = torch.optim.SGD(invalid_model.parameters(), lr=0.0)
            try:
                with Trainer(
                    model=invalid_model,
                    optimizer=invalid_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    checkpoint_policy=RankInvalidRestorePolicy(Path(checkpoint_dir), rank),
                    runtime=runtime,
                ) as invalid_trainer:
                    invalid_trainer.load_checkpoint(Path("unused.pt"))
            except RuntimeError as error:
                invalid_restore_error = str(error)
            else:
                invalid_restore_error = None
            options_model = torch.nn.Linear(1, 1, bias=False)
            options_optimizer = torch.optim.SGD(options_model.parameters(), lr=0.0)
            try:
                with Trainer(
                    model=options_model,
                    optimizer=options_optimizer,
                    train_loader=DataLoader(
                        TensorDataset(torch.ones(1, 1)),
                        batch_size=1,
                    ),
                    train_step=distributed_regression_step,
                    config=TrainerConfig(
                        epochs=1,
                        device="cpu",
                        strategy="ddp",
                        checkpoint_every_epochs=None,
                    ),
                    checkpoint_policy=RecordingCheckpointPolicy(Path(checkpoint_dir)),
                    runtime=runtime,
                ) as options_trainer:
                    options_trainer.load_checkpoint(
                        Path("unused.pt"),
                        options=None if rank == 0 else RestoreOptions(),
                    )
            except RuntimeError as error:
                options_restore_error = str(error)
            else:
                options_restore_error = None
            result_queue.put(
                (
                    rank,
                    result.state.optimizer_step,
                    [observation.metrics.get("sample") for observation in progress],
                    [observation.metrics.get("batch/count") for observation in progress],
                    [observation.logical_step for observation in progress],
                    partial_weight_delta,
                    last_error,
                    primary_last_metrics,
                    local_metrics,
                    shape_error,
                    metric_compute_error,
                    planning_error,
                    step_error,
                    sampler_error,
                    restore_error,
                    divergent_restore_result,
                    result.state.global_step,
                    inspection_error,
                    callback_restore_error,
                    metadata_value,
                    invalid_restore_error,
                    options_restore_error,
                    None,
                )
            )
    except BaseException as error:
        result_queue.put(
            (
                rank,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                str(error),
            )
        )


def run_uneven_accumulation(tmp_path: Path) -> list[tuple[Any, ...]]:
    """Launch the unequal-window CPU DDP fixture."""
    process_context = multiprocessing.get_context("spawn")
    result_queue = process_context.Queue()
    rendezvous = f"file://{tmp_path / 'uneven-rendezvous'}"
    checkpoint_dir = str(tmp_path / "primary-checkpoints")
    processes = [
        process_context.Process(
            target=_uneven_accumulation_worker,
            args=(rank, rendezvous, checkpoint_dir, result_queue),
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
            pytest.fail("Uneven accumulation worker did not shut down coherently")
        assert process.exitcode == 0
    return sorted(result_queue.get(timeout=5) for _ in processes)


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
    assert all(
        not observation.metrics
        for observation in sink.observations
        if observation.event == "task_completed"
    )
    assert len(regression_result.training_history) == 2
    assert {"loss", "mae"}.issubset(regression_result.training_history[-1])


def test_uneven_ddp_accumulation_reduces_each_logical_batch(tmp_path: Path) -> None:
    results = run_uneven_accumulation(tmp_path)

    assert [result[0] for result in results] == [0, 1]
    assert all(result[1] == 2 for result in results)
    assert all(result[2] == pytest.approx([1.5, 1.5]) for result in results)
    assert all(result[3] == pytest.approx([4.0, 4.0]) for result in results)
    assert all(result[4] == [1, 2] for result in results)
    assert all(result[5] == pytest.approx(-3.5) for result in results), results
    assert all("was not reported on rank 0" in result[6] for result in results)
    assert all(result[7] == {"primary": 9.0} for result in results)
    assert results[0][8] == {}
    assert results[1][8] == {"local": 7.0}
    assert all("tensor metadata differs" in result[9] for result in results)
    assert all(
        "stateful metric computation failed: ValueError: "
        "rank-one metric compute failed" in result[10]
        for result in results
    )
    assert all("accumulation planning failed" in result[11] for result in results)
    assert all(
        "train step failed: ValueError: rank-one step failed" in result[12]
        for result in results
    )
    assert all(
        "train sampler epoch failed: ValueError: rank-one sampler failed" in result[13]
        for result in results
    )
    assert all(
        "checkpoint restore failed: ValueError: rank-one restore failed" in result[14]
        for result in results
    )
    assert all(result[15] == pytest.approx(0.1) for result in results)
    assert all(result[16] == 8 for result in results)
    assert all("checkpoint inspection failed" in result[17] for result in results)
    assert all("checkpoint state application failed" in result[18] for result in results)
    assert all(result[19] == [1, 2] for result in results)
    assert all("checkpoint restore payload contract failed" in result[20] for result in results)
    assert all(
        "checkpoint restore request differs across ranks" in result[21]
        for result in results
    )
    assert all(result[22] is None for result in results)
    assert len(list((tmp_path / "primary-checkpoints").glob("checkpoint-*.pt"))) == 1


def test_incomplete_accumulation_window_requires_an_explicit_scale() -> None:
    plan = AccumulationPlan(
        local_microbatches_per_step=3,
        loss_scale=0.5,
    )

    assert plan.window_sizes(4) == (3, 1)
    assert plan.scale_for_window(3, window_index=0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="explicit window_loss_scales"):
        plan.scale_for_window(1, window_index=1)


@pytest.mark.parametrize(
    ("total_count", "rank_weights", "expected"),
    [
        (16, (3, 1), (12, 4)),
        (6, (3, 1), (4, 2)),
        (14, (3, 1), (10, 4)),
        (12, (3, 2, 1), (6, 4, 2)),
        (7, (1e308, 1e308, 1e308), (2, 2, 3)),
        (1, (2**53 + 1, 2**53), (1, 0)),
        (1, (10**400, 1), (1, 0)),
        (1, (3, 1), (1, 0)),
        (0, (3, 2, 1), (0, 0, 0)),
    ],
)
def test_weighted_partition_counts_support_arbitrary_rank_weights(
    total_count: int,
    rank_weights: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    assert weighted_partition_counts(total_count, rank_weights) == expected


def test_weighted_partition_counts_can_require_work_on_every_rank() -> None:
    assert weighted_partition_counts(4, (100, 1), require_nonempty=True) == (3, 1)
    with pytest.raises(ValueError, match="at least one item per rank"):
        weighted_partition_counts(2, (3, 2, 1), require_nonempty=True)


def test_weighted_task_allocation_uses_projected_normalized_load() -> None:
    assignments = allocate_weighted_tasks(
        [
            ("slide-0", 1200),
            ("slide-1", 800),
            ("slide-2", 500),
            ("slide-3", 300),
        ],
        (3, 1),
    )

    assert [(assignment.task_id, assignment.rank) for assignment in assignments] == [
        ("slide-0", 0),
        ("slide-1", 0),
        ("slide-2", 1),
        ("slide-3", 0),
    ]


def test_weighted_task_allocation_is_input_order_independent() -> None:
    tasks = [(f"task-{index}", cost) for index, cost in enumerate((10, 10, 10, 10))]

    forward = allocate_weighted_tasks(tasks, (3, 1))
    reverse = allocate_weighted_tasks(list(reversed(tasks)), (3, 1))

    assert forward == reverse
    assert [assignment.rank for assignment in forward] == [0, 0, 0, 1]


@pytest.mark.parametrize(
    "tasks",
    [
        [("", 1)],
        [("duplicate", 1), ("duplicate", 2)],
        [("negative", -1)],
        [("infinite", float("inf"))],
        [("boolean", True)],
    ],
)
def test_weighted_task_allocation_rejects_invalid_tasks(
    tasks: list[tuple[str, Any]],
) -> None:
    with pytest.raises(ValueError):
        allocate_weighted_tasks(tasks, (1, 1))


def test_weighted_partition_indices_cover_each_item_once() -> None:
    ranges = tuple(weighted_partition_indices(17, rank, (4, 2, 1)) for rank in range(3))

    assert [len(rank_range) for rank_range in ranges] == [10, 5, 2]
    assert [index for rank_range in ranges for index in rank_range] == list(range(17))


def test_weighted_partition_conserves_totals_above_float_integer_precision() -> None:
    total_count = 2**53 + 3
    counts = weighted_partition_counts(total_count, (1, 1))
    ranges = tuple(weighted_partition_indices(total_count, rank, (1, 1)) for rank in range(2))

    assert sum(counts) == total_count
    assert ranges[0].start == 0
    assert ranges[0].stop == ranges[1].start
    assert ranges[1].stop == total_count


@pytest.mark.parametrize(("rank", "local_count"), [(0, 6), (1, 2)])
def test_weighted_accumulation_policy_scales_one_global_window(
    rank: int,
    local_count: int,
) -> None:
    policy = WeightedAccumulationPolicy(8, (3, 1))
    plan = policy.plan(rank=rank, world_size=2, local_batch_count=local_count * 2)

    assert plan.local_microbatches_per_step == local_count
    assert plan.window_sizes(local_count * 2) == (local_count, local_count)
    assert plan.loss_scale == pytest.approx(0.25)


def test_weighted_accumulation_policy_supports_more_than_two_ranks() -> None:
    policy = WeightedAccumulationPolicy(6, [3, 2, 1])

    plans = tuple(
        policy.plan(rank=rank, world_size=3, local_batch_count=local_count * 2)
        for rank, local_count in enumerate((3, 2, 1))
    )

    assert [plan.local_microbatches_per_step for plan in plans] == [3, 2, 1]
    assert all(plan.loss_scale == pytest.approx(0.5) for plan in plans)


def test_weighted_accumulation_policy_can_preserve_a_fixed_partial_scale() -> None:
    policy = WeightedAccumulationPolicy(4, (1,), partial_window="fixed")
    plan = policy.plan(rank=0, world_size=1, local_batch_count=5)

    assert plan.window_sizes(5) == (4, 1)
    assert plan.scale_for_window(4, window_index=0) == pytest.approx(0.25)
    assert plan.scale_for_window(1, window_index=1) == pytest.approx(0.25)


def test_weighted_batch_sampler_assigns_complete_windows_across_three_ranks() -> None:
    samplers = tuple(
        WeightedDistributedBatchSampler(
            13,
            batch_size=1,
            global_microbatches_per_step=6,
            rank=rank,
            rank_weights=(3, 2, 1),
            shuffle=False,
        )
        for rank in range(3)
    )

    assert [list(sampler) for sampler in samplers] == [
        [[0], [1], [2], [6], [7], [8]],
        [[3], [4], [9], [10]],
        [[5], [11]],
    ]
    assert [len(sampler) for sampler in samplers] == [6, 4, 2]


@pytest.mark.parametrize("seed", [-(2**63), 2**64 - 1])
def test_weighted_batch_sampler_accepts_torch_seed_bounds(seed: int) -> None:
    sampler = WeightedDistributedBatchSampler(
        2,
        batch_size=1,
        global_microbatches_per_step=1,
        rank=0,
        rank_weights=(1,),
        seed=seed,
    )

    assert len(list(sampler)) == 2


def test_weighted_batch_sampler_rejects_out_of_range_shuffle_seed() -> None:
    with pytest.raises(ValueError, match="when shuffling"):
        WeightedDistributedBatchSampler(
            1,
            batch_size=1,
            global_microbatches_per_step=1,
            rank=0,
            rank_weights=(1,),
            seed=2**100,
        )


def test_weighted_batch_sampler_rejects_epoch_seed_overflow() -> None:
    sampler = WeightedDistributedBatchSampler(
        1,
        batch_size=1,
        global_microbatches_per_step=1,
        rank=0,
        rank_weights=(1,),
        seed=2**64 - 1,
    )

    with pytest.raises(ValueError, match=r"seed \+ epoch"):
        sampler.set_epoch(1)


def test_unshuffled_weighted_batch_sampler_ignores_arbitrary_seed() -> None:
    sampler = WeightedDistributedBatchSampler(
        1,
        batch_size=1,
        global_microbatches_per_step=1,
        rank=0,
        rank_weights=(1,),
        seed=2**100,
        shuffle=False,
    )
    sampler.set_epoch(2**100)

    assert list(sampler) == [[0]]


def test_trainer_advances_project_batch_sampler_epochs() -> None:
    batch_sampler = EpochRecordingBatchSampler()
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1)),
        batch_sampler=batch_sampler,
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del context
        return StepOutput(loss=module(batch[0]).sum() * 0)

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=step,
        config=TrainerConfig(
            epochs=2,
            device="cpu",
            checkpoint_every_epochs=None,
        ),
    ) as trainer:
        trainer.fit()

    assert batch_sampler.epochs == [0, 1]


def test_stateful_metrics_and_routes_remain_project_named() -> None:
    features = torch.ones(4, 1)
    loader = DataLoader(TensorDataset(features), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    def step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del context
        prediction = module(batch[0])
        return StepOutput(
            loss=prediction.sum() * 0,
            metric_updates={"project": len(batch[0])},
        )

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        train_stateful_metrics={"project": AdditiveCountMetric()},
        train_metric_routes={
            "count": MetricRoute(batch_name=None, epoch_name="project/count")
        },
        observer=RunObserver((sink,)),
    ) as trainer:
        result = trainer.fit()

    assert result.training_history == ({"count": 4.0, "loss": 0.0},)
    completed = [
        observation
        for observation in sink.observations
        if observation.event == "task_completed"
    ]
    assert completed[-1].metrics == {"project/count": 4.0}


def test_validation_progress_preserves_routed_dense_metrics() -> None:
    features = torch.ones(2, 1)
    loader = DataLoader(TensorDataset(features), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    def train_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del context
        return StepOutput(loss=module(batch[0]).sum() * 0)

    def validation_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del module, batch, context
        return StepOutput(metrics={"score": 2.0})

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=train_step,
        validation_loader=loader,
        validation_step=validation_step,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            checkpoint_every_epochs=None,
            display_metric_names=("validation/score",),
        ),
        validation_metric_routes={
            "score": MetricRoute(
                batch_name="validation/score",
                epoch_name="validation/score_epoch",
            )
        },
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    progress = [
        observation
        for observation in sink.observations
        if observation.event == "progress"
        and observation.fields.get("phase") == "validation"
    ]
    assert [observation.metrics for observation in progress] == [
        {"validation/score": 2.0},
        {"validation/score": 2.0},
    ]
    assert [observation.display_metrics for observation in progress] == [
        {"validation/score": 2.0},
        {"validation/score": 2.0},
    ]


def test_empty_training_loader_remains_a_completed_noop_epoch() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    def unreachable_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        raise AssertionError("empty loader invoked its step function")

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=unreachable_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        observer=RunObserver((sink,)),
    ) as trainer:
        result = trainer.fit()

    assert result.training_history == ({},)
    assert result.state.state_dict() == {
        "epoch": 0,
        "global_step": 0,
        "optimizer_step": 0,
        "stopped_early": False,
    }
    assert [observation.event for observation in sink.observations][-2:] == [
        "task_completed",
        "phase_completed",
    ]


def test_stateful_reset_failure_balances_task_and_phase_events() -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()
    with pytest.raises(RuntimeError, match="reset failed"), Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=lambda module, batch, context: StepOutput(
            loss=module(batch[0]).sum() * 0
        ),
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        train_stateful_metrics={"broken": RaisingResetMetric()},
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    assert [observation.event for observation in sink.observations] == [
        "phase_started",
        "task_started",
        "task_failed",
        "phase_failed",
    ]


@pytest.mark.parametrize("hook", ["start", "end"])
def test_callback_failure_reports_phase_failed_without_phase_completed(hook: str) -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()
    with pytest.raises(RuntimeError, match=f"{hook} callback failed"), Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=lambda module, batch, context: StepOutput(
            loss=module(batch[0]).sum() * 0
        ),
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        callbacks=(RaisingCallback(hook),),
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    events = [observation.event for observation in sink.observations]
    assert events.count("phase_failed") == 1
    assert "phase_completed" not in events


def test_trainer_can_leave_outer_fit_phase_lifecycle_to_its_caller() -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=lambda module, batch, context: StepOutput(
            loss=module(batch[0]).sum() * 0
        ),
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            checkpoint_every_epochs=None,
            emit_fit_phase_events=False,
        ),
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    assert not any(
        observation.event.startswith("phase_")
        and observation.fields.get("phase") == "train"
        for observation in sink.observations
    )
    assert [observation.event for observation in sink.observations] == [
        "task_started",
        "progress",
        "task_completed",
    ]


def test_duplicate_train_epoch_routes_fail_with_balanced_task_events() -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    def step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del context
        return StepOutput(
            loss=module(batch[0]).sum() * 0,
            metrics={"a": 1.0, "b": 2.0},
        )

    duplicate_routes = {
        "a": MetricRoute(batch_name=None, epoch_name="duplicate"),
        "b": MetricRoute(batch_name=None, epoch_name="duplicate"),
    }
    with pytest.raises(ValueError, match="multiple metrics route"), Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        train_metric_routes=duplicate_routes,
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    assert [observation.event for observation in sink.observations][-2:] == [
        "task_failed",
        "phase_failed",
    ]


def test_duplicate_validation_epoch_routes_balance_both_lifecycle_scopes() -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    def train_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del context
        return StepOutput(loss=module(batch[0]).sum() * 0)

    def validation_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del module, batch, context
        return StepOutput(metrics={"a": 1.0, "b": 2.0})

    duplicate_routes = {
        "a": MetricRoute(batch_name=None, epoch_name="duplicate"),
        "b": MetricRoute(batch_name=None, epoch_name="duplicate"),
    }
    with pytest.raises(ValueError, match="multiple metrics route"), Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=train_step,
        validation_loader=loader,
        validation_step=validation_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        validation_metric_routes=duplicate_routes,
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    events = [observation.event for observation in sink.observations]
    assert events[-3:] == ["task_failed", "phase_failed", "phase_failed"]


def test_checkpoint_heartbeat_does_not_reactivate_completed_epoch_task(
    tmp_path: Path,
) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()
    observer = RunObserver((sink,), heartbeat_interval_seconds=0.01)
    policy = SlowRecordingCheckpointPolicy(tmp_path / "slow-checkpoint")
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=tmp_path / "slow-checkpoint",
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
        observer=observer,
    ) as trainer:
        trainer.fit()

    heartbeats = [
        observation
        for observation in sink.observations
        if observation.event == "heartbeat"
        and observation.fields.get("message") == "Checkpoint publication is still active."
    ]
    assert heartbeats
    assert all("task_id" not in observation.fields for observation in heartbeats)


def test_project_checkpoint_policy_uses_ordered_publication_and_restore(
    tmp_path: Path,
) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy = RecordingCheckpointPolicy(tmp_path / "project-checkpoints")
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=tmp_path / "project-checkpoints",
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
    ) as trainer:
        trainer.fit()

    assert (tmp_path / "project-checkpoints" / "latest_epoch_0.pt").read_text(
        encoding="utf-8"
    ) == "epoch=0\n"
    assert len(policy.contexts) == 1
    assert policy.contexts[0].training_metrics.keys() == {"loss", "mae"}

    restored_model = torch.nn.Linear(1, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.0)
    with Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=4, device="cpu", checkpoint_every_epochs=None),
        checkpoint_policy=policy,
    ) as trainer:
        trainer.load_checkpoint(tmp_path / "opaque.pt")
        assert trainer.state.state_dict() == {
            "epoch": 2,
            "global_step": 7,
            "optimizer_step": 3,
            "stopped_early": False,
        }


def test_checkpoint_save_policy_validates_configuration() -> None:
    for mode in ("", "newest", 1):
        with pytest.raises(ValueError, match="checkpoint mode"):
            CheckpointSavePolicy(mode=cast(Any, mode))
    for every_epochs in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            CheckpointSavePolicy(every_epochs=cast(Any, every_epochs))
    with pytest.raises(ValueError, match="boolean"):
        CheckpointSavePolicy(save_best=cast(Any, 1))


def test_trainer_delivers_checkpoint_receipts_to_callbacks_and_observer(
    tmp_path: Path,
) -> None:
    class PublicationCallback(Callback):
        def __init__(self) -> None:
            self.publications: list[CheckpointPublication] = []

        def on_checkpoint_published(
            self,
            state: TrainerState,
            publication: CheckpointPublication,
        ) -> None:
            assert state.epoch == 0
            self.publications.append(publication)

    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    callback = PublicationCallback()
    sink = RecordingSink()
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=tmp_path,
        checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
        callbacks=(callback,),
        observer=RunObserver((sink,)),
    ) as trainer:
        trainer.fit()

    assert len(callback.publications) == 1
    receipt = callback.publications[0].published[0]
    assert receipt.path == (tmp_path / "latest_epoch_0.pt").resolve()
    assert receipt.role == "latest"
    assert receipt.epoch == 0
    assert receipt.size_bytes == receipt.path.stat().st_size
    assert receipt.sha256 == hashlib.sha256(receipt.path.read_bytes()).hexdigest()
    publication_events = [
        observation
        for observation in sink.observations
        if observation.event == "task_completed"
        and observation.fields.get("task_id") == "checkpoint-publication"
    ]
    assert len(publication_events) == 1
    assert publication_events[0].fields["checkpoints"] == [
        {
            "path": str(receipt.path),
            "role": "latest",
            "epoch": 0,
            "size_bytes": receipt.size_bytes,
            "sha256": receipt.sha256,
        }
    ]


def test_checkpoint_receipt_callback_failure_surfaces_from_flush(tmp_path: Path) -> None:
    class FailingPublicationCallback(Callback):
        def on_checkpoint_published(
            self,
            state: TrainerState,
            publication: CheckpointPublication,
        ) -> None:
            del state, publication
            raise RuntimeError("receipt consumer failed")

    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_dir=tmp_path,
        checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
        callbacks=(FailingPublicationCallback(),),
    )
    trainer.publish_checkpoint(
        epoch=0,
        training_metrics={},
        validation_metrics=None,
    )
    with pytest.raises(RuntimeError, match="receipt consumer failed"):
        trainer.flush_local_checkpoints(message="testing receipt failure")
    trainer.close()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("all", {"epoch_0.pt", "epoch_1.pt"}),
        ("latest", {"latest_epoch_1.pt"}),
    ],
)
def test_checkpoint_save_policy_owns_resumable_retention(
    tmp_path: Path,
    mode: Literal["all", "latest"],
    expected: set[str],
) -> None:
    policy = CheckpointSavePolicy(mode=mode, save_best=False)
    for epoch in range(2):
        writers = TrainerCheckpointWriters(
            resumable=lambda path, epoch=epoch: path.write_text(
                str(epoch), encoding="utf-8"
            )
        )
        publish_checkpoint_plan(
            checkpoint_module.build_trainer_checkpoint_plan(
                tmp_path,
                epoch=epoch,
                save_policy=policy,
                writers=writers,
                save_resumable=True,
                save_best=False,
            )
        )

    assert {path.name for path in tmp_path.glob("*.pt")} == expected


def test_failed_latest_write_preserves_previous_checkpoint(tmp_path: Path) -> None:
    previous = tmp_path / "latest_epoch_0.pt"
    previous.write_text("previous", encoding="utf-8")

    def fail_writer(path: Path) -> None:
        del path
        raise RuntimeError("serialization failed")

    plan = checkpoint_module.build_trainer_checkpoint_plan(
        tmp_path,
        epoch=1,
        save_policy=CheckpointSavePolicy(save_best=False),
        writers=TrainerCheckpointWriters(resumable=fail_writer),
        save_resumable=True,
        save_best=False,
    )
    with pytest.raises(RuntimeError, match="serialization failed"):
        publish_checkpoint_plan(plan)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not (tmp_path / "latest_epoch_1.pt").exists()


def test_best_checkpoint_follows_improvement_independent_of_cadence(
    tmp_path: Path,
) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    values = (3.0, 2.0, 4.0)
    policy = RecordingCheckpointPolicy(tmp_path)

    def validation_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del module, batch
        return StepOutput(metrics={"score": values[context.epoch]})

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        validation_loader=loader,
        validation_step=validation_step,
        config=TrainerConfig(epochs=3, device="cpu"),
        callbacks=(EarlyStopping("score", patience=3),),
        checkpoint_dir=tmp_path,
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(every_epochs=3),
    ) as trainer:
        trainer.fit()

    assert (tmp_path / "best.safetensors").read_text(encoding="utf-8") == (
        "best_epoch=1\n"
    )
    assert (tmp_path / "latest_epoch_2.pt").read_text(encoding="utf-8") == (
        "epoch=2\n"
    )
    assert [context.epoch for context in policy.contexts] == [0, 1, 2]


def test_save_best_requires_one_early_stopping_callback(tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def validation_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del module, batch, context
        return StepOutput(metrics={"score": 1.0})

    with pytest.raises(ValueError, match="exactly one"):
        Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_step=regression_step,
            validation_loader=loader,
            validation_step=validation_step,
            config=TrainerConfig(epochs=1, device="cpu"),
            checkpoint_dir=tmp_path,
            checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
            checkpoint_save_policy=CheckpointSavePolicy(),
        )

    second_model = torch.nn.Linear(1, 1)
    second_optimizer = torch.optim.SGD(second_model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="exactly one"):
        Trainer(
            model=second_model,
            optimizer=second_optimizer,
            train_loader=loader,
            train_step=regression_step,
            validation_loader=loader,
            validation_step=validation_step,
            config=TrainerConfig(epochs=1, device="cpu"),
            callbacks=(
                EarlyStopping("score", patience=1),
                EarlyStopping("score", patience=1),
            ),
            checkpoint_dir=tmp_path,
            checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
            checkpoint_save_policy=CheckpointSavePolicy(),
        )


def test_save_best_can_be_disabled_without_validation_or_early_stopping(
    tmp_path: Path,
) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=tmp_path,
        checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
    ) as trainer:
        trainer.fit()

    assert (tmp_path / "latest_epoch_0.pt").is_file()
    assert not (tmp_path / "best.safetensors").exists()


def test_early_stopping_epoch_does_not_replace_best_checkpoint(tmp_path: Path) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    values = (1.0, 2.0)

    def validation_step(
        module: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del module, batch
        return StepOutput(metrics={"score": values[context.epoch]})

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        validation_loader=loader,
        validation_step=validation_step,
        config=TrainerConfig(epochs=3, device="cpu"),
        callbacks=(EarlyStopping("score", patience=1),),
        checkpoint_dir=tmp_path,
        checkpoint_policy=RecordingCheckpointPolicy(tmp_path),
        checkpoint_save_policy=CheckpointSavePolicy(),
    ) as trainer:
        result = trainer.fit()

    assert result.state.stopped_early
    assert result.state.epoch == 1
    assert (tmp_path / "best.safetensors").read_text(encoding="utf-8") == (
        "best_epoch=0\n"
    )


def test_typed_checkpoint_inspection_selects_generic_restore_and_reset(
    tmp_path: Path,
) -> None:
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    source_model = torch.nn.Linear(1, 1)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.2, momentum=0.9)
    source_scheduler = WarmupLinearLR(source_optimizer, warmup_ratio=0.0, total_steps=8)
    source_model(torch.ones(1, 1)).sum().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_callback = EarlyStopping("loss", patience=3)
    source_callback.load_state_dict({"best": 0.5, "bad_checks": 2})

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = WarmupLinearLR(optimizer, warmup_ratio=0.0, total_steps=8)
    callback = EarlyStopping("loss", patience=3)
    policy = TypedStateCheckpointPolicy(
        tmp_path,
        optimizer_state=source_optimizer.state_dict(),
        scheduler_state=source_scheduler.state_dict(),
        callback_state=source_callback.state_dict(),
    )
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=4, device="cpu", checkpoint_every_epochs=None),
        scheduler=scheduler,
        callbacks=(callback,),
        checkpoint_policy=policy,
    ) as trainer:
        inspection = trainer.inspect_checkpoint(tmp_path / "typed.pt")
        restored = trainer.load_checkpoint(
            tmp_path / "typed.pt",
            options=inspection.restore_options,
        )

        assert inspection.metadata == {"objective_changed": True}
        assert trainer.state.state_dict() == {
            "epoch": 1,
            "global_step": 8,
            "optimizer_step": 4,
            "stopped_early": False,
        }
        assert callback.state_dict() == {"best": None, "bad_checks": 0}
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.175)
        assert scheduler.last_epoch == 1
        assert restored.restored_components == frozenset(
            {"model", "optimizer", "scheduler", "trainer"}
        )
        assert restored.reset_components == frozenset({"callbacks", "stopped_early"})
        assert restored.optimizer_state_dict is None
        assert restored.scheduler_state_dict is None
        assert restored.callback_state_dicts == {}


def test_restore_options_reset_optimizer_scheduler_and_restore_terminal_callback(
    tmp_path: Path,
) -> None:
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = WarmupLinearLR(optimizer, warmup_ratio=0.0, total_steps=4)
    callback = EarlyStopping("loss", patience=3)
    policy = TypedStateCheckpointPolicy(
        tmp_path,
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=scheduler.state_dict(),
        callback_state={"best": 0.25, "bad_checks": 3},
    )
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=4, device="cpu", checkpoint_every_epochs=None),
        scheduler=scheduler,
        callbacks=(callback,),
        checkpoint_policy=policy,
    ) as trainer:
        optimizer.param_groups[0]["lr"] = 0.5
        optimizer.step()
        scheduler.step()
        restored = trainer.load_checkpoint(
            tmp_path / "typed.pt",
            options=RestoreOptions(
                optimizer="reset",
                scheduler="reset",
            ),
        )

        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        assert scheduler.last_epoch == 0
        assert callback.state_dict() == {"best": 0.25, "bad_checks": 3}
        assert trainer.state.stopped_early
        parameter_before_fit = model.weight.detach().clone()
        terminal_result = trainer.fit()
        torch.testing.assert_close(model.weight, parameter_before_fit)
        assert terminal_result.training_history == ()
        assert terminal_result.validation_history == ()
        assert restored.restored_components == frozenset(
            {"model", "callbacks", "trainer", "stopped_early"}
        )
        assert restored.reset_components == frozenset({"optimizer", "scheduler"})


@pytest.mark.parametrize(
    ("optimizer_action", "scheduler_action", "expected_lr"),
    [
        ("reset", "restore", 0.175),
        ("restore", "reset", 0.01),
    ],
)
def test_mixed_optimizer_scheduler_restore_actions_synchronize_learning_rate(
    tmp_path: Path,
    optimizer_action: Literal["restore", "reset"],
    scheduler_action: Literal["restore", "reset"],
    expected_lr: float,
) -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    source_model = torch.nn.Linear(1, 1)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.2)
    source_scheduler = WarmupLinearLR(source_optimizer, warmup_ratio=0.0, total_steps=8)
    source_optimizer.step()
    source_scheduler.step()

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = WarmupLinearLR(optimizer, warmup_ratio=0.0, total_steps=8)
    policy = TypedStateCheckpointPolicy(
        tmp_path,
        optimizer_state=source_optimizer.state_dict(),
        scheduler_state=source_scheduler.state_dict(),
        callback_state={},
    )
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=2, device="cpu", checkpoint_every_epochs=None),
        scheduler=scheduler,
        checkpoint_policy=policy,
    ) as trainer:
        trainer.load_checkpoint(
            tmp_path / "typed.pt",
            options=RestoreOptions(
                optimizer=optimizer_action,
                scheduler=scheduler_action,
                callbacks="reset",
                stopped_early="reset",
            ),
        )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert scheduler.get_last_lr() == pytest.approx([expected_lr])


def test_checkpoint_policy_cannot_pre_report_generic_resets(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_policy=FalseResetReportingPolicy(tmp_path),
    ) as trainer, pytest.raises(ValueError, match="cannot pre-report Mammoth-managed"):
        trainer.load_checkpoint(tmp_path / "typed.pt")


def test_checkpoint_restore_infers_missing_loop_cursors(tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.ones(3, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(
            epochs=4,
            device="cpu",
            gradient_accumulation_steps=2,
            checkpoint_every_epochs=None,
        ),
        checkpoint_policy=CursorlessCheckpointPolicy(tmp_path),
    ) as trainer:
        trainer.load_checkpoint(tmp_path / "opaque.pt")

        assert trainer.state.epoch == 1
        assert trainer.state.optimizer_step == 4
        assert trainer.state.global_step == 6


def test_checkpoint_restore_accepts_pretraining_coordinate(tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_policy=InitialCheckpointPolicy(tmp_path),
    ) as trainer:
        trainer.load_checkpoint(tmp_path / "pretraining.pt")

        assert trainer.state.state_dict() == {
            "epoch": -1,
            "global_step": 0,
            "optimizer_step": 0,
            "stopped_early": False,
        }


def test_forced_checkpoint_reports_interruption_reason(tmp_path: Path) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy = RecordingCheckpointPolicy(tmp_path / "project-checkpoints")
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_dir=tmp_path / "project-checkpoints",
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
    ) as trainer:
        trainer.publish_checkpoint_now(reason="interrupted")

    assert policy.contexts[-1].reason == "interrupted"
    assert (tmp_path / "project-checkpoints" / "latest_epoch_-1.pt").is_file()


def test_fit_publishes_interrupted_checkpoint_before_reraising(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "interrupted-checkpoint"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / "best.safetensors"
    best_checkpoint.write_text("existing best", encoding="utf-8")
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy = RecordingCheckpointPolicy(checkpoint_dir)

    class InterruptingCallback(Callback):
        def on_train_start(self, state: TrainerState) -> None:
            del state
            raise KeyboardInterrupt("stop training")

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_dir=checkpoint_dir,
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
        callbacks=(InterruptingCallback(),),
    ) as trainer, pytest.raises(KeyboardInterrupt, match="stop training"):
        trainer.all_gather_object = lambda value: pytest.fail(
            f"interruption checkpoint entered a collective with {value!r}"
        )
        trainer.fit()

    assert [context.reason for context in policy.contexts] == ["interrupted"]
    assert (checkpoint_dir / "latest_epoch_-1.pt").is_file()
    assert best_checkpoint.read_text(encoding="utf-8") == "existing best"


def test_ddp_interrupt_reaches_primary_checkpoint_policy(tmp_path: Path) -> None:
    """A peer-rank KeyboardInterrupt remains an interrupt on every rank."""
    results = run_distributed_interrupt(tmp_path)

    assert [result[1] for result in results] == ["KeyboardInterrupt", "KeyboardInterrupt"]
    assert "train step interrupted" in results[0][2]
    assert "stop distributed training" in results[1][2]
    assert results[0][3] == ["interrupted"]
    assert results[1][3] == []
    assert (tmp_path / "interrupted-checkpoint" / "latest_epoch_-1.pt").is_file()


def test_fit_can_disable_interrupted_checkpoint_publication(tmp_path: Path) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy = RecordingCheckpointPolicy(tmp_path / "disabled-interrupted-checkpoint")

    def interrupt_step(
        model: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        del model, batch, context
        raise KeyboardInterrupt("stop training")

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=interrupt_step,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            checkpoint_every_epochs=None,
            checkpoint_on_interrupt=False,
        ),
        checkpoint_dir=tmp_path / "disabled-interrupted-checkpoint",
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
    ) as trainer, pytest.raises(KeyboardInterrupt, match="stop training"):
        trainer.fit()

    assert policy.contexts == []
    assert not (tmp_path / "disabled-interrupted-checkpoint").exists()


def test_nonprimary_interruption_does_not_enter_checkpoint_collectives(
    tmp_path: Path,
) -> None:
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy = RecordingCheckpointPolicy(tmp_path / "nonprimary-interrupted-checkpoint")

    class InterruptingCallback(Callback):
        def on_train_start(self, state: TrainerState) -> None:
            del state
            raise KeyboardInterrupt("stop nonprimary training")

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
        checkpoint_dir=tmp_path / "nonprimary-interrupted-checkpoint",
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
        callbacks=(InterruptingCallback(),),
    ) as trainer, pytest.raises(KeyboardInterrupt, match="stop nonprimary training"):
        trainer.rank = 1
        trainer.all_gather_object = lambda value: pytest.fail(
            f"nonprimary interruption entered a collective with {value!r}"
        )
        trainer.fit()

    assert policy.contexts == []
    assert not (tmp_path / "nonprimary-interrupted-checkpoint").exists()


def test_compile_runs_after_ddp_and_preserves_ddp_no_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDDP(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.module(value)

        def no_sync(self) -> Any:
            return nullcontext()

    class CompiledModule(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.module(value)

    observed: dict[str, Any] = {}
    base_model = torch.nn.Linear(1, 1)
    ddp_model = FakeDDP(base_model)
    compiled_model = CompiledModule(ddp_model)

    monkeypatch.setattr(trainer_module, "DistributedDataParallel", FakeDDP)
    monkeypatch.setattr(trainer_module, "distributed_identity", lambda strategy: (0, 1))
    monkeypatch.setattr(trainer_module, "wrap_model", lambda model, device, strategy: ddp_model)

    def fake_compile(model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        observed.update(model=model, kwargs=kwargs)
        return compiled_model

    monkeypatch.setattr(torch, "compile", fake_compile)
    optimizer = torch.optim.SGD(base_model.parameters(), lr=0.0)
    with Trainer(
        model=base_model,
        optimizer=optimizer,
        train_loader=DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
        train_step=distributed_regression_step,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            strategy="ddp",
            checkpoint_every_epochs=None,
            compile_config=TorchCompileConfig(mode="default", fullgraph=False),
        ),
    ) as trainer:
        assert trainer.base_model is base_model
        assert observed["model"] is ddp_model
        assert trainer.execution_model is compiled_model
        assert trainer.gradient_accumulation_context() is not None


def test_compile_config_rejects_mode_with_explicit_options() -> None:
    with pytest.raises(ValueError, match="mode and options are mutually exclusive"):
        TorchCompileConfig(mode="default", options={"epilogue_fusion": True})


def test_compile_config_rejects_empty_backend_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        TorchCompileConfig(backend="")


def test_project_checkpoint_policy_plans_after_publisher_backpressure(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "project-checkpoints"
    checkpoint_root.mkdir()
    writer_started = threading.Event()
    release_writer = threading.Event()
    captures: list[int] = []

    class RetainingPolicy(RecordingCheckpointPolicy):
        def capture(self, context: TrainerCheckpointContext) -> TrainerCheckpointWriters:
            captures.append(context.epoch)

            def writer(path: Path) -> None:
                if context.epoch == 0:
                    writer_started.set()
                    if not release_writer.wait(timeout=5):
                        raise TimeoutError("test did not release checkpoint writer")
                path.write_text(str(context.epoch), encoding="utf-8")

            return TrainerCheckpointWriters(resumable=writer)

    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=2, device="cpu"),
        checkpoint_dir=checkpoint_root,
        checkpoint_policy=RetainingPolicy(checkpoint_root),
        checkpoint_save_policy=CheckpointSavePolicy(save_best=False),
    )
    errors: list[BaseException] = []

    def fit() -> None:
        try:
            trainer.fit()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=fit)
    worker.start()
    assert writer_started.wait(timeout=5)
    worker.join(timeout=0.1)
    assert worker.is_alive()
    assert captures == [0]
    release_writer.set()
    worker.join(timeout=5)
    trainer.close()

    assert not worker.is_alive()
    assert errors == []
    assert captures == [0, 1]
    assert [path.name for path in checkpoint_root.glob("latest_epoch_*.pt")] == [
        "latest_epoch_1.pt"
    ]


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
    with pytest.raises(ValueError, match="distinct sink names"):
        MetricRoute(batch_name="loss", epoch_name="loss")


@pytest.mark.parametrize("value", [0, False, "2"])
def test_uniform_accumulation_rejects_invalid_window_size(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        UniformAccumulationPolicy(value)


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


def test_early_stopping_stops_on_patience_threshold_and_signals_improvement() -> None:
    callback = EarlyStopping("score", mode="max", patience=3)
    state = TrainerState()

    assert not callback.should_stop(state)
    callback.on_validation_end(state, {"score": 1.0})
    assert callback.improved
    assert not callback.should_stop(state)

    for bad_check in range(1, 4):
        callback.on_validation_end(state, {"score": 1.0})
        assert not callback.improved
        assert callback.bad_checks == bad_check
        assert callback.should_stop(state) is (bad_check >= 3)

    assert callback.state_dict() == {"best": 1.0, "bad_checks": 3}


def test_restored_early_stop_makes_fit_a_no_op() -> None:
    class FailingCallback(Callback):
        def on_train_start(self, state: TrainerState) -> None:
            raise AssertionError("callbacks must not run for a terminal restored state")

    model = torch.nn.Linear(1, 1)
    initial_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer_state = optimizer.state_dict()
    sink = RecordingSink()

    def train_step(module: torch.nn.Module, batch: Any, context: StepContext) -> StepOutput:
        raise AssertionError("the loader and step function must not run")

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
        train_step=train_step,
        callbacks=(FailingCallback(),),
        observer=RunObserver((sink,)),
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_every_epochs=None),
    ) as trainer:
        trainer.state.stopped_early = True
        result = trainer.fit()

    assert result.state is trainer.state
    assert result.training_history == ()
    assert result.validation_history == ()
    assert sink.observations == []
    assert optimizer.state_dict() == optimizer_state
    for initial, current in zip(initial_parameters, model.parameters(), strict=True):
        assert torch.equal(initial, current)


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


def test_registered_checkpoint_non_strict_restore_ignores_removed_callback(
    tmp_path: Path,
) -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.tensor([0, 1])),
        batch_size=2,
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_dir = tmp_path / "checkpoints"
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=classification_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        callbacks=(EarlyStopping("loss", patience=2),),
        checkpoint_dir=checkpoint_dir,
    ) as trainer:
        trainer.fit()

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    with Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        train_loader=loader,
        train_step=classification_step,
        config=TrainerConfig(epochs=2, device="cpu", checkpoint_every_epochs=None),
    ) as restored:
        report = restored.load_checkpoint(
            checkpoint_dir / "checkpoint-0000.pt",
            strict=False,
        )

    assert "callbacks" not in report.restored_components
    assert report.epoch == 0


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
                CheckpointArtifact(
                    first,
                    writer("best", b"best"),
                    role="best",
                    epoch=1,
                ),
                CheckpointArtifact(
                    second,
                    writer("resume", b"resume"),
                    role="latest",
                    epoch=1,
                ),
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
        published=(
            PublishedCheckpoint(
                path=first.resolve(),
                role="best",
                epoch=1,
                size_bytes=4,
                sha256=hashlib.sha256(b"best").hexdigest(),
            ),
            PublishedCheckpoint(
                path=second.resolve(),
                role="latest",
                epoch=1,
                size_bytes=6,
                sha256=hashlib.sha256(b"resume").hexdigest(),
            ),
        ),
        retired=(previous,),
    )
    assert first.read_bytes() == b"best"
    assert second.read_bytes() == b"resume"
    assert not previous.exists()
    assert retained.read_bytes() == b"retained"
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_checkpoint_artifact_preserves_legacy_positional_mode_arguments(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "legacy-mode.pt"

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"checkpoint")

    artifact = CheckpointArtifact(destination, write, 0o640, False)
    publish_checkpoint_plan(
        CheckpointPlan(
            checkpoint_root=tmp_path,
            artifacts=(artifact,),
        )
    )

    assert artifact.mode == 0o640
    assert artifact.preserve_permissions is False
    assert artifact.role == "epoch"
    assert artifact.epoch == -1
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_checkpoint_receipt_supports_unreadable_final_artifact_mode(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "sealed.pt"
    payload = b"sealed checkpoint"

    publication = publish_checkpoint_plan(
        CheckpointPlan(
            checkpoint_root=tmp_path,
            artifacts=(
                CheckpointArtifact(
                    destination,
                    lambda temporary: temporary.write_bytes(payload),
                    0o000,
                    False,
                ),
            ),
        )
    )

    assert stat.S_IMODE(destination.stat().st_mode) == 0o000
    assert publication.published == (
        PublishedCheckpoint(
            path=destination.resolve(),
            role="epoch",
            epoch=-1,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_checkpoint_receipt_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    destination = tmp_path / "fifo.pt"

    with pytest.raises(FileNotFoundError, match="did not create a file"):
        publish_checkpoint_plan(
            CheckpointPlan(
                checkpoint_root=tmp_path,
                artifacts=(
                    CheckpointArtifact(
                        destination,
                        lambda temporary: os.mkfifo(temporary),
                    ),
                ),
            )
        )


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


def test_registered_checkpoint_publication_does_not_require_ordered_plan_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "supports_dir_fd", set())
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        config=TrainerConfig(epochs=1, device="cpu"),
        checkpoint_dir=tmp_path,
    ) as trainer:
        trainer.fit()

    assert (tmp_path / "checkpoint-0000.pt").is_file()


def test_generic_receipt_publication_replaces_destination_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"sentinel")
    destination = tmp_path / "checkpoint.pt"
    destination.symlink_to(outside)

    with AsyncCheckpointPublisher() as publisher:
        future = publisher.publish_with_receipt(
            destination,
            {"value": torch.tensor([1])},
            role="epoch",
            epoch=0,
        )
        publisher.flush()
        assert future.done()

    receipt = future.result().published[0]
    assert outside.read_bytes() == b"sentinel"
    assert destination.is_file()
    assert not destination.is_symlink()
    assert receipt.path == destination.absolute()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_generic_receipt_rejects_fifo_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_fifo(payload: Any, temporary: Path) -> None:
        del payload
        os.mkfifo(temporary)

    monkeypatch.setattr(torch, "save", write_fifo)
    publisher = AsyncCheckpointPublisher()
    future = publisher.publish_with_receipt(
        tmp_path / "checkpoint.pt",
        {"value": 1},
        role="epoch",
        epoch=0,
    )
    with pytest.raises(FileNotFoundError, match="did not create"):
        future.result(timeout=1)
    with pytest.raises(FileNotFoundError, match="did not create"):
        publisher.close()


def test_generic_receipt_publication_applies_backpressure_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    snapshots: list[int] = []
    original_publish = checkpoint_module.publish_torch_payload_with_receipt

    def record_snapshot(payload: Mapping[str, Any]) -> Any:
        snapshots.append(cast(int, payload["sequence"]))
        return dict(payload)

    def blocking_publish(
        path: Path,
        payload: Mapping[str, Any],
        *,
        role: checkpoint_module.CheckpointRole,
        epoch: int,
    ) -> CheckpointPublication:
        if payload["sequence"] == 1:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release generic checkpoint writer")
        return original_publish(path, payload, role=role, epoch=epoch)

    monkeypatch.setattr(checkpoint_module, "snapshot_to_cpu", record_snapshot)
    monkeypatch.setattr(
        checkpoint_module,
        "publish_torch_payload_with_receipt",
        blocking_publish,
    )
    errors: list[BaseException] = []
    with AsyncCheckpointPublisher(max_pending=1) as publisher:
        publisher.publish_with_receipt(
            tmp_path / "first.pt",
            {"sequence": 1},
            role="epoch",
            epoch=0,
        )
        assert started.wait(timeout=5)

        def publish_second() -> None:
            try:
                publisher.publish_with_receipt(
                    tmp_path / "second.pt",
                    {"sequence": 2},
                    role="epoch",
                    epoch=1,
                )
            except BaseException as error:
                errors.append(error)

        submitter = threading.Thread(target=publish_second)
        submitter.start()
        submitter.join(timeout=0.1)
        assert submitter.is_alive()
        assert snapshots == [1]
        release.set()
        submitter.join(timeout=5)
        assert not submitter.is_alive()
        publisher.flush()

    assert errors == []
    assert snapshots == [1, 2]


def test_checkpoint_failure_during_snapshot_preserves_admitted_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slot admitted before snapshot remains usable if prior work then fails."""
    first_started = threading.Event()
    release_first = threading.Event()
    second_snapshot_started = threading.Event()
    allow_second_snapshot = threading.Event()
    real_snapshot = checkpoint_module.snapshot_to_cpu

    def fail_first(temporary: Path) -> None:
        first_started.set()
        assert release_first.wait(timeout=5.0)
        temporary.write_bytes(b"partial")
        raise OSError("first publication failed")

    def snapshot(payload: Any) -> Any:
        if isinstance(payload, Mapping) and payload.get("sequence") == 2:
            second_snapshot_started.set()
            assert allow_second_snapshot.wait(timeout=5.0)
        return real_snapshot(payload)

    monkeypatch.setattr(checkpoint_module, "snapshot_to_cpu", snapshot)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    publisher = AsyncCheckpointPublisher(max_pending=2)
    first_future = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "first.pt", fail_first),),
        )
    )
    assert first_started.wait(timeout=5.0)
    first_completed = threading.Event()
    first_future.add_done_callback(lambda _: first_completed.set())
    submitted: list[Future[Path]] = []
    errors: list[BaseException] = []

    def submit_second() -> None:
        try:
            submitted.append(
                publisher.publish(
                    checkpoint_root / "second.pt",
                    {"sequence": 2},
                )
            )
        except BaseException as error:
            errors.append(error)

    submitter = threading.Thread(target=submit_second)
    submitter.start()
    assert second_snapshot_started.wait(timeout=5.0)
    release_first.set()
    assert first_completed.wait(timeout=5.0)
    allow_second_snapshot.set()
    submitter.join(timeout=5.0)

    assert not submitter.is_alive()
    assert errors == []
    assert len(submitted) == 1
    assert submitted[0].result(timeout=5.0) == checkpoint_root / "second.pt"
    with pytest.raises(OSError, match="first publication failed"):
        publisher.flush()
    publisher.close()


def test_checkpoint_failure_survives_interrupted_adapter_acknowledgment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter ownership preserves a raw failure before generic acknowledgment."""
    first_started = threading.Event()
    release_first = threading.Event()
    second_snapshot_started = threading.Event()
    allow_second_snapshot = threading.Event()
    real_snapshot = checkpoint_module.snapshot_to_cpu

    def fail_first(temporary: Path) -> None:
        first_started.set()
        assert release_first.wait(timeout=5.0)
        temporary.write_bytes(b"partial")
        raise OSError("preserved publication failure")

    def snapshot(payload: Any) -> Any:
        if isinstance(payload, Mapping) and payload.get("sequence") == 2:
            second_snapshot_started.set()
            assert allow_second_snapshot.wait(timeout=5.0)
        return real_snapshot(payload)

    monkeypatch.setattr(checkpoint_module, "snapshot_to_cpu", snapshot)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    publisher = AsyncCheckpointPublisher(max_pending=2)
    first_future = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "first.pt", fail_first),),
        )
    )
    assert first_started.wait(timeout=5.0)
    first_completed = threading.Event()
    first_future.add_done_callback(lambda _: first_completed.set())
    real_acknowledge = publisher._acknowledge
    acknowledgment_calls = 0

    def interrupt_after_acknowledgment(submission: Any) -> None:
        nonlocal acknowledgment_calls
        acknowledgment_calls += 1
        real_acknowledge(submission)
        if acknowledgment_calls == 1:
            raise KeyboardInterrupt("adapter acknowledgment interrupted")

    monkeypatch.setattr(publisher, "_acknowledge", interrupt_after_acknowledgment)
    errors: list[BaseException] = []

    def submit_second() -> None:
        try:
            publisher.publish(
                checkpoint_root / "second.pt",
                {"sequence": 2},
            )
        except BaseException as error:
            errors.append(error)

    submitter = threading.Thread(target=submit_second)
    submitter.start()
    assert second_snapshot_started.wait(timeout=5.0)
    release_first.set()
    assert first_completed.wait(timeout=5.0)
    allow_second_snapshot.set()
    submitter.join(timeout=5.0)

    assert not submitter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    with pytest.raises(OSError, match="preserved publication failure"):
        publisher.flush()
    publisher.close()


def test_checkpoint_future_callbacks_observe_final_pipeline_state(
    tmp_path: Path,
) -> None:
    """Re-entrant and process-exception callbacks cannot corrupt publication."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    started = threading.Event()
    release = threading.Event()
    callback_completed = threading.Event()

    def write(temporary: Path) -> None:
        started.set()
        assert release.wait(timeout=5.0)
        temporary.write_bytes(b"complete")

    def write_second(temporary: Path) -> None:
        temporary.write_bytes(b"second")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    future = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "complete.pt", write),),
        )
    )
    assert started.wait(timeout=5.0)
    second_future = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "second.pt", write_second),),
        )
    )

    def flush_from_callback(_: Future[CheckpointPublication]) -> None:
        publisher.flush()
        callback_completed.set()
        raise KeyboardInterrupt("callback interrupted")

    future.add_done_callback(flush_from_callback)
    release.set()

    publisher.close()
    publication = future.result(timeout=5.0)
    assert callback_completed.wait(timeout=5.0)
    assert publication.published[0].path == checkpoint_root / "complete.pt"
    assert second_future.result(timeout=5.0).published[0].path == (
        checkpoint_root / "second.pt"
    )


def test_checkpoint_close_retains_handoff_interrupt_across_cleanup_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close retry surfaces acceptance interruption after cleanup can finish."""
    publisher = AsyncCheckpointPublisher()
    handoff_interrupt = KeyboardInterrupt("accepted handoff interrupted")

    def interrupt_handoff(submission: Any) -> Any:
        del submission
        raise handoff_interrupt

    monkeypatch.setattr(
        publisher._pipeline,
        "_complete_submission_handoff",
        interrupt_handoff,
    )
    future = publisher.publish(tmp_path / "checkpoint.pt", {"value": 1})
    assert future.result(timeout=5.0) == tmp_path / "checkpoint.pt"
    real_flush_pipeline = publisher._flush_pipeline
    cleanup_calls = 0

    def interrupt_cleanup(*, close: bool, raise_error: bool = True) -> BaseException | None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise SystemExit("cleanup interrupted")
        return real_flush_pipeline(close=close, raise_error=raise_error)

    monkeypatch.setattr(publisher, "_flush_pipeline", interrupt_cleanup)

    with pytest.raises(SystemExit, match="cleanup interrupted"):
        publisher.close()
    with pytest.raises(KeyboardInterrupt, match="accepted handoff interrupted") as raised:
        publisher.close()

    assert raised.value is handoff_interrupt
    publisher.close()


def test_checkpoint_future_callback_cannot_replace_worker_failure(
    tmp_path: Path,
) -> None:
    """A callback process exception cannot corrupt serializer attribution."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    release = threading.Event()

    def fail(temporary: Path) -> None:
        assert release.wait(timeout=5.0)
        temporary.write_bytes(b"partial")
        raise OSError("serializer failed")

    publisher = AsyncCheckpointPublisher()
    future = publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "failed.pt", fail),),
        )
    )

    def interrupt_callback(_: Future[CheckpointPublication]) -> None:
        raise KeyboardInterrupt("callback interrupted")

    future.add_done_callback(interrupt_callback)
    release.set()

    with pytest.raises(OSError, match="serializer failed"):
        future.result(timeout=5.0)
    with pytest.raises(OSError, match="serializer failed"):
        publisher.flush()
    publisher.close()


def test_checkpoint_future_callback_can_submit_and_wait_for_more_work(
    tmp_path: Path,
) -> None:
    """User callbacks do not block internal completion of nested publications."""
    release = threading.Event()
    callback_finished = threading.Event()
    nested_results: list[Path] = []

    def write_first(temporary: Path) -> None:
        assert release.wait(timeout=5.0)
        temporary.write_bytes(b"first")

    publisher = AsyncCheckpointPublisher()
    first = publisher.submit(
        CheckpointPlan(
            tmp_path,
            (CheckpointArtifact(tmp_path / "first.pt", write_first),),
        )
    )

    def publish_nested(_: Future[CheckpointPublication]) -> None:
        nested = publisher.publish(tmp_path / "nested.pt", {"value": 2})
        nested_results.append(nested.result(timeout=5.0))
        callback_finished.set()

    first.add_done_callback(publish_nested)
    release.set()

    assert first.result(timeout=5.0).published[0].path == tmp_path / "first.pt"
    assert callback_finished.wait(timeout=5.0)
    assert nested_results == [tmp_path / "nested.pt"]
    publisher.close()


def test_checkpoint_completed_future_callback_is_immediate_during_other_work(
    tmp_path: Path,
) -> None:
    """Post-completion callback registration remains synchronous."""
    publisher = AsyncCheckpointPublisher(max_pending=2)
    completed = publisher.publish(tmp_path / "complete.pt", {"value": 1})
    assert completed.result(timeout=5.0) == tmp_path / "complete.pt"
    release = threading.Event()

    def block(temporary: Path) -> None:
        assert release.wait(timeout=5.0)
        temporary.write_bytes(b"blocked")

    publisher.submit(
        CheckpointPlan(
            tmp_path,
            (CheckpointArtifact(tmp_path / "blocked.pt", block),),
        )
    )
    callbacks: list[Future[Path]] = []
    completed.add_done_callback(callbacks.append)

    assert callbacks == [completed]
    release.set()
    publisher.close()


def test_concurrent_checkpoint_flush_and_close_wait_for_shutdown(
    tmp_path: Path,
) -> None:
    """Concurrent terminal calls serialize behind the active close."""
    started = threading.Event()
    release = threading.Event()
    first_close_done = threading.Event()
    flush_done = threading.Event()
    second_close_done = threading.Event()

    def block(temporary: Path) -> None:
        started.set()
        assert release.wait(timeout=5.0)
        temporary.write_bytes(b"complete")

    publisher = AsyncCheckpointPublisher()
    publisher.submit(
        CheckpointPlan(
            tmp_path,
            (CheckpointArtifact(tmp_path / "checkpoint.pt", block),),
        )
    )
    assert started.wait(timeout=5.0)
    first_closer = threading.Thread(
        target=lambda: (publisher.close(), first_close_done.set())
    )
    first_closer.start()
    with publisher._lifecycle_condition:
        assert publisher._lifecycle_condition.wait_for(
            lambda: publisher._closing,
            timeout=5.0,
        )
    flusher = threading.Thread(target=lambda: (publisher.flush(), flush_done.set()))
    second_closer = threading.Thread(
        target=lambda: (publisher.close(), second_close_done.set())
    )
    flusher.start()
    second_closer.start()

    assert not flush_done.wait(timeout=0.1)
    assert not second_close_done.wait(timeout=0.1)
    release.set()
    for thread in (first_closer, flusher, second_closer):
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    assert first_close_done.is_set()
    assert flush_done.is_set()
    assert second_close_done.is_set()


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
    assert publisher._pipeline.pending_count == 1

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
    assert publication.published == (
        PublishedCheckpoint(
            path=resolved_root / "latest.pt",
            role="epoch",
            epoch=-1,
            size_bytes=8,
            sha256=hashlib.sha256(b"relative").hexdigest(),
        ),
    )
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


def test_async_checkpoint_flush_preserves_process_exception_priority(
    tmp_path: Path,
) -> None:
    """A later worker signal remains primary without losing an earlier failure."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    def fail(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("ordinary failure")

    def interrupt(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise KeyboardInterrupt("worker interrupted")

    publisher = AsyncCheckpointPublisher(max_pending=2)
    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "failed.pt", fail),),
        )
    )
    publisher.submit(
        CheckpointPlan(
            checkpoint_root,
            (CheckpointArtifact(checkpoint_root / "interrupted.pt", interrupt),),
        )
    )

    with pytest.raises(KeyboardInterrupt, match="worker interrupted") as raised:
        publisher.flush()

    assert "ordinary failure" in "\n".join(raised.value.__notes__)
    publisher.close()


def test_async_checkpoint_flush_retains_failures_if_aggregation_is_interrupted(
    tmp_path: Path,
) -> None:
    """An interrupted diagnostic note cannot consume acknowledged failures."""

    class InterruptingFailure(OSError):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self._interrupt_note_once = True

        def add_note(self, note: str) -> None:
            if self._interrupt_note_once:
                self._interrupt_note_once = False
                raise KeyboardInterrupt("failure aggregation interrupted")
            super().add_note(note)

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    first_error = InterruptingFailure("first publication failed")

    def fail_first(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise first_error

    def fail_second(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("second publication failed")

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
            (CheckpointArtifact(checkpoint_root / "second.pt", fail_second),),
        )
    )

    with pytest.raises(KeyboardInterrupt, match="failure aggregation interrupted"):
        publisher.flush()
    with pytest.raises(InterruptingFailure, match="first publication failed") as raised:
        publisher.flush()

    assert "second publication failed" in "\n".join(raised.value.__notes__)
    publisher.close()


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


def test_optimizer_step_metrics_observe_post_scheduler_state() -> None:
    loader = DataLoader(MappingDataset(), batch_size=4)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    sink = RecordingSink()

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        scheduler=scheduler,
        optimizer_step_metrics=lambda state: {
            "post_step_lr": optimizer.param_groups[0]["lr"],
            "reported_optimizer_step": state.optimizer_step,
        },
        metric_specs={
            "post_step_lr": MetricSpec(reduction="last", distributed=False),
            "reported_optimizer_step": MetricSpec(reduction="last", distributed=False),
        },
        train_metric_routes={
            "post_step_lr": MetricRoute(batch_name=None, epoch_name="Learning_Rate"),
            "reported_optimizer_step": MetricRoute(
                batch_name=None,
                epoch_name="optimizer_step",
            ),
        },
        observer=RunObserver((sink,)),
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            scheduler_interval="optimizer",
            checkpoint_every_epochs=None,
        ),
    ) as trainer:
        result = trainer.fit()

    completed = [
        observation
        for observation in sink.observations
        if observation.event == "task_completed"
        and observation.fields.get("phase") == "train"
    ]
    assert completed[-1].metrics == {
        "Learning_Rate": pytest.approx(0.025),
        "optimizer_step": 2.0,
    }
    assert result.training_history[0]["post_step_lr"] == pytest.approx(0.025)


def test_trainer_flushes_each_logged_optimizer_window_to_jsonl(tmp_path: Path) -> None:
    context = create_execution_context(
        tmp_path / "run",
        run_name="trainer-jsonl",
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("python", "train.py"),
        execution_id="attempt",
    )
    writer = ExecutionEventWriter.for_process(
        context,
        rank=0,
        monotonic_clock=lambda: 1.0,
    )
    loader = DataLoader(MappingDataset(), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with RunObserver((JsonlEventSink(writer),)) as observer, Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        observer=observer,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            checkpoint_every_epochs=None,
        ),
    ) as trainer:
        trainer.fit()

    progress = [
        event
        for event in read_execution_events(context.execution_dir / "rank-0.jsonl")
        if event.event == "progress" and event.phase == "train"
    ]
    assert [event.completed for event in progress] == [1, 2, 3, 4]
    assert all("loss" in event.display_metrics for event in progress)


def test_zero_based_optimizer_logical_clock_preserves_resume_history() -> None:
    loader = DataLoader(MappingDataset(), batch_size=8)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    sink = RecordingSink()

    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        train_step=regression_step,
        observer=RunObserver((sink,)),
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            checkpoint_every_epochs=None,
            optimizer_step_logical_clock="zero_based",
        ),
    ) as trainer:
        trainer.state.optimizer_step = 4
        trainer.fit()

    progress = [
        observation
        for observation in sink.observations
        if observation.event == "progress"
    ]
    assert progress[0].logical_step == 4
    assert progress[0].fields["coordinates"]["optimizer_step"] == 5


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


def test_single_runtime_owns_weighted_helpers_and_execution_lifecycle(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runtime-session"
    runtime = initialize_torch_runtime(
        TorchRuntimeConfig(device="cpu", workload_weights=(2,))
    )
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="runtime-session",
            invocation_kind="test",
            intended_phases=("validate",),
            command=("python", "validate.py"),
            execution_id="runtime-session-attempt",
        )
    )
    session = runtime.create_execution_session()

    assert runtime.workload_weights == (2.0,)
    assert runtime.local_partition_count(7, require_nonempty=True) == 7
    assert runtime.local_partition_indices(7) == range(0, 7)
    assert runtime.broadcast_bool(True)
    assert runtime.shared_string_union(("dice", "loss", "dice")) == ("dice", "loss")

    session.start_phase("validate")
    session.complete_phase(message="validation complete")
    session.close()

    events = read_execution_events(
        run_dir / "logs" / "executions" / "runtime-session-attempt" / "rank-0.jsonl"
    )
    assert [event.event for event in events] == [
        "process_started",
        "phase_started",
        "phase_completed",
        "process_completed",
    ]
    assert events[-1].exit_code == 0


class _RecordingSessionSink:
    """Record observer closure for execution-session ownership tests."""

    def __init__(self, order: list[str], label: str = "observer") -> None:
        self.order = order
        self.label = label
        self.closed = False

    def observe(self, observation: Observation) -> None:
        del observation

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self.order.append(self.label)


def _create_test_execution_session(
    tmp_path: Path,
    name: str,
) -> tuple[Any, Any]:
    """Create one started single-process runtime/session test fixture."""
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=tmp_path / name,
            run_name=name,
            invocation_kind="test",
            intended_phases=("train",),
            command=("python", "train.py"),
            execution_id=f"{name}-attempt",
        )
    )
    return runtime, runtime.create_execution_session()


def test_execution_session_closes_owned_resources_before_runtime_in_reverse_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    release_pipeline = threading.Event()
    runtime, session = _create_test_execution_session(tmp_path, "owned-resource-order")
    sink = _RecordingSessionSink(order)
    original_logging_close = runtime.execution_logging.close
    original_release_lease = runtime._release_logical_run_lease

    class RecordingTrainer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def close(self) -> None:
            order.append("trainer-checkpoint-flush")
            release_pipeline.set()

    def close_logging() -> None:
        order.append("execution-logging")
        original_logging_close()

    def release_lease() -> None:
        order.append("lease")
        original_release_lease()

    monkeypatch.setattr(torch_runtime_module, "Trainer", RecordingTrainer)
    monkeypatch.setattr(runtime.execution_logging, "close", close_logging)
    monkeypatch.setattr(runtime, "_release_logical_run_lease", release_lease)
    monkeypatch.setattr(runtime, "close_process_group", lambda: order.append("process-group"))

    observer = session.create_observer((sink,))

    def run_pipeline(value: str) -> str:
        assert release_pipeline.wait(timeout=5.0)
        order.append(value)
        return value

    pipeline = session.create_background_pipeline(
        run_pipeline,
        thread_name_prefix="test-session-pipeline",
    )
    pipeline.submit("background-pipeline")
    session.create_trainer(observer=observer)
    session.start_phase("train")
    session.complete_phase()
    session.close()

    assert order == [
        "trainer-checkpoint-flush",
        "background-pipeline",
        "observer",
        "execution-logging",
        "lease",
        "process-group",
    ]


def test_execution_session_closes_trainers_before_later_created_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    _, session = _create_test_execution_session(tmp_path, "resource-class-order")

    class RecordingTrainer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def close(self) -> None:
            order.append("trainer")

    monkeypatch.setattr(torch_runtime_module, "Trainer", RecordingTrainer)
    session.create_trainer()
    session.create_observer((_RecordingSessionSink(order),))
    session.close()

    assert order == ["trainer", "observer"]


def test_execution_session_close_is_idempotent_without_duplicate_terminal_events(
    tmp_path: Path,
) -> None:
    runtime, session = _create_test_execution_session(tmp_path, "idempotent-session")
    session.start_phase("train")
    session.complete_phase()

    session.close()
    session.close()

    events = read_execution_events(
        tmp_path
        / "idempotent-session"
        / "logs"
        / "executions"
        / "idempotent-session-attempt"
        / "rank-0.jsonl"
    )
    assert [event.event for event in events].count("phase_completed") == 1
    assert [event.event for event in events].count("process_completed") == 1
    assert runtime._closed is True


def test_execution_session_recovers_interrupted_pipeline_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session cleanup finishes accepted work before propagating interruption."""
    _, session = _create_test_execution_session(tmp_path, "interrupted-pipeline-cleanup")
    pipeline = session.create_background_pipeline(lambda value: value + 1)
    submission = pipeline.submit(4)
    original_flush = pipeline.flush
    calls = 0

    def interrupt_once() -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("pipeline cleanup interrupted")
        return original_flush()

    monkeypatch.setattr(pipeline, "flush", interrupt_once)

    with pytest.raises(KeyboardInterrupt, match="pipeline cleanup interrupted"):
        session.close()

    assert not pipeline.owns(submission)
    assert pipeline.close() == ()
    session.close()


def test_execution_session_closes_pipeline_after_interrupted_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructed pipeline is closed if session ownership cannot register."""
    _, session = _create_test_execution_session(tmp_path, "pipeline-registration")
    created: list[BoundedBackgroundPipeline[Any, Any]] = []
    real_pipeline_type = torch_runtime_module.BoundedBackgroundPipeline

    def capture_pipeline(*args: Any, **kwargs: Any) -> BoundedBackgroundPipeline[Any, Any]:
        pipeline = real_pipeline_type(*args, **kwargs)
        created.append(pipeline)
        return pipeline

    def interrupt_registration(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise KeyboardInterrupt("registration interrupted")

    monkeypatch.setattr(torch_runtime_module, "BoundedBackgroundPipeline", capture_pipeline)
    monkeypatch.setattr(session, "_register_owned_resource", interrupt_registration)

    with pytest.raises(KeyboardInterrupt, match="registration interrupted"):
        session.create_background_pipeline(lambda value: value)

    assert len(created) == 1
    assert not created[0]._worker_thread.is_alive()
    assert created[0].close() == ()
    session.close()


def test_execution_session_pipeline_factory_cannot_cross_concurrent_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session close either owns a constructed pipeline or makes its factory fail."""
    _, session = _create_test_execution_session(tmp_path, "pipeline-close-race")
    constructed = threading.Event()
    allow_registration = threading.Event()
    created: list[BoundedBackgroundPipeline[Any, Any]] = []
    errors: list[BaseException] = []
    real_pipeline_type = torch_runtime_module.BoundedBackgroundPipeline

    def pause_after_construction(
        *args: Any,
        **kwargs: Any,
    ) -> BoundedBackgroundPipeline[Any, Any]:
        pipeline = real_pipeline_type(*args, **kwargs)
        created.append(pipeline)
        constructed.set()
        assert allow_registration.wait(timeout=5.0)
        return pipeline

    def create_pipeline() -> None:
        try:
            session.create_background_pipeline(lambda value: value)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(
        torch_runtime_module,
        "BoundedBackgroundPipeline",
        pause_after_construction,
    )
    factory = threading.Thread(target=create_pipeline)
    factory.start()
    assert constructed.wait(timeout=5.0)
    session.close()
    allow_registration.set()
    factory.join(timeout=5.0)

    assert not factory.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert len(created) == 1
    assert not created[0]._worker_thread.is_alive()
    assert created[0].close() == ()


def test_execution_session_preserves_every_pipeline_cleanup_failure(
    tmp_path: Path,
) -> None:
    """An active workload error retains every attributed pipeline failure."""
    _, session = _create_test_execution_session(tmp_path, "pipeline-cleanup-failures")
    workload_error = RuntimeError("workload failed")

    def fail(value: int) -> int:
        raise OSError(f"publication {value} failed")

    with pytest.raises(RuntimeError, match="workload failed") as raised, session:
        pipeline = session.create_background_pipeline(fail, max_pending=2)
        pipeline.submit(1)
        pipeline.submit(2)
        raise workload_error

    assert raised.value is workload_error
    notes = "\n".join(workload_error.__notes__)
    assert "publication 1 failed" in notes
    assert "publication 2 failed" in notes


def test_execution_session_retries_interrupted_pipeline_acknowledgment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session fallback acknowledges outcomes before tearing down runtime state."""
    _, session = _create_test_execution_session(tmp_path, "pipeline-acknowledgment")
    workload_error = RuntimeError("workload failed")

    def fail(_: int) -> int:
        raise OSError("publication failed")

    pipeline = session.create_background_pipeline(fail)
    submission = pipeline.submit(1)
    original_acknowledge = pipeline.acknowledge
    calls = 0

    def interrupt_once(value: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_acknowledge(value)
            raise KeyboardInterrupt("acknowledgment interrupted")
        original_acknowledge(value)

    monkeypatch.setattr(pipeline, "acknowledge", interrupt_once)

    with pytest.raises(RuntimeError, match="workload failed") as raised, session:
        raise workload_error

    assert raised.value is workload_error
    notes = "\n".join(workload_error.__notes__)
    assert "publication failed" in notes
    assert "acknowledgment interrupted" in notes
    assert not pipeline.owns(submission)


def test_execution_session_closes_observer_after_trainer_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    _, session = _create_test_execution_session(tmp_path, "construction-failure")
    sink = _RecordingSessionSink(order)

    def fail_trainer(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("trainer construction failed")

    monkeypatch.setattr(torch_runtime_module, "Trainer", fail_trainer)

    with pytest.raises(RuntimeError, match="trainer construction failed"), session:
        observer = session.create_observer((sink,))
        session.create_trainer(observer=observer)

    assert sink.closed is True


def test_execution_session_borrows_directly_supplied_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session = _create_test_execution_session(tmp_path, "borrowed-observer")
    borrowed = _RecordingSessionSink([])

    class RecordingTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.observer = kwargs["observer"]

        def close(self) -> None:
            pass

    monkeypatch.setattr(torch_runtime_module, "Trainer", RecordingTrainer)
    session.create_trainer(observer=borrowed)
    session.close()

    assert borrowed.closed is False


def test_execution_session_preserves_workload_error_when_owned_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session = _create_test_execution_session(tmp_path, "primary-error")
    workload_error = RuntimeError("workload failed")

    class FailingTrainer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def close(self) -> None:
            raise OSError("checkpoint flush failed")

    monkeypatch.setattr(torch_runtime_module, "Trainer", FailingTrainer)

    with pytest.raises(RuntimeError, match="workload failed") as raised, session:
        session.create_trainer()
        with session.phase_scope("train"):
            raise workload_error

    assert raised.value is workload_error
    assert any("checkpoint flush failed" in note for note in workload_error.__notes__)


def test_execution_session_destroys_only_runtime_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[str] = []
    monkeypatch.setattr(torch_runtime_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch_runtime_module.dist,
        "destroy_process_group",
        lambda: destroyed.append("destroyed"),
    )

    borrowed_runtime, borrowed_session = _create_test_execution_session(
        tmp_path, "borrowed-process-group"
    )
    borrowed_session.close()
    assert borrowed_runtime._owns_process_group is False
    assert destroyed == []

    owned_runtime, owned_session = _create_test_execution_session(
        tmp_path, "owned-process-group"
    )
    owned_runtime._owns_process_group = True
    owned_session.close()
    assert destroyed == ["destroyed"]
    assert owned_runtime._owns_process_group is False


@pytest.mark.parametrize(
    ("error", "expected_exit_code", "expected_signal", "expected_status"),
    [
        (RuntimeError("validation failed"), 1, None, "failed"),
        (KeyboardInterrupt("validation interrupted"), 130, 2, "interrupted"),
        (SystemExit(7), 7, None, "failed"),
    ],
)
def test_execution_session_scope_derives_terminal_process_state(
    tmp_path: Path,
    error: BaseException,
    expected_exit_code: int,
    expected_signal: int | None,
    expected_status: str,
) -> None:
    run_dir = tmp_path / f"session-{type(error).__name__}"
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="runtime-session",
            invocation_kind="test",
            intended_phases=("validate",),
            command=("python", "validate.py"),
            execution_id=f"session-{type(error).__name__.lower()}",
        )
    )
    session = runtime.create_execution_session()

    with pytest.raises(type(error)) as raised, session.phase_scope("validate"):
        raise error
    session.close(error=raised.value)

    events = read_execution_events(
        run_dir
        / "logs"
        / "executions"
        / f"session-{type(error).__name__.lower()}"
        / "rank-0.jsonl"
    )
    assert [event.event for event in events] == [
        "process_started",
        "phase_started",
        "phase_failed",
        "process_completed",
    ]
    assert events[-2].extensions["status"] == expected_status
    assert events[-1].exit_code == expected_exit_code
    assert events[-1].signal == expected_signal


def test_execution_session_cannot_report_failed_phase_as_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-session"
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="failed-session",
            invocation_kind="test",
            intended_phases=("validate",),
            command=("python", "validate.py"),
            execution_id="failed-session-attempt",
        )
    )
    session = runtime.create_execution_session()
    session.start_phase("validate")
    session.fail_phase(RuntimeError("failed"))
    session.close(exit_code=0)

    events = read_execution_events(
        run_dir / "logs" / "executions" / "failed-session-attempt" / "rank-0.jsonl"
    )
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == 1


def test_execution_session_reraises_late_runtime_cleanup_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cleanup-failed-session"
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="cleanup-failed-session",
            invocation_kind="test",
            intended_phases=("validate",),
            command=("python", "validate.py"),
            execution_id="cleanup-failed-session-attempt",
        )
    )
    session = runtime.create_execution_session()
    session.start_phase("validate")
    session.complete_phase()
    runtime.close_process_group = lambda: (_ for _ in ()).throw(
        RuntimeError("process group cleanup failed")
    )

    with pytest.raises(RuntimeError, match="process group cleanup failed"):
        session.close()

    events = read_execution_events(
        run_dir
        / "logs"
        / "executions"
        / "cleanup-failed-session-attempt"
        / "rank-0.jsonl"
    )
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == 0
    assert events[-1].message is None


def test_execution_session_derives_interrupt_from_presentation_cleanup(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cleanup-interrupted-session"
    runtime = initialize_torch_runtime(TorchRuntimeConfig(device="cpu"))
    runtime.start_execution(
        TorchExecutionRequest(
            run_dir=run_dir,
            run_name="cleanup-interrupted-session",
            invocation_kind="test",
            intended_phases=("validate",),
            command=("python", "validate.py"),
            execution_id="cleanup-interrupted-session-attempt",
        )
    )
    session = runtime.create_execution_session()
    session.start_phase("validate")

    def interrupt_cleanup() -> None:
        raise KeyboardInterrupt("cleanup interrupted")

    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        session.close(before_close=interrupt_cleanup)

    events = read_execution_events(
        run_dir
        / "logs"
        / "executions"
        / "cleanup-interrupted-session-attempt"
        / "rank-0.jsonl"
    )
    assert events[-2].event == "phase_failed"
    assert events[-2].extensions["status"] == "interrupted"
    assert events[-1].event == "process_completed"
    assert events[-1].exit_code == 130
    assert events[-1].signal == 2


def test_runtime_validates_launch_and_weight_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="multi-process launch"):
        initialize_torch_runtime(
            TorchRuntimeConfig(device="cpu", strict_launch_environment=True)
        )

    with pytest.raises(RuntimeError, match="one value per rank"):
        initialize_torch_runtime(
            TorchRuntimeConfig(
                strategy="ddp",
                device="cpu",
                rank=0,
                local_rank=0,
                world_size=2,
                workload_weights=(1,),
            )
        )

    with pytest.raises(RuntimeError, match="global rank and local rank"):
        initialize_torch_runtime(
            TorchRuntimeConfig(
                strategy="ddp",
                device="cpu",
                rank=1,
                local_rank=0,
                world_size=2,
                require_global_local_rank_match=True,
            )
        )


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
    assert {result[5] for result in results} == {4}
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
