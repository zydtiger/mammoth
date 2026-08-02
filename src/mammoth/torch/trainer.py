"""Bounded generic trainer for constructed PyTorch modules and data loaders.

Consuming projects provide model, optimizer, loaders, and step functions. This
module owns only ordinary single/DDP loop mechanics, logging, and registered
checkpoint publication.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mammoth.logging import RunObserver
from mammoth.torch.batch import move_batch_to_device
from mammoth.torch.callbacks import Callback
from mammoth.torch.checkpoint import (
    AsyncCheckpointPublisher,
    Stateful,
    StateRegistry,
    checkpoint_payload,
    restore_checkpoint,
)
from mammoth.torch.metrics import MetricAccumulator, MetricSpec
from mammoth.torch.state import TrainerState

Precision = Literal["fp32", "bf16", "fp16"]
Strategy = Literal["single", "ddp"]
SchedulerInterval = Literal["optimizer", "epoch", "validation"]


@dataclass(frozen=True, slots=True)
class StepContext:
    """Ordinary loop coordinates supplied to a project step function."""

    training: bool
    epoch: int
    batch_index: int
    global_step: int
    optimizer_step: int


@dataclass(frozen=True, slots=True)
class StepOutput:
    """Scalar loss and already-computed project metrics returned by one step."""

    loss: torch.Tensor | None = None
    metrics: Mapping[str, float | torch.Tensor] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.loss is not None and (
            not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0
        ):
            raise ValueError("step loss must be a scalar tensor or None")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("step metric weight must be positive and finite")


StepFunction = Callable[[torch.nn.Module, Any, StepContext], StepOutput]
BatchMover = Callable[[Any, torch.device], Any]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Model-independent ordinary training-loop policy."""

    epochs: int
    device: str = "auto"
    strategy: Strategy = "single"
    precision: Precision = "fp32"
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float | None = None
    validation_every_epochs: int = 1
    log_every_batches: int = 1
    scheduler_interval: SchedulerInterval = "epoch"
    scheduler_monitor: str | None = None
    checkpoint_every_epochs: int | None = 1
    checkpoint_filename: str = "checkpoint-{epoch:04d}.pt"
    max_pending_checkpoints: int = 1
    non_blocking_transfer: bool = False
    train_phase: str = "train"
    validation_phase: str = "validation"
    display_metric_names: tuple[str, ...] = ("loss",)

    def __post_init__(self) -> None:
        positive_integer("epochs", self.epochs)
        positive_integer("gradient_accumulation_steps", self.gradient_accumulation_steps)
        positive_integer("validation_every_epochs", self.validation_every_epochs)
        positive_integer("log_every_batches", self.log_every_batches)
        positive_integer("max_pending_checkpoints", self.max_pending_checkpoints)
        if self.checkpoint_every_epochs is not None:
            positive_integer("checkpoint_every_epochs", self.checkpoint_every_epochs)
        if self.strategy not in {"single", "ddp"}:
            raise ValueError(f"Unsupported trainer strategy: {self.strategy!r}")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported trainer precision: {self.precision!r}")
        if self.scheduler_interval not in {"optimizer", "epoch", "validation"}:
            raise ValueError(f"Unsupported scheduler interval: {self.scheduler_interval!r}")
        if self.max_gradient_norm is not None and (
            isinstance(self.max_gradient_norm, bool)
            or not math.isfinite(self.max_gradient_norm)
            or self.max_gradient_norm <= 0
        ):
            raise ValueError("max_gradient_norm must be positive and finite")
        for name, value in (
            ("train_phase", self.train_phase),
            ("validation_phase", self.validation_phase),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        checkpoint_path = Path(self.checkpoint_filename)
        if checkpoint_path.name != self.checkpoint_filename:
            raise ValueError("checkpoint_filename must be a single filename template")


@dataclass(frozen=True, slots=True)
class TrainerResult:
    """Terminal trainer coordinates and epoch summaries."""

    state: TrainerState
    training_history: tuple[Mapping[str, float], ...]
    validation_history: tuple[Mapping[str, float], ...]


class Trainer:
    """Train one constructed module with caller-provided batch semantics."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader[Any],
        train_step: StepFunction,
        config: TrainerConfig,
        validation_loader: DataLoader[Any] | None = None,
        validation_step: StepFunction | None = None,
        scheduler: Any | None = None,
        observer: RunObserver | None = None,
        callbacks: Sequence[Callback] = (),
        metric_specs: Mapping[str, MetricSpec] | None = None,
        checkpoint_dir: Path | None = None,
        extra_state: Mapping[str, Stateful] | None = None,
        batch_mover: BatchMover | None = None,
    ) -> None:
        if (validation_loader is None) != (validation_step is None):
            raise ValueError("validation_loader and validation_step must be provided together")
        self.config = config
        self.device = resolve_device(config.device)
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 precision requires a CUDA device")
        self.rank, self.world_size = distributed_identity(config.strategy)
        self.base_model = model.to(self.device)
        self.model = wrap_model(self.base_model, self.device, config.strategy)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_step = train_step
        self.validation_loader = validation_loader
        self.validation_step = validation_step
        self.scheduler = scheduler
        self.observer = observer or RunObserver()
        self.callbacks = tuple(callbacks)
        self.metric_specs = dict(metric_specs or {})
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.batch_mover = batch_mover or self._default_batch_mover
        self.state = TrainerState()
        self.scaler = torch.GradScaler("cuda", enabled=config.precision == "fp16")
        self.registry = StateRegistry()
        self.registry.register("model", self.base_model)
        self.registry.register("optimizer", self.optimizer)
        self.registry.register("trainer", self.state)
        self.registry.register("scaler", self.scaler)
        if scheduler is not None:
            self.registry.register("scheduler", scheduler)
        for index, callback in enumerate(self.callbacks):
            self.registry.register(f"callback-{index}", callback)
        for name, value in (extra_state or {}).items():
            self.registry.register(f"project-{name}", value)
        self.publisher = AsyncCheckpointPublisher(
            max_pending=config.max_pending_checkpoints
        )
        self._closed = False

    def fit(self) -> TrainerResult:
        """Execute configured epochs and return project metric summaries."""
        training_history: list[Mapping[str, float]] = []
        validation_history: list[Mapping[str, float]] = []
        for callback in self.callbacks:
            callback.on_train_start(self.state)
        self.observer.emit("phase_started", phase=self.config.train_phase)
        try:
            for epoch in range(self.state.epoch + 1, self.config.epochs):
                set_sampler_epoch(self.train_loader, epoch)
                training_metrics = self.train_epoch(epoch)
                training_history.append(training_metrics)
                self.state.epoch = epoch
                if self.scheduler is not None and self.config.scheduler_interval == "epoch":
                    self.scheduler.step()
                for callback in self.callbacks:
                    callback.on_epoch_end(self.state, training_metrics)

                validation_metrics: Mapping[str, float] | None = None
                if (
                    self.validation_loader is not None
                    and (epoch + 1) % self.config.validation_every_epochs == 0
                ):
                    validation_metrics = self.validate_epoch(epoch)
                    validation_history.append(validation_metrics)
                    if (
                        self.scheduler is not None
                        and self.config.scheduler_interval == "validation"
                    ):
                        self.step_validation_scheduler(validation_metrics)
                    for callback in self.callbacks:
                        callback.on_validation_end(self.state, validation_metrics)

                if self.should_checkpoint(epoch):
                    self.publish_checkpoint(epoch)
                if any(callback.should_stop(self.state) for callback in self.callbacks):
                    self.state.stopped_early = True
                    break
            self.observer.emit("phase_completed", phase=self.config.train_phase)
        except BaseException as error:
            self.observer.emit("phase_failed", phase=self.config.train_phase, message=str(error))
            raise
        finally:
            self.publisher.flush()
        for callback in self.callbacks:
            callback.on_train_end(self.state)
        return TrainerResult(
            state=self.state,
            training_history=tuple(training_history),
            validation_history=tuple(validation_history),
        )

    def train_epoch(self, epoch: int) -> Mapping[str, float]:
        """Run one ordinary gradient epoch with accumulation and clipping."""
        self.model.train()
        accumulator = MetricAccumulator(self.metric_specs)
        total_batches = len(self.train_loader)
        task_id = f"epoch-{epoch}"
        self.observer.emit("task_started", phase=self.config.train_phase, task_id=task_id)
        self.optimizer.zero_grad(set_to_none=True)
        group_start = 0
        for batch_index, batch in enumerate(self.train_loader):
            group_size = min(
                self.config.gradient_accumulation_steps,
                total_batches - group_start,
            )
            should_step = batch_index - group_start + 1 == group_size
            sync_context = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            moved = self.batch_mover(batch, self.device)
            context = StepContext(
                training=True,
                epoch=epoch,
                batch_index=batch_index,
                global_step=self.state.global_step,
                optimizer_step=self.state.optimizer_step,
            )
            with sync_context, self.autocast_context():
                output = self.train_step(self.model, moved, context)
                if output.loss is None:
                    raise ValueError("train step must return a scalar loss")
                scaled_loss = output.loss / group_size
            backward_loss = self.scaler.scale(scaled_loss)
            assert isinstance(backward_loss, torch.Tensor)
            backward_loss.backward()  # type: ignore[no-untyped-call]
            if should_step:
                self.optimizer_step()
                group_start = batch_index + 1
            self.state.global_step += 1
            metrics = output_metrics(output)
            accumulator.update(metrics, weight=output.weight)
            if (
                (batch_index + 1) % self.config.log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                self.observer.progress(
                    phase=self.config.train_phase,
                    task_id=task_id,
                    completed=batch_index + 1,
                    total=total_batches,
                    metrics=metrics,
                    display_metrics=display_metrics(metrics, self.config.display_metric_names),
                    coordinates={
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_step": self.state.global_step,
                        "optimizer_step": self.state.optimizer_step,
                    },
                    final=batch_index + 1 == total_batches,
                    unit="batch",
                )
        summary = accumulator.compute(
            device=self.device,
            distributed=self.config.strategy == "ddp",
        )
        self.observer.emit("task_completed", phase=self.config.train_phase, task_id=task_id)
        return summary

    def validate_epoch(self, epoch: int) -> Mapping[str, float]:
        """Run one no-gradient validation epoch through the project step function."""
        assert self.validation_loader is not None and self.validation_step is not None
        self.model.eval()
        accumulator = MetricAccumulator(self.metric_specs)
        total_batches = len(self.validation_loader)
        task_id = f"epoch-{epoch}"
        self.observer.emit("phase_started", phase=self.config.validation_phase)
        self.observer.emit(
            "task_started", phase=self.config.validation_phase, task_id=task_id
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(self.validation_loader):
                moved = self.batch_mover(batch, self.device)
                context = StepContext(
                    training=False,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=self.state.global_step,
                    optimizer_step=self.state.optimizer_step,
                )
                with self.autocast_context():
                    output = self.validation_step(self.model, moved, context)
                metrics = output_metrics(output)
                accumulator.update(metrics, weight=output.weight)
                self.observer.progress(
                    phase=self.config.validation_phase,
                    task_id=task_id,
                    completed=batch_index + 1,
                    total=total_batches,
                    metrics=metrics,
                    display_metrics=display_metrics(metrics, self.config.display_metric_names),
                    coordinates={"epoch": epoch, "batch": batch_index},
                    final=batch_index + 1 == total_batches,
                    unit="batch",
                )
        summary = accumulator.compute(
            device=self.device,
            distributed=self.config.strategy == "ddp",
        )
        self.observer.emit(
            "task_completed", phase=self.config.validation_phase, task_id=task_id
        )
        self.observer.emit("phase_completed", phase=self.config.validation_phase)
        return summary

    def optimizer_step(self) -> None:
        """Unscale, clip, step, schedule, and clear one accumulated gradient."""
        if self.config.max_gradient_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.base_model.parameters(), self.config.max_gradient_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimizer_step += 1
        if self.scheduler is not None and self.config.scheduler_interval == "optimizer":
            self.scheduler.step()

    def autocast_context(self) -> Any:
        """Return configured device autocast without affecting fp32 execution."""
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def step_validation_scheduler(self, metrics: Mapping[str, float]) -> None:
        """Advance a validation scheduler with an optional project metric."""
        scheduler = self.scheduler
        if scheduler is None:
            raise RuntimeError("validation scheduler is not configured")
        if self.config.scheduler_monitor is None:
            scheduler.step()
            return
        if self.config.scheduler_monitor not in metrics:
            raise KeyError(
                f"Scheduler metric {self.config.scheduler_monitor!r} was not reported"
            )
        scheduler.step(metrics[self.config.scheduler_monitor])

    def should_checkpoint(self, epoch: int) -> bool:
        """Return whether the primary rank should publish this completed epoch."""
        every = self.config.checkpoint_every_epochs
        return (
            self.checkpoint_dir is not None
            and every is not None
            and (epoch + 1) % every == 0
            and self.rank == 0
        )

    def publish_checkpoint(self, epoch: int) -> None:
        """Submit one registered-state snapshot for bounded atomic publication."""
        assert self.checkpoint_dir is not None
        filename = self.config.checkpoint_filename.format(
            epoch=epoch,
            global_step=self.state.global_step,
            optimizer_step=self.state.optimizer_step,
        )
        if Path(filename).name != filename:
            raise ValueError("formatted checkpoint filename must remain a single filename")
        self.publisher.publish(
            self.checkpoint_dir / filename,
            checkpoint_payload(self.registry),
        )

    def load_checkpoint(self, path: Path, *, strict: bool = True) -> None:
        """Restore every registered state before calling :meth:`fit`."""
        restore_checkpoint(path, self.registry, strict=strict, map_location=self.device)

    def close(self) -> None:
        """Flush and close checkpoint publication resources."""
        if self._closed:
            return
        self._closed = True
        self.publisher.close()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _default_batch_mover(self, batch: Any, device: torch.device) -> Any:
        return move_batch_to_device(
            batch,
            device,
            non_blocking=self.config.non_blocking_transfer,
        )


def output_metrics(output: StepOutput) -> dict[str, float]:
    """Convert scalar tensors to detached finite floats and include loss."""
    metrics: dict[str, float] = {}
    for name, value in output.metrics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(f"metric {name!r} tensor must be scalar")
            scalar = float(value.detach().item())
        else:
            scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"metric {name!r} must be finite")
        metrics[name] = scalar
    if output.loss is not None:
        metrics.setdefault("loss", float(output.loss.detach().item()))
    return metrics


def display_metrics(
    metrics: Mapping[str, float], names: Sequence[str]
) -> dict[str, float]:
    """Select the explicitly configured bounded live metric subset."""
    selected = {name: metrics[name] for name in names if name in metrics}
    if len(selected) > 16:
        raise ValueError("display_metric_names may select at most 16 reported metrics")
    return selected


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` or one explicit torch device string."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def distributed_identity(strategy: Strategy) -> tuple[int, int]:
    """Return initialized process-group rank identity for standard DDP."""
    if strategy == "single":
        return 0, 1
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("DDP strategy requires an initialized torch.distributed process group")
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def wrap_model(
    model: torch.nn.Module,
    device: torch.device,
    strategy: Strategy,
) -> torch.nn.Module:
    """Wrap one constructed model in standard DDP when configured."""
    if strategy == "single":
        return model
    device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
    return DistributedDataParallel(model, device_ids=device_ids)


def set_sampler_epoch(loader: DataLoader[Any], epoch: int) -> None:
    """Advance a distributed sampler without requiring one concrete sampler type."""
    method = getattr(loader.sampler, "set_epoch", None)
    if callable(method):
        method(epoch)


def positive_integer(name: str, value: Any) -> None:
    """Validate one positive integer configuration field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
