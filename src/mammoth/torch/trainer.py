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
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

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
    TrainerCheckpointContext,
    TrainerCheckpointPolicy,
    checkpoint_payload,
    restore_checkpoint,
)
from mammoth.torch.metrics import (
    MetricAccumulator,
    MetricRoute,
    MetricSpec,
    StatefulMetric,
    compute_stateful_metrics,
    reset_stateful_metrics,
    route_metrics,
    scalar_metrics,
    snapshot_stateful_metrics,
    update_stateful_metrics,
)
from mammoth.torch.runtime import Strategy, TorchExecutionRuntime, resolve_device
from mammoth.torch.scheduling import AccumulationPolicy, UniformAccumulationPolicy
from mammoth.torch.state import TrainerState

Precision = Literal["fp32", "bf16", "fp16"]
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
    """Loss, scalar metrics, and transient stateful-metric updates from one step."""

    loss: torch.Tensor | None = None
    metrics: Mapping[str, float | torch.Tensor] = field(default_factory=dict)
    metric_updates: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.loss is not None and (
            not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0
        ):
            raise ValueError("step loss must be a scalar tensor or None")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("step metric weight must be positive and finite")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("step metrics must be a mapping")
        if not isinstance(self.metric_updates, Mapping):
            raise TypeError("step metric_updates must be a mapping")


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
    emit_fit_phase_events: bool = True

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
        if not isinstance(self.emit_fit_phase_events, bool):
            raise ValueError("emit_fit_phase_events must be a boolean")
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
        train_metric_routes: Mapping[str, MetricRoute] | None = None,
        validation_metric_routes: Mapping[str, MetricRoute] | None = None,
        train_stateful_metrics: Mapping[str, StatefulMetric] | None = None,
        validation_stateful_metrics: Mapping[str, StatefulMetric] | None = None,
        accumulation_policy: AccumulationPolicy | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_policy: TrainerCheckpointPolicy | None = None,
        extra_state: Mapping[str, Stateful] | None = None,
        batch_mover: BatchMover | None = None,
        runtime: TorchExecutionRuntime | None = None,
    ) -> None:
        if (validation_loader is None) != (validation_step is None):
            raise ValueError("validation_loader and validation_step must be provided together")
        if checkpoint_dir is not None and checkpoint_policy is not None:
            raise ValueError(
                "checkpoint_dir and checkpoint_policy select different checkpoint formats"
            )
        self.config = config
        self.runtime = runtime
        if runtime is not None and runtime.strategy != config.strategy:
            raise ValueError(
                f"Trainer strategy {config.strategy!r} does not match runtime "
                f"strategy {runtime.strategy!r}"
            )
        self.device = runtime.device if runtime is not None else resolve_device(config.device)
        if runtime is not None and config.device != "auto":
            configured_device = resolve_device(config.device)
            if configured_device != self.device:
                raise ValueError(
                    f"Trainer device {configured_device} does not match runtime device "
                    f"{self.device}"
                )
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 precision requires a CUDA device")
        if runtime is None:
            self.rank, self.world_size = distributed_identity(config.strategy)
        else:
            self.rank, self.world_size = runtime.rank, runtime.world_size
        self.base_model = model.to(self.device)
        self.model = wrap_model(self.base_model, self.device, config.strategy)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_step = train_step
        self.validation_loader = validation_loader
        self.validation_step = validation_step
        self.scheduler = scheduler
        runtime_observer = (
            runtime.execution_logging.observer
            if runtime is not None and runtime.execution_logging is not None
            else None
        )
        self.observer = observer or runtime_observer or RunObserver()
        self.callbacks = tuple(callbacks)
        self.metric_specs = dict(metric_specs or {})
        self.train_metric_routes = dict(train_metric_routes or {})
        self.validation_metric_routes = dict(validation_metric_routes or {})
        self.train_stateful_metrics = dict(train_stateful_metrics or {})
        self.validation_stateful_metrics = dict(validation_stateful_metrics or {})
        self.accumulation_policy = accumulation_policy or UniformAccumulationPolicy(
            config.gradient_accumulation_steps
        )
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_policy = checkpoint_policy
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
        if self.config.emit_fit_phase_events:
            self.observer.emit("phase_started", phase=self.config.train_phase)
        fit_error: BaseException | None = None
        try:
            self.coordinate(
                "train-start callbacks",
                self.run_train_start_callbacks,
            )
            for epoch in range(self.state.epoch + 1, self.config.epochs):
                self.coordinate(
                    "train sampler epoch",
                    partial(set_sampler_epoch, self.train_loader, epoch),
                )
                training_metrics = self.train_epoch(epoch)
                training_history.append(training_metrics)
                self.state.epoch = epoch
                if self.scheduler is not None and self.config.scheduler_interval == "epoch":
                    self.coordinate("epoch scheduler", self.scheduler.step)
                self.coordinate(
                    "epoch callbacks",
                    partial(self.run_epoch_callbacks, training_metrics),
                )

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
                        self.coordinate(
                            "validation scheduler",
                            partial(
                                self.step_validation_scheduler,
                                validation_metrics,
                            ),
                        )
                    self.coordinate(
                        "validation callbacks",
                        partial(
                            self.run_validation_callbacks,
                            validation_metrics,
                        ),
                    )

                local_should_stop = self.coordinate(
                    "stop callbacks",
                    lambda: any(
                        callback.should_stop(self.state) for callback in self.callbacks
                    ),
                )
                stop_decisions = self.all_gather_object(local_should_stop)
                if not isinstance(stop_decisions[0], bool):
                    raise RuntimeError("rank zero returned an invalid early-stop decision")
                should_stop = stop_decisions[0]
                self.state.stopped_early = should_stop
                self.publish_checkpoint_if_due(
                    epoch,
                    training_metrics,
                    validation_metrics,
                )
                if should_stop:
                    break
            self.flush_checkpoints()
            self.coordinate(
                "train-end callbacks",
                self.run_train_end_callbacks,
            )
            if self.config.emit_fit_phase_events:
                self.observer.emit("phase_completed", phase=self.config.train_phase)
        except BaseException as error:
            fit_error = error
            if self.config.emit_fit_phase_events:
                self.observer.emit(
                    "phase_failed",
                    phase=self.config.train_phase,
                    message=str(error),
                )
            raise
        finally:
            if fit_error is not None:
                try:
                    self.flush_checkpoints()
                except BaseException as flush_error:
                    fit_error.add_note(f"Checkpoint shutdown also failed: {flush_error}")
        return TrainerResult(
            state=self.state,
            training_history=tuple(training_history),
            validation_history=tuple(validation_history),
        )

    def train_epoch(self, epoch: int) -> Mapping[str, float]:
        """Run one ordinary gradient epoch with accumulation and clipping."""
        self.coordinate("training mode setup", self.model.train)
        accumulator = MetricAccumulator(self.metric_specs)
        total_batches = self.coordinate(
            "training loader length",
            lambda: len(self.train_loader),
        )
        plan = None
        window_sizes: tuple[int, ...] = ()
        planning_error: BaseException | None = None
        try:
            plan = self.accumulation_policy.plan(
                rank=self.rank,
                world_size=self.world_size,
                local_batch_count=total_batches,
            )
            window_sizes = plan.window_sizes(total_batches)
        except BaseException as error:
            planning_error = error
        self.raise_distributed_failure("accumulation planning", planning_error)
        assert plan is not None
        self.validate_optimizer_window_count(len(window_sizes))
        task_id = f"epoch-{epoch}"
        self.observer.emit(
            "task_started",
            phase=self.config.train_phase,
            task_id=task_id,
            epoch=epoch,
            epoch_total=self.config.epochs,
        )
        try:
            window_stateful_baseline = self.coordinate(
                "training metric setup",
                lambda: reset_and_snapshot_stateful_metrics(
                    self.train_stateful_metrics
                ),
            )
            with self.observer.periodic_heartbeats(
                phase=self.config.train_phase,
                task_id=task_id,
                message="Training epoch is still active.",
            ):
                self.coordinate(
                    "initial gradient reset",
                    partial(self.optimizer.zero_grad, set_to_none=True),
                )
                loader_iterator = self.coordinate(
                    "training loader setup",
                    lambda: iter(self.train_loader),
                )
                batch_index = 0
                window_accumulator = MetricAccumulator(self.metric_specs)
                for window_offset, window_size in enumerate(window_sizes):
                    window_index = window_offset + 1
                    window_error: BaseException | None = None
                    last_batch_index = batch_index
                    for _ in range(window_size):
                        if window_error is not None:
                            break
                        try:
                            batch = next(loader_iterator)
                            moved = self.batch_mover(batch, self.device)
                            context = StepContext(
                                training=True,
                                epoch=epoch,
                                batch_index=batch_index,
                                global_step=self.state.global_step,
                                optimizer_step=self.state.optimizer_step,
                            )
                            with self.gradient_accumulation_context():
                                with self.autocast_context():
                                    output = self.train_step(self.model, moved, context)
                                    if output.loss is None:
                                        raise ValueError(
                                            "train step must return a scalar loss"
                                        )
                                    scaled = output.loss * plan.scale_for_window(
                                        window_size,
                                        window_index=window_offset,
                                    )
                                metrics = output_metrics(output)
                                accumulator.update(metrics, weight=output.weight)
                                window_accumulator.update(
                                    metrics,
                                    weight=output.weight,
                                )
                                update_stateful_metrics(
                                    self.train_stateful_metrics,
                                    output.metric_updates,
                                )
                                backward_loss = self.scaler.scale(scaled)
                                assert isinstance(backward_loss, torch.Tensor)
                                backward_loss.backward()  # type: ignore[no-untyped-call]
                        except BaseException as error:
                            window_error = error
                        else:
                            self.state.global_step += 1
                            last_batch_index = batch_index
                            batch_index += 1
                    self.raise_distributed_failure("train step", window_error)
                    self.synchronize_gradients()
                    self.coordinate("optimizer step", self.optimizer_step)

                    window_metrics, window_stateful_baseline = self.coordinate(
                        "training metric reduction",
                        partial(
                            self.compute_training_window,
                            window_accumulator,
                            window_stateful_baseline,
                        ),
                    )
                    window_accumulator = MetricAccumulator(self.metric_specs)
                    if (
                        window_index % self.config.log_every_batches == 0
                        or window_index == len(window_sizes)
                    ):
                        routed = self.coordinate(
                            "training metric routing",
                            partial(
                                route_metrics,
                                window_metrics,
                                self.train_metric_routes,
                                "batch",
                            ),
                        )
                        self.observer.progress(
                            phase=self.config.train_phase,
                            task_id=task_id,
                            completed=window_index,
                            total=len(window_sizes),
                            metrics=routed,
                            display_metrics=display_metrics(
                                routed,
                                self.config.display_metric_names,
                            ),
                            coordinates={
                                "epoch": epoch,
                                "batch": last_batch_index,
                                "global_step": self.state.global_step,
                                "optimizer_step": self.state.optimizer_step,
                            },
                            logical_step=self.state.optimizer_step,
                            final=window_index == len(window_sizes),
                            unit="optimizer step",
                        )
            def compute_summary() -> tuple[dict[str, float], dict[str, float]]:
                scalar_summary = accumulator.compute(
                    device=self.device,
                    distributed=self.config.strategy == "ddp",
                )
                stateful_summary = compute_stateful_metrics(
                    self.train_stateful_metrics,
                    device=self.device,
                    distributed=self.config.strategy == "ddp",
                )
                summary = merge_metrics(scalar_summary, stateful_summary)
                return (
                    summary,
                    route_metrics(summary, self.train_metric_routes, "epoch"),
                )

            summary, routed_summary = self.coordinate(
                "training metric summary",
                compute_summary,
            )
        except BaseException as error:
            self.observer.emit(
                "task_failed",
                phase=self.config.train_phase,
                task_id=task_id,
                epoch=epoch,
                message=str(error),
            )
            raise
        self.observer.emit(
            "task_completed",
            phase=self.config.train_phase,
            task_id=task_id,
            epoch=epoch,
            metrics=routed_summary,
            display_metrics={},
            logical_step=epoch,
        )
        return summary

    def validate_epoch(self, epoch: int) -> Mapping[str, float]:
        """Run one no-gradient validation epoch through the project step function."""
        assert self.validation_loader is not None and self.validation_step is not None
        validation_loader = self.validation_loader
        self.coordinate("validation mode setup", self.model.eval)
        accumulator = MetricAccumulator(self.metric_specs)
        total_batches = self.coordinate(
            "validation loader length",
            lambda: len(validation_loader),
        )
        task_id = f"epoch-{epoch}"
        self.observer.emit("phase_started", phase=self.config.validation_phase)
        self.observer.emit(
            "task_started", phase=self.config.validation_phase, task_id=task_id
        )
        try:
            self.coordinate(
                "validation metric setup",
                lambda: reset_stateful_metrics(self.validation_stateful_metrics),
            )
            with self.observer.periodic_heartbeats(
                phase=self.config.validation_phase,
                task_id=task_id,
                message="Validation epoch is still active.",
            ), torch.no_grad():
                validation_error: BaseException | None = None
                validation_iterator = self.coordinate(
                    "validation loader setup",
                    lambda: iter(validation_loader),
                )
                for batch_index in range(total_batches):
                    if validation_error is not None:
                        break
                    try:
                        batch = next(validation_iterator)
                        moved = self.batch_mover(batch, self.device)
                        context = StepContext(
                            training=False,
                            epoch=epoch,
                            batch_index=batch_index,
                            global_step=self.state.global_step,
                            optimizer_step=self.state.optimizer_step,
                        )
                        with (
                            self.gradient_accumulation_context(),
                            self.autocast_context(),
                        ):
                            output = self.validation_step(self.model, moved, context)
                        metrics = output_metrics(output)
                        accumulator.update(metrics, weight=output.weight)
                        update_stateful_metrics(
                            self.validation_stateful_metrics,
                            output.metric_updates,
                        )
                        routed = route_metrics(
                            metrics,
                            self.validation_metric_routes,
                            "batch",
                        )
                    except BaseException as error:
                        validation_error = error
                        break
                    self.observer.progress(
                        phase=self.config.validation_phase,
                        task_id=task_id,
                        completed=batch_index + 1,
                        total=total_batches,
                        metrics=routed,
                        display_metrics=display_metrics(
                            routed,
                            self.config.display_metric_names,
                        ),
                        coordinates={"epoch": epoch, "batch": batch_index},
                        final=batch_index + 1 == total_batches,
                        unit="batch",
                    )
                self.raise_distributed_failure(
                    "validation step",
                    validation_error,
                )
            def compute_validation_summary() -> tuple[
                dict[str, float],
                dict[str, float],
            ]:
                scalar_summary = accumulator.compute(
                    device=self.device,
                    distributed=self.config.strategy == "ddp",
                )
                stateful_summary = compute_stateful_metrics(
                    self.validation_stateful_metrics,
                    device=self.device,
                    distributed=self.config.strategy == "ddp",
                )
                summary = merge_metrics(scalar_summary, stateful_summary)
                return (
                    summary,
                    route_metrics(
                        summary,
                        self.validation_metric_routes,
                        "epoch",
                    ),
                )

            summary, routed_summary = self.coordinate(
                "validation metric summary",
                compute_validation_summary,
            )
        except BaseException as error:
            self.observer.emit(
                "task_failed",
                phase=self.config.validation_phase,
                task_id=task_id,
                epoch=epoch,
                message=str(error),
            )
            self.observer.emit(
                "phase_failed",
                phase=self.config.validation_phase,
                message=str(error),
            )
            raise
        self.observer.emit(
            "task_completed",
            phase=self.config.validation_phase,
            task_id=task_id,
            epoch=epoch,
            metrics=routed_summary,
            display_metrics={},
            logical_step=epoch,
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

    def compute_training_window(
        self,
        accumulator: MetricAccumulator,
        stateful_baseline: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> tuple[dict[str, float], dict[str, dict[str, torch.Tensor]]]:
        """Reduce one completed logical window and capture the next baseline."""
        scalar_window = accumulator.compute(
            device=self.device,
            distributed=self.config.strategy == "ddp",
        )
        stateful_window = compute_stateful_metrics(
            self.train_stateful_metrics,
            device=self.device,
            distributed=self.config.strategy == "ddp",
            baseline=stateful_baseline,
        )
        next_baseline = snapshot_stateful_metrics(self.train_stateful_metrics)
        return merge_metrics(scalar_window, stateful_window), next_baseline

    def run_epoch_callbacks(self, metrics: Mapping[str, float]) -> None:
        """Run project callbacks at the shared post-training epoch boundary."""
        for callback in self.callbacks:
            callback.on_epoch_end(self.state, metrics)

    def run_train_start_callbacks(self) -> None:
        """Run project callbacks at the shared training-start boundary."""
        for callback in self.callbacks:
            callback.on_train_start(self.state)

    def run_train_end_callbacks(self) -> None:
        """Run project callbacks at the shared training-end boundary."""
        for callback in self.callbacks:
            callback.on_train_end(self.state)

    def run_validation_callbacks(self, metrics: Mapping[str, float]) -> None:
        """Run project callbacks at the shared post-validation boundary."""
        for callback in self.callbacks:
            callback.on_validation_end(self.state, metrics)

    def gradient_accumulation_context(self) -> Any:
        """Suppress automatic DDP reduction until the shared logical-step boundary."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.no_sync()
        return nullcontext()

    def synchronize_gradients(self) -> None:
        """Average accumulated gradients after rank-wide metadata consensus."""
        if self.config.strategy == "single":
            return
        parameters = tuple(self.base_model.named_parameters())
        local_metadata = tuple(
            (
                name,
                parameter.grad is not None,
                None if parameter.grad is None else tuple(parameter.grad.shape),
                None if parameter.grad is None else str(parameter.grad.dtype),
                None if parameter.grad is None else str(parameter.grad.layout),
            )
            for name, parameter in parameters
        )
        gathered_metadata = self.all_gather_object(local_metadata)
        if any(metadata != local_metadata for metadata in gathered_metadata):
            raise RuntimeError("gradient presence or tensor metadata differs across ranks")
        for _, parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.layout != torch.strided:
                raise ValueError("distributed accumulated gradients must use strided layout")
            if self.runtime is not None:
                self.runtime.all_reduce_sum(gradient)
            else:
                torch.distributed.all_reduce(
                    gradient,
                    op=torch.distributed.ReduceOp.SUM,
                )
            gradient.div_(self.world_size)

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
        """Return rank zero's authoritative checkpoint decision on every rank."""
        every = self.config.checkpoint_every_epochs
        local_decision = (
            self.rank == 0
            and (self.checkpoint_dir is not None or self.checkpoint_policy is not None)
            and every is not None
            and (epoch + 1) % every == 0
        )
        decisions = self.all_gather_object(local_decision if self.rank == 0 else None)
        primary_decision = decisions[0]
        if not isinstance(primary_decision, bool):
            raise RuntimeError("rank zero returned an invalid checkpoint decision")
        return primary_decision

    def publish_checkpoint_if_due(
        self,
        epoch: int,
        training_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float] | None,
    ) -> None:
        """Publish on rank zero and propagate planning or submission failures."""
        if not self.should_checkpoint(epoch):
            return
        local_error: BaseException | None = None
        with self.observer.periodic_heartbeats(
            phase=self.config.train_phase,
            message="Checkpoint publication is still active.",
        ):
            if self.rank == 0:
                try:
                    self.publish_checkpoint(
                        epoch,
                        training_metrics,
                        validation_metrics,
                    )
                except BaseException as error:
                    local_error = error
            self.raise_distributed_failure("checkpoint publication", local_error)

    def publish_checkpoint(
        self,
        epoch: int,
        training_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float] | None,
    ) -> None:
        """Submit a project plan or the default registered-state snapshot."""
        if self.checkpoint_policy is not None:
            self.publisher.wait_for_submission_slot()
            plan = self.checkpoint_policy.plan(
                TrainerCheckpointContext(
                    epoch=epoch,
                    global_step=self.state.global_step,
                    optimizer_step=self.state.optimizer_step,
                    stopped_early=self.state.stopped_early,
                    training_metrics=dict(training_metrics),
                    validation_metrics=(
                        None if validation_metrics is None else dict(validation_metrics)
                    ),
                )
            )
            if plan is not None:
                self.publisher.submit(plan)
            return

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
        """Restore project-owned or registered state before calling :meth:`fit`."""
        if self.checkpoint_policy is not None:
            if not strict:
                raise ValueError("strict=False applies only to registered-state checkpoints")
            self.coordinate(
                "checkpoint restore",
                partial(
                    self.checkpoint_policy.restore,
                    Path(path),
                    self.state,
                    device=self.device,
                ),
            )
            self.validate_restored_trainer_state()
            return
        self.coordinate(
            "checkpoint restore",
            partial(
                restore_checkpoint,
                path,
                self.registry,
                strict=strict,
                map_location=self.device,
            ),
        )
        self.validate_restored_trainer_state()

    def validate_restored_trainer_state(self) -> None:
        """Require identical restored loop coordinates on every DDP rank."""
        local_state = self.state.state_dict()
        gathered_states = self.all_gather_object(local_state)
        primary_state = gathered_states[0]
        if any(state != primary_state for state in gathered_states[1:]):
            raise RuntimeError("restored trainer state differs across ranks")

    def flush_checkpoints(self) -> None:
        """Flush rank-zero publication and propagate a coherent DDP failure."""
        local_error: BaseException | None = None
        with self.observer.periodic_heartbeats(
            phase=self.config.train_phase,
            message="Final checkpoint publication is still active.",
        ):
            if self.rank == 0:
                try:
                    self.publisher.flush()
                except BaseException as error:
                    local_error = error
            self.raise_distributed_failure("checkpoint flush", local_error)

    def raise_distributed_failure(
        self,
        operation: str,
        local_error: BaseException | None,
    ) -> None:
        """Raise the first rank-reported infrastructure failure everywhere."""
        if self.config.strategy == "single":
            if local_error is not None:
                raise local_error
            return
        failure_flag = torch.tensor(
            1 if local_error is not None else 0,
            dtype=torch.int32,
            device=self.device,
        )
        if self.runtime is not None:
            self.runtime.all_reduce_max(failure_flag)
        else:
            torch.distributed.all_reduce(
                failure_flag,
                op=torch.distributed.ReduceOp.MAX,
            )
        if not bool(failure_flag.item()):
            return
        local_status = (
            None
            if local_error is None
            else f"{type(local_error).__name__}: {local_error}"
        )
        statuses = self.all_gather_object(local_status)
        failure = next((status for status in statuses if status is not None), None)
        if failure is not None:
            raise RuntimeError(f"{operation} failed: {failure}") from local_error

    def coordinate[T](self, operation: str, function: Callable[[], T]) -> T:
        """Run project-controlled work behind rank-wide failure consensus."""
        result: T | None = None
        local_error: BaseException | None = None
        try:
            result = function()
        except BaseException as error:
            local_error = error
        self.raise_distributed_failure(operation, local_error)
        return cast(T, result)

    def validate_optimizer_window_count(self, local_count: int) -> None:
        """Require every DDP rank to reach the same optimizer-step collectives."""
        counts = self.all_gather_object(local_count)
        if any(not isinstance(count, int) or isinstance(count, bool) for count in counts):
            raise RuntimeError("accumulation policy returned invalid distributed step counts")
        if len(set(counts)) != 1:
            raise ValueError(
                "accumulation plans must produce the same optimizer-step count on every rank; "
                f"received {counts}"
            )

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Gather a small policy or failure value on every configured rank."""
        if self.config.strategy == "single":
            return (value,)
        if self.runtime is not None:
            return self.runtime.all_gather_object(value)
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, value)
        return tuple(gathered)

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
    metrics = scalar_metrics(output.metrics)
    if output.loss is not None:
        loss = scalar_metrics({"loss": output.loss})["loss"]
        metrics.setdefault("loss", loss)
    return metrics


def merge_metrics(
    scalar: Mapping[str, float],
    stateful: Mapping[str, float],
) -> dict[str, float]:
    """Merge scalar and stateful summaries without silently replacing names."""
    overlap = sorted(set(scalar).intersection(stateful))
    if overlap:
        raise ValueError(f"scalar and stateful metrics returned duplicate names: {overlap}")
    return {**scalar, **stateful}


def reset_and_snapshot_stateful_metrics(
    metrics: Mapping[str, StatefulMetric],
) -> dict[str, dict[str, torch.Tensor]]:
    """Reset metric state and capture its logical-window baseline."""
    reset_stateful_metrics(metrics)
    return snapshot_stateful_metrics(metrics)


def display_metrics(
    metrics: Mapping[str, float], names: Sequence[str]
) -> dict[str, float]:
    """Select the explicitly configured bounded live metric subset."""
    selected = {name: metrics[name] for name in names if name in metrics}
    if len(selected) > 16:
        raise ValueError("display_metric_names may select at most 16 reported metrics")
    return selected


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
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        broadcast_buffers=False,
    )


def set_sampler_epoch(loader: DataLoader[Any], epoch: int) -> None:
    """Advance distinct epoch-aware sample and batch samplers."""
    seen: set[int] = set()
    for sampler in (loader.sampler, loader.batch_sampler):
        identity = id(sampler)
        if identity in seen:
            continue
        seen.add(identity)
        method = getattr(sampler, "set_epoch", None)
        if callable(method):
            method(epoch)


def positive_integer(name: str, value: Any) -> None:
    """Validate one positive integer configuration field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
