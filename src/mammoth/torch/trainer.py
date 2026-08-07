"""Bounded generic trainer for constructed PyTorch modules and data loaders.

Consuming projects provide model, optimizer, loaders, and step functions. This
module owns only ordinary single/DDP loop mechanics, logging, and registered
checkpoint publication.
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mammoth.logging import RunObserver
from mammoth.torch.batch import CudaPrefetchingBatchIterator, move_batch_to_device
from mammoth.torch.callbacks import Callback, EarlyStopping
from mammoth.torch.checkpoint import (
    AsyncCheckpointPublisher,
    CheckpointCaptureMode,
    CheckpointComponent,
    CheckpointInspection,
    CheckpointPublication,
    CheckpointReason,
    CheckpointSavePolicy,
    RestoreOptions,
    Stateful,
    StateRegistry,
    TrainerCheckpointContext,
    TrainerCheckpointPolicy,
    TrainerCheckpointRestore,
    TrainerCheckpointWriters,
    build_trainer_checkpoint_plan,
    checkpoint_payload,
    load_checkpoint_state,
    snapshot_to_cpu,
)
from mammoth.torch.device import resolve_device
from mammoth.torch.metrics import (
    MetricAccumulator,
    MetricRoute,
    MetricSpec,
    StatefulMetric,
    compute_stateful_metrics,
    prepared_scalar_metrics,
    reset_stateful_metrics,
    route_metrics,
    scalar_metrics,
    snapshot_stateful_metrics,
    update_stateful_metrics,
)
from mammoth.torch.scheduling import AccumulationPolicy, UniformAccumulationPolicy
from mammoth.torch.state import TrainerState

if TYPE_CHECKING:
    from mammoth.torch.runtime import TorchExecutionRuntime

Precision = Literal["fp32", "bf16", "fp16"]
Strategy = Literal["single", "ddp"]
SchedulerInterval = Literal["optimizer", "epoch", "validation"]
OptimizerStepLogicalClock = Literal["completed", "zero_based"]


@dataclass(frozen=True, slots=True)
class TorchCompileConfig:
    """Project-selected ``torch.compile`` options for the execution model."""

    mode: str | None = "default"
    fullgraph: bool = False
    dynamic: bool | None = None
    backend: str | Callable[..., Any] | None = None
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode is not None and (not isinstance(self.mode, str) or not self.mode):
            raise ValueError("compile mode must be a non-empty string or None")
        if not isinstance(self.fullgraph, bool):
            raise ValueError("compile fullgraph must be a boolean")
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise ValueError("compile dynamic must be a boolean or None")
        if isinstance(self.backend, str) and not self.backend:
            raise ValueError("compile backend must be a non-empty string")
        if self.backend is not None and not isinstance(self.backend, str) and not callable(
            self.backend
        ):
            raise ValueError("compile backend must be a string, callable, or None")
        if self.options is not None and not isinstance(self.options, Mapping):
            raise ValueError("compile options must be a mapping or None")
        if self.mode is not None and self.options is not None:
            raise ValueError("compile mode and options are mutually exclusive")


@dataclass(frozen=True, slots=True)
class StepContext:
    """Ordinary loop coordinates supplied to a project step function.

    In DDP, ``global_step`` gives each rank's microbatches a deterministic
    position within the current globally counted accumulation window.
    """

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
OptimizerStepMetrics = Callable[[TrainerState], Mapping[str, float | torch.Tensor]]


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
    checkpoint_capture_mode: CheckpointCaptureMode = "auto"
    checkpoint_cuda_headroom_bytes: int = 0
    checkpoint_on_interrupt: bool = True
    non_blocking_transfer: bool = False
    cuda_prefetch: bool = True
    train_phase: str = "train"
    validation_phase: str = "validation"
    display_metric_names: tuple[str, ...] = ("loss",)
    emit_fit_phase_events: bool = True
    optimizer_step_logical_clock: OptimizerStepLogicalClock = "completed"
    compile_config: TorchCompileConfig | None = None

    def __post_init__(self) -> None:
        positive_integer("epochs", self.epochs)
        positive_integer("gradient_accumulation_steps", self.gradient_accumulation_steps)
        positive_integer("validation_every_epochs", self.validation_every_epochs)
        positive_integer("log_every_batches", self.log_every_batches)
        positive_integer("max_pending_checkpoints", self.max_pending_checkpoints)
        if self.checkpoint_capture_mode not in {"auto", "cpu"}:
            raise ValueError("checkpoint_capture_mode must be 'auto' or 'cpu'")
        if (
            isinstance(self.checkpoint_cuda_headroom_bytes, bool)
            or not isinstance(self.checkpoint_cuda_headroom_bytes, int)
            or self.checkpoint_cuda_headroom_bytes < 0
        ):
            raise ValueError("checkpoint_cuda_headroom_bytes must be a non-negative integer")
        if self.checkpoint_every_epochs is not None:
            positive_integer("checkpoint_every_epochs", self.checkpoint_every_epochs)
        if self.strategy not in {"single", "ddp"}:
            raise ValueError(f"Unsupported trainer strategy: {self.strategy!r}")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported trainer precision: {self.precision!r}")
        if self.scheduler_interval not in {"optimizer", "epoch", "validation"}:
            raise ValueError(f"Unsupported scheduler interval: {self.scheduler_interval!r}")
        if self.optimizer_step_logical_clock not in {"completed", "zero_based"}:
            raise ValueError(
                "optimizer_step_logical_clock must be 'completed' or 'zero_based'"
            )
        if self.max_gradient_norm is not None and (
            isinstance(self.max_gradient_norm, bool)
            or not math.isfinite(self.max_gradient_norm)
            or self.max_gradient_norm <= 0
        ):
            raise ValueError("max_gradient_norm must be positive and finite")
        if not isinstance(self.emit_fit_phase_events, bool):
            raise ValueError("emit_fit_phase_events must be a boolean")
        if not isinstance(self.checkpoint_on_interrupt, bool):
            raise ValueError("checkpoint_on_interrupt must be a boolean")
        if not isinstance(self.cuda_prefetch, bool):
            raise ValueError("cuda_prefetch must be a boolean")
        if self.compile_config is not None and not isinstance(
            self.compile_config, TorchCompileConfig
        ):
            raise ValueError("compile_config must be TorchCompileConfig or None")
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
        optimizer_step_metrics: OptimizerStepMetrics | None = None,
        metric_specs: Mapping[str, MetricSpec] | None = None,
        train_metric_routes: Mapping[str, MetricRoute] | None = None,
        validation_metric_routes: Mapping[str, MetricRoute] | None = None,
        train_stateful_metrics: Mapping[str, StatefulMetric] | None = None,
        validation_stateful_metrics: Mapping[str, StatefulMetric] | None = None,
        accumulation_policy: AccumulationPolicy | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_policy: TrainerCheckpointPolicy | None = None,
        checkpoint_save_policy: CheckpointSavePolicy | None = None,
        extra_state: Mapping[str, Stateful] | None = None,
        batch_mover: BatchMover | None = None,
        runtime: TorchExecutionRuntime | None = None,
    ) -> None:
        if (validation_loader is None) != (validation_step is None):
            raise ValueError("validation_loader and validation_step must be provided together")
        if checkpoint_save_policy is not None and (
            checkpoint_dir is None or checkpoint_policy is None
        ):
            raise ValueError(
                "checkpoint_save_policy requires checkpoint_dir and checkpoint_policy"
            )
        if (
            checkpoint_dir is not None
            and checkpoint_policy is not None
            and checkpoint_save_policy is None
        ):
            raise ValueError(
                "project checkpoint publication requires checkpoint_save_policy"
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
        wrapped_model = wrap_model(self.base_model, self.device, config.strategy)
        self._ddp_model = (
            wrapped_model if isinstance(wrapped_model, DistributedDataParallel) else None
        )
        self.execution_model = compile_execution_model(
            wrapped_model,
            config.compile_config,
        )
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
        early_stopping_callbacks = tuple(
            callback for callback in self.callbacks if isinstance(callback, EarlyStopping)
        )
        if checkpoint_save_policy is not None and checkpoint_save_policy.save_best:
            if validation_loader is None:
                raise ValueError("save_best requires validation")
            if len(early_stopping_callbacks) != 1:
                raise ValueError(
                    "save_best requires exactly one Mammoth EarlyStopping callback"
                )
        self._checkpoint_early_stopping = (
            early_stopping_callbacks[0] if len(early_stopping_callbacks) == 1 else None
        )
        self.optimizer_step_metrics = optimizer_step_metrics
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
        self.checkpoint_save_policy = checkpoint_save_policy
        self._uses_default_batch_mover = batch_mover is None
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
        self._initial_optimizer_state = snapshot_to_cpu(self.optimizer.state_dict())
        self._initial_scheduler_state = (
            None if self.scheduler is None else snapshot_to_cpu(self.scheduler.state_dict())
        )
        self._initial_callback_states = tuple(
            snapshot_to_cpu(callback.state_dict()) for callback in self.callbacks
        )
        self._checkpoint_restore: TrainerCheckpointRestore | None = None
        self.publisher = AsyncCheckpointPublisher(
            max_pending=config.max_pending_checkpoints,
            capture_mode=config.checkpoint_capture_mode,
            cuda_headroom_bytes=config.checkpoint_cuda_headroom_bytes,
        )
        self._checkpoint_publication_futures: deque[
            Future[CheckpointPublication]
        ] = deque()
        self._closed = False

    def fit(self) -> TrainerResult:
        """Execute configured epochs and return project metric summaries."""
        training_history: list[Mapping[str, float]] = []
        validation_history: list[Mapping[str, float]] = []
        if self.state.stopped_early:
            return TrainerResult(
                state=self.state,
                training_history=(),
                validation_history=(),
            )
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
            if isinstance(error, KeyboardInterrupt) and self.config.checkpoint_on_interrupt:
                try:
                    self.publish_interrupted_checkpoint()
                except BaseException as checkpoint_error:
                    error.add_note(
                        "Interrupted checkpoint publication also failed: "
                        f"{checkpoint_error}"
                    )
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
                    if isinstance(fit_error, KeyboardInterrupt):
                        self.flush_local_checkpoints(
                            message="Interrupted checkpoint shutdown is still active."
                        )
                    else:
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
        self.coordinate("training mode setup", self.execution_model.train)
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
        global_window_sizes, rank_window_offsets = self.distributed_window_layout(
            window_sizes
        )
        task_id = f"epoch-{epoch}"
        self.observer.emit(
            "task_started",
            phase=self.config.train_phase,
            task_id=task_id,
            epoch=epoch,
            epoch_total=self.config.epochs,
        )
        batch_iterator: CudaPrefetchingBatchIterator | None = None
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
                batch_iterator = self.batch_iterator(loader_iterator)
                batch_index = 0
                window_accumulator = MetricAccumulator(self.metric_specs)
                for window_offset, window_size in enumerate(window_sizes):
                    window_index = window_offset + 1
                    window_global_step = self.state.global_step
                    window_error: BaseException | None = None
                    final_backward_loss: torch.Tensor | None = None
                    last_batch_index = batch_index
                    for local_window_offset in range(window_size):
                        if window_error is not None:
                            break
                        synchronizes_gradients = local_window_offset + 1 == window_size
                        try:
                            moved = next(batch_iterator)
                            context = StepContext(
                                training=True,
                                epoch=epoch,
                                batch_index=batch_index,
                                global_step=(
                                    window_global_step
                                    + rank_window_offsets[window_offset]
                                    + local_window_offset
                                ),
                                optimizer_step=self.state.optimizer_step,
                            )
                            with self.gradient_accumulation_context(
                                synchronizes_gradients=synchronizes_gradients
                            ):
                                with self.autocast_context():
                                    output = self.train_step(self.execution_model, moved, context)
                                    if output.loss is None:
                                        raise ValueError(
                                            "train step must return a scalar loss"
                                        )
                                    scaled = output.loss * plan.scale_for_window(
                                        window_size,
                                        window_index=window_offset,
                                    )
                                metrics = output_metrics(output)
                                required_finite = {"loss": output.loss}
                                accumulator.update(
                                    metrics,
                                    weight=output.weight,
                                    required_finite=required_finite,
                                )
                                window_accumulator.update(
                                    metrics,
                                    weight=output.weight,
                                    required_finite=required_finite,
                                )
                                update_stateful_metrics(
                                    self.train_stateful_metrics,
                                    output.metric_updates,
                                )
                                backward_loss = self.scaler.scale(scaled)
                                assert isinstance(backward_loss, torch.Tensor)
                                if synchronizes_gradients and self._ddp_model is not None:
                                    final_backward_loss = backward_loss
                                else:
                                    backward_loss.backward()  # type: ignore[no-untyped-call]
                        except BaseException as error:
                            window_error = error
                        else:
                            last_batch_index = batch_index
                            batch_index += 1
                    self.raise_distributed_failure("train step", window_error)
                    if self._ddp_model is not None:
                        assert final_backward_loss is not None
                        final_backward_loss.backward()  # type: ignore[no-untyped-call]
                    self.coordinate("optimizer step", self.optimizer_step)
                    self.state.global_step += global_window_sizes[window_offset]
                    optimizer_step_metrics = self.optimizer_step_metrics
                    if optimizer_step_metrics is not None:
                        optimizer_metrics = self.coordinate(
                            "optimizer-step metrics",
                            partial(
                                self.compute_optimizer_step_metrics,
                                optimizer_step_metrics,
                            ),
                        )
                        accumulator.update(optimizer_metrics)
                        window_accumulator.update(optimizer_metrics)

                    should_observe = (
                        window_index % self.config.log_every_batches == 0
                        or window_index == len(window_sizes)
                    )
                    if should_observe:
                        window_metrics, window_stateful_baseline = self.coordinate(
                            "training metric reduction",
                            partial(
                                self.compute_training_window,
                                window_accumulator,
                                window_stateful_baseline,
                            ),
                        )
                    else:
                        window_stateful_baseline = self.coordinate(
                            "training metric baseline",
                            partial(snapshot_stateful_metrics, self.train_stateful_metrics),
                        )
                    window_accumulator = MetricAccumulator(self.metric_specs)
                    if should_observe:
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
                            logical_step=self.optimizer_step_logical_step(),
                            final=window_index == len(window_sizes),
                        )
                        self.observer.flush()
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
        finally:
            if batch_iterator is not None:
                batch_iterator.close()
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
        self.coordinate("validation mode setup", self.execution_model.eval)
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
        batch_iterator: CudaPrefetchingBatchIterator | None = None
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
                batch_iterator = self.batch_iterator(validation_iterator)
                for batch_index in range(total_batches):
                    if validation_error is not None:
                        break
                    try:
                        moved = next(batch_iterator)
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
                            output = self.validation_step(self.execution_model, moved, context)
                        metrics = output_metrics(output)
                        required_finite = (
                            {} if output.loss is None else {"loss": output.loss}
                        )
                        accumulator.update(
                            metrics,
                            weight=output.weight,
                            required_finite=required_finite,
                        )
                        update_stateful_metrics(
                            self.validation_stateful_metrics,
                            output.metric_updates,
                        )
                        routed = route_metrics(
                            scalar_metrics(
                                metrics,
                                required_finite=required_finite,
                            ),
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
        finally:
            if batch_iterator is not None:
                batch_iterator.close()
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

    def compute_optimizer_step_metrics(
        self,
        provider: OptimizerStepMetrics,
    ) -> dict[str, float | torch.Tensor]:
        """Validate consumer metrics after one optimizer/scheduler boundary."""
        return prepared_scalar_metrics(provider(self.state))

    def optimizer_step_logical_step(self) -> int:
        """Return the configured sink clock without changing completed-step state."""
        if self.config.optimizer_step_logical_clock == "zero_based":
            return self.state.optimizer_step - 1
        return self.state.optimizer_step

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

    def deliver_checkpoint_publication(
        self,
        publication: CheckpointPublication,
    ) -> None:
        """Deliver one committed publication through observer and callback lifecycles."""
        checkpoints = [
            {
                "path": str(checkpoint.path),
                "role": checkpoint.role,
                "epoch": checkpoint.epoch,
                "size_bytes": checkpoint.size_bytes,
                "sha256": checkpoint.sha256,
            }
            for checkpoint in publication.published
        ]
        self.observer.emit(
            "task_completed",
            phase=self.config.train_phase,
            task_id="checkpoint-publication",
            checkpoints=checkpoints,
            retired=[str(path) for path in publication.retired],
        )
        for callback in self.callbacks:
            callback.on_checkpoint_published(self.state, publication)

    def gradient_accumulation_context(self, *, synchronizes_gradients: bool = False) -> Any:
        """Suppress DDP reduction only before a native window's final backward."""
        if self._ddp_model is not None and not synchronizes_gradients:
            return self._ddp_model.no_sync()
        return nullcontext()

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

    def checkpoint_selection(
        self,
        epoch: int,
        validation_metrics: Mapping[str, float] | None,
    ) -> tuple[bool, bool]:
        """Return rank zero's resumable/best selection on every rank."""
        local_selection: tuple[bool, bool] | None = None
        if self.rank == 0:
            if self.checkpoint_save_policy is not None:
                policy = self.checkpoint_save_policy
                save_resumable = (epoch + 1) % policy.every_epochs == 0
                early_stopping = self._checkpoint_early_stopping
                save_best = bool(
                    policy.save_best
                    and validation_metrics is not None
                    and early_stopping is not None
                    and early_stopping.improved
                )
                local_selection = (save_resumable, save_best)
            else:
                every = self.config.checkpoint_every_epochs
                local_selection = (
                    self.checkpoint_dir is not None
                    and every is not None
                    and (epoch + 1) % every == 0,
                    False,
                )
        decisions = self.all_gather_object(local_selection)
        primary_decision = decisions[0]
        if (
            not isinstance(primary_decision, tuple)
            or len(primary_decision) != 2
            or any(not isinstance(value, bool) for value in primary_decision)
        ):
            raise RuntimeError("rank zero returned an invalid checkpoint selection")
        return primary_decision

    def publish_checkpoint_if_due(
        self,
        epoch: int,
        training_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float] | None,
    ) -> None:
        """Publish on rank zero and propagate planning or submission failures."""
        save_resumable, save_best = self.checkpoint_selection(
            epoch,
            validation_metrics,
        )
        if not save_resumable and not save_best:
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
                        save_resumable=save_resumable,
                        save_best=save_best,
                    )
                except BaseException as error:
                    local_error = error
            self.raise_distributed_failure("checkpoint publication", local_error)

    def publish_checkpoint(
        self,
        epoch: int,
        training_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float] | None,
        *,
        reason: CheckpointReason = "scheduled",
        save_resumable: bool = True,
        save_best: bool = False,
    ) -> None:
        """Capture project state and submit Mammoth-selected artifacts."""
        if self.checkpoint_policy is not None:
            if self.checkpoint_dir is None or self.checkpoint_save_policy is None:
                raise RuntimeError("project checkpoint publication is not configured")
            self.publisher.wait_for_submission_slot()
            writers = self.checkpoint_policy.capture(
                TrainerCheckpointContext(
                    epoch=epoch,
                    global_step=self.state.global_step,
                    optimizer_step=self.state.optimizer_step,
                    stopped_early=self.state.stopped_early,
                    training_metrics=dict(training_metrics),
                    validation_metrics=(
                        None if validation_metrics is None else dict(validation_metrics)
                    ),
                    reason=reason,
                    restore=self._checkpoint_restore,
                )
            )
            if not isinstance(writers, TrainerCheckpointWriters):
                raise TypeError(
                    "checkpoint policy must return TrainerCheckpointWriters"
                )
            plan = build_trainer_checkpoint_plan(
                self.checkpoint_dir,
                epoch=epoch,
                save_policy=self.checkpoint_save_policy,
                writers=writers,
                save_resumable=save_resumable,
                save_best=save_best,
            )
            self._checkpoint_publication_futures.append(self.publisher.submit(plan))
            return

        assert self.checkpoint_dir is not None
        filename = self.config.checkpoint_filename.format(
            epoch=epoch,
            global_step=self.state.global_step,
            optimizer_step=self.state.optimizer_step,
        )
        if Path(filename).name != filename:
            raise ValueError("formatted checkpoint filename must remain a single filename")
        destination = self.checkpoint_dir / filename
        self._checkpoint_publication_futures.append(
            self.publisher.publish_with_receipt(
                destination,
                checkpoint_payload(self.registry),
                role="epoch",
                epoch=epoch,
            )
        )

    def publish_checkpoint_now(
        self,
        *,
        reason: Literal["manual", "interrupted"] = "manual",
        training_metrics: Mapping[str, float] | None = None,
        validation_metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Force one synchronous checkpoint publication on rank zero."""
        if reason not in {"manual", "interrupted"}:
            raise ValueError("forced checkpoint reason must be 'manual' or 'interrupted'")
        local_available = self.rank == 0 and self.checkpoint_publication_configured()
        decisions = self.all_gather_object(local_available if self.rank == 0 else None)
        if not isinstance(decisions[0], bool):
            raise RuntimeError("rank zero returned an invalid checkpoint availability decision")
        if not decisions[0]:
            return
        local_error: BaseException | None = None
        if self.rank == 0:
            try:
                self.publish_checkpoint(
                    self.state.epoch,
                    training_metrics or {},
                    validation_metrics,
                    reason=reason,
                    save_resumable=True,
                    save_best=False,
                )
            except BaseException as error:
                local_error = error
        self.raise_distributed_failure("forced checkpoint publication", local_error)
        self.flush_checkpoints()

    def publish_interrupted_checkpoint(self) -> None:
        """Publish rank zero's interruption snapshot without entering collectives."""
        if self.rank != 0 or not self.checkpoint_publication_configured():
            return
        self.publish_checkpoint(
            self.state.epoch,
            {},
            None,
            reason="interrupted",
        )
        self.flush_local_checkpoints(
            message="Interrupted checkpoint publication is still active."
        )

    def checkpoint_publication_configured(self) -> bool:
        """Return whether this trainer has a complete checkpoint save contract."""
        if self.checkpoint_dir is None:
            return False
        if self.checkpoint_policy is None:
            return True
        return self.checkpoint_save_policy is not None

    def inspect_checkpoint(self, path: Path) -> CheckpointInspection:
        """Inspect one checkpoint on rank zero and share the typed result."""
        checkpoint_path = Path(path)

        def inspect_primary() -> CheckpointInspection | None:
            if self.rank != 0:
                return None
            if self.checkpoint_policy is not None:
                inspection = self.checkpoint_policy.inspect(
                    checkpoint_path,
                )
                if not isinstance(inspection, CheckpointInspection):
                    raise TypeError("checkpoint policy must return CheckpointInspection")
                pickle.dumps(inspection)
                return inspection
            state = load_checkpoint_state(checkpoint_path, map_location="cpu")
            return CheckpointInspection(
                available_components=self.checkpoint_components(state),
            )

        local_inspection = self.coordinate("checkpoint inspection", inspect_primary)
        gathered = self.all_gather_object(local_inspection)
        inspection = gathered[0]
        if not isinstance(inspection, CheckpointInspection):
            raise RuntimeError("rank zero returned an invalid checkpoint inspection")
        return inspection

    def load_checkpoint(
        self,
        path: Path,
        *,
        options: RestoreOptions | None = None,
        strict: bool = True,
    ) -> TrainerCheckpointRestore:
        """Restore selected generic state and return the synchronized typed report."""
        checkpoint_path = Path(path)
        restore_request = (str(checkpoint_path), options, strict)
        self.coordinate(
            "checkpoint restore request serialization",
            partial(pickle.dumps, restore_request),
        )
        gathered_requests = self.all_gather_object(restore_request)
        for request in gathered_requests:
            if (
                not isinstance(request, tuple)
                or len(request) != 3
                or not isinstance(request[0], str)
                or (
                    request[1] is not None
                    and not isinstance(request[1], RestoreOptions)
                )
                or not isinstance(request[2], bool)
            ):
                raise TypeError(
                    "checkpoint restore request must contain a path, "
                    "RestoreOptions or None, and strict boolean"
                )
        if any(request != gathered_requests[0] for request in gathered_requests[1:]):
            raise RuntimeError("checkpoint restore request differs across ranks")
        restore_options = options
        if restore_options is None:
            restore_options = self.inspect_checkpoint(checkpoint_path).restore_options

        if self.checkpoint_policy is not None:
            if not strict:
                raise ValueError("strict=False applies only to registered-state checkpoints")
            restored = self.coordinate(
                "checkpoint restore",
                partial(
                    self.checkpoint_policy.restore,
                    checkpoint_path,
                    device=self.device,
                    options=restore_options,
                ),
            )
        else:
            state = self.coordinate(
                "checkpoint restore",
                partial(
                    load_checkpoint_state,
                    checkpoint_path,
                    map_location=self.device,
                ),
            )
            restored = self.coordinate(
                "registered checkpoint translation",
                partial(self.registered_checkpoint_restore, state, strict=strict),
            )
        restored = self.synchronize_checkpoint_restore_payload(restored)
        local_window_sizes = self.coordinate(
            "checkpoint restore accumulation planning",
            self.checkpoint_restore_window_sizes,
        )
        global_window_sizes, _ = self.distributed_window_layout(local_window_sizes)
        applied = self.coordinate(
            "checkpoint state application",
            partial(
                self.apply_checkpoint_restore,
                restored,
                restore_options,
                global_window_sizes=global_window_sizes,
            ),
        )
        self.validate_restored_trainer_state(applied)
        self._checkpoint_restore = applied
        return applied

    def checkpoint_components(
        self,
        state: Mapping[str, Any],
    ) -> frozenset[CheckpointComponent]:
        """Return generic component categories present in registered state."""
        components: set[CheckpointComponent] = set()
        if "model" in state:
            components.add("model")
        if "optimizer" in state:
            components.add("optimizer")
        if "scheduler" in state:
            components.add("scheduler")
        if "trainer" in state:
            components.update(("trainer", "stopped_early"))
        if "scaler" in state:
            components.add("scaler")
        if any(name.startswith("callback-") for name in state):
            components.add("callbacks")
        if any(name.startswith("project-") for name in state):
            components.add("project")
        return frozenset(components)

    def registered_checkpoint_restore(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool,
    ) -> TrainerCheckpointRestore:
        """Translate registered state into the same typed restore contract."""
        missing = sorted(self.registry.names.difference(state))
        unexpected = sorted(set(state).difference(self.registry.names))
        if strict and (missing or unexpected):
            raise ValueError(
                f"checkpoint state mismatch; missing={missing}, unexpected={unexpected}"
            )
        if any(not isinstance(value, Mapping) for value in state.values()):
            raise ValueError("registered checkpoint component states must be mappings")

        managed_names = {
            "optimizer",
            "scheduler",
            "trainer",
            *(name for name in state if name.startswith("callback-")),
        }
        direct_state = {
            name: value
            for name, value in state.items()
            if name in self.registry.names and name not in managed_names
        }
        self.registry.load_state_dict(direct_state, strict=False)
        components = set(self.checkpoint_components(direct_state))

        restored_state = TrainerState()
        trainer_state = state.get("trainer")
        if trainer_state is not None:
            restored_state.load_state_dict(cast(Mapping[str, Any], trainer_state))
        else:
            restored_state.load_state_dict(self.state.state_dict())
        callback_states: dict[int, Mapping[str, Any]] = {}
        for name, value in state.items():
            if not name.startswith("callback-"):
                continue
            if name not in self.registry.names:
                continue
            index_text = name.removeprefix("callback-")
            if not index_text.isdecimal():
                raise ValueError(f"invalid registered callback state name: {name!r}")
            callback_states[int(index_text)] = cast(Mapping[str, Any], value)
        return TrainerCheckpointRestore(
            epoch=restored_state.epoch,
            optimizer_step=restored_state.optimizer_step,
            global_step=restored_state.global_step,
            stopped_early=restored_state.stopped_early,
            optimizer_state_dict=cast(
                Mapping[str, Any] | None,
                state.get("optimizer"),
            ),
            scheduler_state_dict=cast(
                Mapping[str, Any] | None,
                state.get("scheduler"),
            ),
            callback_state_dicts=callback_states,
            restored_components=frozenset(components),
        )

    def apply_checkpoint_restore(
        self,
        restored: TrainerCheckpointRestore,
        options: RestoreOptions,
        *,
        global_window_sizes: tuple[int, ...],
    ) -> TrainerCheckpointRestore:
        """Apply generic component actions and infer absent loop cursors."""
        if not isinstance(restored, TrainerCheckpointRestore):
            raise TypeError("checkpoint policy must return TrainerCheckpointRestore")
        if not isinstance(options, RestoreOptions):
            raise TypeError("checkpoint restore options must be RestoreOptions")
        managed_components = {
            "optimizer",
            "scheduler",
            "callbacks",
            "trainer",
            "stopped_early",
        }
        pre_reported_managed = (
            restored.restored_components | restored.reset_components
        ) & managed_components
        if pre_reported_managed:
            raise ValueError(
                "checkpoint policies cannot pre-report Mammoth-managed components: "
                f"{sorted(pre_reported_managed)}"
            )
        restored_components = set(restored.restored_components)
        reset_components = set(restored.reset_components)

        if options.optimizer == "restore" and restored.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(dict(restored.optimizer_state_dict))
            restored_components.add("optimizer")
        elif options.optimizer == "reset":
            self.optimizer.load_state_dict(self._initial_optimizer_state)
            reset_components.add("optimizer")

        if self.scheduler is not None:
            if options.scheduler == "restore" and restored.scheduler_state_dict is not None:
                self.scheduler.load_state_dict(dict(restored.scheduler_state_dict))
                self.synchronize_optimizer_scheduler_rates()
                restored_components.add("scheduler")
            elif options.scheduler == "reset":
                assert self._initial_scheduler_state is not None
                self.scheduler.load_state_dict(self._initial_scheduler_state)
                self.synchronize_optimizer_scheduler_rates()
                reset_components.add("scheduler")

        if options.callbacks == "restore":
            for index, state in restored.callback_state_dicts.items():
                if index >= len(self.callbacks):
                    raise ValueError(f"checkpoint callback index {index} is not registered")
                self.callbacks[index].load_state_dict(state)
            if restored.callback_state_dicts:
                restored_components.add("callbacks")
        else:
            for callback, initial_state in zip(
                self.callbacks,
                self._initial_callback_states,
                strict=True,
            ):
                callback.load_state_dict(initial_state)
            reset_components.add("callbacks")

        completed_epochs = restored.epoch + 1
        self.state.epoch = restored.epoch
        self.state.optimizer_step = (
            restored.optimizer_step
            if restored.optimizer_step is not None
            else completed_epochs * len(global_window_sizes)
        )
        self.state.global_step = (
            restored.global_step
            if restored.global_step is not None
            else completed_epochs * sum(global_window_sizes)
        )
        restored_components.add("trainer")
        if options.stopped_early == "restore":
            self.state.stopped_early = restored.stopped_early
            restored_components.add("stopped_early")
        else:
            self.state.stopped_early = False
            reset_components.add("stopped_early")

        return replace(
            restored,
            stopped_early=self.state.stopped_early,
            optimizer_state_dict=None,
            scheduler_state_dict=None,
            callback_state_dicts={},
            restored_components=frozenset(restored_components),
            reset_components=frozenset(reset_components),
        )

    def checkpoint_restore_window_sizes(self) -> tuple[int, ...]:
        """Plan local accumulation without entering a distributed collective."""
        local_batch_count = len(self.train_loader)
        plan = self.accumulation_policy.plan(
            rank=self.rank,
            world_size=self.world_size,
            local_batch_count=local_batch_count,
        )
        return plan.window_sizes(local_batch_count)

    def synchronize_checkpoint_restore_payload(
        self,
        restored: object,
    ) -> TrainerCheckpointRestore:
        """Use rank zero's validated generic restore payload on every rank."""
        validated = self.coordinate(
            "checkpoint restore payload contract",
            partial(self.require_checkpoint_restore, restored),
        )
        self.coordinate(
            "checkpoint restore payload serialization",
            partial(pickle.dumps, validated),
        )
        synchronized = self.broadcast_object(validated if self.rank == 0 else None)
        if not isinstance(synchronized, TrainerCheckpointRestore):
            raise RuntimeError("rank zero returned an invalid checkpoint restore payload")
        return synchronized

    def require_checkpoint_restore(self, restored: object) -> TrainerCheckpointRestore:
        """Validate one rank-local project restore result before collectives."""
        if not isinstance(restored, TrainerCheckpointRestore):
            raise TypeError("checkpoint policy must return TrainerCheckpointRestore")
        return restored

    def synchronize_optimizer_scheduler_rates(self) -> None:
        """Apply a reset scheduler's initial rates to optimizer parameter groups."""
        if self.scheduler is None:
            return
        learning_rates = self.scheduler.get_last_lr()
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise ValueError("scheduler learning-rate count does not match optimizer groups")
        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            parameter_group["lr"] = learning_rate

    def validate_restored_trainer_state(
        self,
        restored: TrainerCheckpointRestore,
    ) -> None:
        """Require identical restored coordinates and reports on every DDP rank."""
        local_state = self.state.state_dict()
        local_report = {
            "state": local_state,
            "restored_components": sorted(restored.restored_components),
            "reset_components": sorted(restored.reset_components),
        }
        gathered_states = self.all_gather_object(local_report)
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
                    self.flush_checkpoint_publications()
                except BaseException as error:
                    local_error = error
            self.raise_distributed_failure("checkpoint flush", local_error)

    def flush_local_checkpoints(self, *, message: str) -> None:
        """Flush this rank's publisher without entering distributed collectives."""
        with self.observer.periodic_heartbeats(
            phase=self.config.train_phase,
            message=message,
        ):
            if self.rank == 0:
                self.flush_checkpoint_publications()

    def flush_checkpoint_publications(self) -> None:
        """Flush the worker and deliver every successful retained receipt once."""
        first_error: BaseException | None = None
        try:
            self.publisher.flush()
        except BaseException as error:
            first_error = error
        while self._checkpoint_publication_futures:
            future = self._checkpoint_publication_futures.popleft()
            try:
                publication = future.result()
            except BaseException as error:
                first_error = combine_failures(first_error, error)
                continue
            try:
                self.deliver_checkpoint_publication(publication)
            except BaseException as error:
                first_error = combine_failures(first_error, error)
        if first_error is not None:
            raise first_error

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
        local_status = (
            None
            if local_error is None
            else (
                "interrupted" if isinstance(local_error, KeyboardInterrupt) else "failed",
                f"{type(local_error).__name__}: {local_error}",
            )
        )
        statuses = self.all_gather_object(local_status)
        if all(status is None for status in statuses):
            return
        interruption = next(
            (
                status
                for status in statuses
                if isinstance(status, tuple)
                and len(status) == 2
                and status[0] == "interrupted"
            ),
            None,
        )
        if interruption is not None:
            if isinstance(local_error, KeyboardInterrupt):
                raise local_error
            raise KeyboardInterrupt(f"{operation} interrupted: {interruption[1]}")
        failure = next(
            (
                status
                for status in statuses
                if isinstance(status, tuple) and len(status) == 2
            ),
            None,
        )
        if failure is not None:
            raise RuntimeError(f"{operation} failed: {failure[1]}") from local_error

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

    def distributed_window_layout(
        self,
        local_sizes: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return global window sizes and this rank's deterministic offsets."""
        gathered = self.all_gather_object(local_sizes)
        if any(
            not isinstance(sizes, tuple)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in sizes
            )
            for sizes in gathered
        ):
            raise RuntimeError("accumulation policy returned invalid distributed window sizes")
        window_counts = tuple(len(sizes) for sizes in gathered)
        if len(set(window_counts)) != 1:
            raise ValueError(
                "accumulation plans must produce the same optimizer-step count on every rank; "
                f"received {window_counts}"
            )
        global_sizes = tuple(
            sum(sizes[index] for sizes in gathered)
            for index in range(window_counts[0])
        )
        rank_offsets = tuple(
            sum(gathered[rank][index] for rank in range(self.rank))
            for index in range(window_counts[0])
        )
        return global_sizes, rank_offsets

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Gather a small policy or failure value on every configured rank."""
        if self.config.strategy == "single":
            return (value,)
        if self.runtime is not None:
            return self.runtime.all_gather_object(value)
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, value)
        return tuple(gathered)

    def broadcast_object(self, value: Any, *, source_rank: int = 0) -> Any:
        """Broadcast a small authoritative value from one configured rank."""
        if self.config.strategy == "single":
            return value
        if self.runtime is not None:
            return self.runtime.broadcast_object(value, source_rank=source_rank)
        objects = [value]
        torch.distributed.broadcast_object_list(objects, src=source_rank)
        return objects[0]

    def close(self) -> None:
        """Flush and close checkpoint publication resources."""
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self.flush_checkpoint_publications()
        except BaseException as error:
            first_error = error
        try:
            self.publisher.close()
        except BaseException as error:
            first_error = combine_failures(first_error, error)
        if first_error is not None:
            raise first_error

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

    def _prefetch_batch_mover(self, batch: Any, device: torch.device) -> Any:
        return move_batch_to_device(batch, device, non_blocking=True)

    def batch_iterator(self, batches: Iterator[Any]) -> CudaPrefetchingBatchIterator:
        """Return the bounded default-mover CUDA prefetch pipeline for one loader."""
        prefetch = self.config.cuda_prefetch and self._uses_default_batch_mover
        return CudaPrefetchingBatchIterator(
            batches,
            self.device,
            self.batch_mover,
            enabled=prefetch,
            prefetch_mover=self._prefetch_batch_mover if prefetch else None,
        )


def output_metrics(output: StepOutput) -> dict[str, float | torch.Tensor]:
    """Validate and detach scalar tensors without forcing host materialization."""
    metrics = prepared_scalar_metrics(output.metrics)
    if output.loss is not None:
        loss = prepared_scalar_metrics({"loss": output.loss})["loss"]
        metrics.setdefault("loss", loss)
    return metrics


def combine_failures(
    primary: BaseException | None,
    secondary: BaseException,
) -> BaseException:
    """Preserve the first failure while retaining later cleanup context."""
    if primary is None:
        return secondary
    if secondary is not primary:
        primary.add_note(
            "A later checkpoint lifecycle failure also occurred: "
            f"{type(secondary).__name__}: {secondary}"
        )
    return primary


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


def compile_execution_model(
    model: torch.nn.Module,
    config: TorchCompileConfig | None,
) -> torch.nn.Module:
    """Apply caller-selected compilation after device placement and DDP wrapping."""
    if config is None:
        return model
    compiled = torch.compile(
        model,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
        backend="inductor" if config.backend is None else config.backend,
        mode=config.mode,
        options=None if config.options is None else dict(config.options),
    )
    return cast(torch.nn.Module, compiled)


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
