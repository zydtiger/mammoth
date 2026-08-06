"""Project-neutral learning-rate and distributed-workload scheduling.

The trainer consumes the reusable warmup-linear scheduler and generic weighted
workload plans without assigning project, hardware, model, or dataset meaning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Protocol

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Sampler

IncompleteWindow = Literal["step", "error"]
PartialWindow = Literal["error", "fixed"]

_TORCH_MIN_SEED = -(2**63)
_TORCH_MAX_SEED = 2**64 - 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WeightedTaskAssignment:
    """Assign one opaque caller task and its cost to a rank."""

    task_id: str
    cost: int | float
    rank: int


class WarmupLinearLR(LRScheduler):
    """Warm linearly from zero, then decay linearly to zero.

    Consuming projects choose ``warmup_ratio`` and ``total_steps``. Saved state
    may resume against the same horizon or a longer configured horizon; a
    shorter configured horizon is rejected. The serialized fields intentionally
    match the historical TiSAM scheduler for checkpoint compatibility.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_ratio: float,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer")
        if (
            isinstance(warmup_ratio, bool)
            or not isinstance(warmup_ratio, int | float)
            or not math.isfinite(warmup_ratio)
            or not 0 <= warmup_ratio < 1
        ):
            raise ValueError("warmup_ratio must be finite and satisfy 0 <= value < 1")
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return BERT-style warmup-linear rates at the current step."""
        step = self.last_epoch
        warmup_steps = int(self.warmup_ratio * self.total_steps)
        if step < warmup_steps:
            multiplier = float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, self.total_steps - warmup_steps)
            )
            multiplier = max(0.0, 1.0 - progress)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore progress while retaining an intentionally extended horizon."""
        configured_total_steps = self.total_steps
        saved_total_steps = state_dict.get("total_steps")
        if not isinstance(saved_total_steps, int) or isinstance(saved_total_steps, bool):
            super().load_state_dict(state_dict)
            return
        if configured_total_steps < saved_total_steps:
            raise ValueError(
                "Cannot resume WarmupLinearLR with a shorter configured horizon: "
                f"checkpoint total_steps={saved_total_steps}, "
                f"configured total_steps={configured_total_steps}."
            )

        super().load_state_dict(state_dict)
        if configured_total_steps == saved_total_steps:
            return

        self.total_steps = configured_total_steps
        resumed_lrs = self.get_lr()
        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            resumed_lrs,
            strict=True,
        ):
            parameter_group["lr"] = learning_rate
        self._last_lr = resumed_lrs
        logger.warning(
            "Extended WarmupLinearLR horizon from %d to %d total steps at restored "
            "step %d; rebased optimizer learning rates onto the extended schedule.",
            saved_total_steps,
            configured_total_steps,
            self.last_epoch,
        )


def _validated_rank_weights(
    rank_weights: Sequence[int | float],
) -> tuple[int | float, ...]:
    """Return finite positive caller-supplied rank weights."""
    if not rank_weights:
        raise ValueError("rank_weights must contain at least one weight")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, int | float)
        or (isinstance(weight, float) and not math.isfinite(weight))
        or weight <= 0
        for weight in rank_weights
    ):
        raise ValueError("rank_weights must contain only positive finite numbers")
    return tuple(rank_weights)


def allocate_weighted_tasks(
    tasks: Sequence[tuple[str, int | float]],
    rank_weights: Sequence[int | float],
) -> tuple[WeightedTaskAssignment, ...]:
    """Allocate opaque costed tasks by lowest projected normalized rank load.

    Tasks are considered largest-cost-first with task-ID tie-breaking. Each task
    is assigned to the rank minimizing ``(current load + cost) / rank weight``;
    an exact tie selects the lower rank. The returned assignments are sorted by
    task ID so input ordering never affects the result.
    """
    weights = _validated_rank_weights(rank_weights)
    exact_weights = tuple(Fraction(weight) for weight in weights)
    validated_tasks: list[tuple[str, int | float, Fraction]] = []
    seen_task_ids: set[str] = set()
    for task_id, cost in tasks:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task IDs must be non-empty strings")
        if task_id in seen_task_ids:
            raise ValueError(f"task IDs must be unique; received {task_id!r} more than once")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or (isinstance(cost, float) and not math.isfinite(cost))
            or cost < 0
        ):
            raise ValueError("task costs must be non-negative finite numbers")
        seen_task_ids.add(task_id)
        validated_tasks.append((task_id, cost, Fraction(cost)))

    loads = [Fraction() for _ in exact_weights]
    assignments: list[WeightedTaskAssignment] = []
    for task_id, cost, exact_cost in sorted(
        validated_tasks,
        key=lambda task: (-task[2], task[0]),
    ):
        owner = min(
            range(len(exact_weights)),
            key=lambda rank: (
                (loads[rank] + exact_cost) / exact_weights[rank],
                rank,
            ),
        )
        loads[owner] += exact_cost
        assignments.append(WeightedTaskAssignment(task_id=task_id, cost=cost, rank=owner))
    return tuple(sorted(assignments, key=lambda assignment: assignment.task_id))


def weighted_partition_counts(
    total_count: int,
    rank_weights: Sequence[int | float],
    *,
    require_nonempty: bool = False,
) -> tuple[int, ...]:
    """Apportion an integer total by largest remainders, breaking ties by later rank."""
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
        raise ValueError("total_count must be a non-negative integer")
    weights = _validated_rank_weights(rank_weights)
    if require_nonempty and total_count < len(weights):
        raise ValueError(
            "total_count must provide at least one item per rank when require_nonempty=True"
        )
    if total_count == 0:
        return (0,) * len(weights)

    exact_weights = tuple(Fraction(weight) for weight in weights)
    weight_sum = sum(exact_weights, start=Fraction())
    quotas = tuple(total_count * weight / weight_sum for weight in exact_weights)
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: (quotas[index] - counts[index], index),
        reverse=True,
    )
    for index in remainder_order[: total_count - sum(counts)]:
        counts[index] += 1

    if require_nonempty:
        for empty_index in (index for index, count in enumerate(counts) if count == 0):
            donor = max(
                (index for index, count in enumerate(counts) if count > 1),
                key=lambda index: (counts[index] - quotas[index], counts[index], -index),
            )
            counts[donor] -= 1
            counts[empty_index] = 1
    return tuple(counts)


def weighted_partition_indices(
    total_count: int,
    rank: int,
    rank_weights: Sequence[int | float],
    *,
    require_nonempty: bool = False,
) -> range:
    """Return one rank's contiguous portion of a weighted integer partition."""
    counts = weighted_partition_counts(
        total_count,
        rank_weights,
        require_nonempty=require_nonempty,
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < len(counts):
        raise ValueError(f"rank must be between 0 and {len(counts) - 1}")
    start = sum(counts[:rank])
    return range(start, start + counts[rank])


class WeightedDistributedBatchSampler(Sampler[list[int]]):
    """Yield full weighted accumulation windows from opaque dataset indices."""

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        global_microbatches_per_step: int,
        rank: int,
        rank_weights: Sequence[int | float],
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size < 0:
            raise ValueError("dataset_size must be a non-negative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(global_microbatches_per_step, bool)
            or not isinstance(global_microbatches_per_step, int)
            or global_microbatches_per_step < 1
        ):
            raise ValueError("global_microbatches_per_step must be a positive integer")
        weights = _validated_rank_weights(rank_weights)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < len(weights):
            raise ValueError(f"rank must be between 0 and {len(weights) - 1}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if shuffle and not _TORCH_MIN_SEED <= seed <= _TORCH_MAX_SEED:
            raise ValueError(
                "seed must be between -(2**63) and 2**64 - 1 when shuffling"
            )

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.global_microbatches_per_step = global_microbatches_per_step
        self.rank = rank
        self.rank_weights = weights
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.local_microbatch_counts = weighted_partition_counts(
            global_microbatches_per_step,
            weights,
            require_nonempty=True,
        )
        self.local_microbatch_count = self.local_microbatch_counts[rank]
        self.global_samples_per_window = global_microbatches_per_step * batch_size
        self.window_count = dataset_size // self.global_samples_per_window

    def __iter__(self) -> Iterator[list[int]]:
        """Yield rank-local microbatches for every complete global window."""
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        usable_sample_count = self.window_count * self.global_samples_per_window
        microbatch_start = sum(self.local_microbatch_counts[: self.rank])
        microbatch_stop = microbatch_start + self.local_microbatch_count
        for window_start in range(0, usable_sample_count, self.global_samples_per_window):
            window_indices = indices[window_start : window_start + self.global_samples_per_window]
            for microbatch_index in range(microbatch_start, microbatch_stop):
                sample_start = microbatch_index * self.batch_size
                yield window_indices[sample_start : sample_start + self.batch_size]

    def __len__(self) -> int:
        """Return this rank's local microbatch count for one epoch."""
        return self.window_count * self.local_microbatch_count

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle seed for an epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if self.shuffle and not _TORCH_MIN_SEED <= self.seed + epoch <= _TORCH_MAX_SEED:
            raise ValueError("seed + epoch must remain within Torch's supported seed range")
        self.epoch = epoch


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Describe one rank's repeated local microbatch window."""

    local_microbatches_per_step: int
    loss_scale: float
    incomplete_window: IncompleteWindow = "step"
    window_loss_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.local_microbatches_per_step, bool)
            or not isinstance(self.local_microbatches_per_step, int)
            or self.local_microbatches_per_step < 1
        ):
            raise ValueError("local_microbatches_per_step must be a positive integer")
        if (
            isinstance(self.loss_scale, bool)
            or not isinstance(self.loss_scale, int | float)
            or not math.isfinite(self.loss_scale)
            or self.loss_scale <= 0
        ):
            raise ValueError("loss_scale must be positive and finite")
        if self.incomplete_window not in {"step", "error"}:
            raise ValueError(f"Unsupported incomplete-window policy: {self.incomplete_window!r}")
        if self.window_loss_scales is not None:
            for scale in self.window_loss_scales:
                if (
                    isinstance(scale, bool)
                    or not isinstance(scale, int | float)
                    or not math.isfinite(scale)
                    or scale <= 0
                ):
                    raise ValueError("window_loss_scales must be positive and finite")

    def window_sizes(self, local_batch_count: int) -> tuple[int, ...]:
        """Return local microbatch counts for every optimizer step this epoch."""
        if (
            isinstance(local_batch_count, bool)
            or not isinstance(local_batch_count, int)
            or local_batch_count < 0
        ):
            raise ValueError("local_batch_count must be a non-negative integer")
        full_windows, remainder = divmod(
            local_batch_count,
            self.local_microbatches_per_step,
        )
        if remainder and self.incomplete_window == "error":
            raise ValueError(
                "local batch count does not fill the configured accumulation windows"
            )
        sizes = [self.local_microbatches_per_step] * full_windows
        if remainder:
            sizes.append(remainder)
        if self.window_loss_scales is not None and len(self.window_loss_scales) != len(
            sizes
        ):
            raise ValueError(
                "window_loss_scales must contain one scale per optimizer step"
            )
        return tuple(sizes)

    def scale_for_window(
        self,
        local_microbatch_count: int,
        *,
        window_index: int,
    ) -> float:
        """Return the explicit full or incomplete logical-window loss scale."""
        if not 1 <= local_microbatch_count <= self.local_microbatches_per_step:
            raise ValueError("window microbatch count is outside the accumulation plan")
        if window_index < 0:
            raise ValueError("window_index must be non-negative")
        if self.window_loss_scales is not None:
            try:
                return float(self.window_loss_scales[window_index])
            except IndexError as error:
                raise ValueError("window_index is outside window_loss_scales") from error
        if local_microbatch_count == self.local_microbatches_per_step:
            return float(self.loss_scale)
        raise ValueError("incomplete windows require explicit window_loss_scales")


class AccumulationPolicy(Protocol):
    """Build one rank-local plan without assigning project workload meaning."""

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        """Return the plan for one epoch on the calling rank."""
        ...


@dataclass(frozen=True, slots=True)
class UniformAccumulationPolicy:
    """Use the same accumulation window on every rank."""

    microbatches_per_step: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.microbatches_per_step, bool)
            or not isinstance(self.microbatches_per_step, int)
            or self.microbatches_per_step < 1
        ):
            raise ValueError("microbatches_per_step must be a positive integer")
        AccumulationPlan(
            local_microbatches_per_step=self.microbatches_per_step,
            loss_scale=1.0 / self.microbatches_per_step,
        )

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        """Return a conventional mean-loss accumulation plan."""
        del rank, world_size
        full_windows, remainder = divmod(local_batch_count, self.microbatches_per_step)
        window_loss_scales = None
        if remainder:
            window_loss_scales = (
                *((1.0 / self.microbatches_per_step,) * full_windows),
                1.0 / remainder,
            )
        return AccumulationPlan(
            local_microbatches_per_step=self.microbatches_per_step,
            loss_scale=1.0 / self.microbatches_per_step,
            window_loss_scales=window_loss_scales,
        )


@dataclass(frozen=True, slots=True)
class WeightedAccumulationPolicy:
    """Scale caller-weighted rank-local work into one global mean-loss window."""

    global_microbatches_per_step: int
    rank_weights: Sequence[int | float]
    partial_window: PartialWindow = "error"

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_microbatches_per_step, bool)
            or not isinstance(self.global_microbatches_per_step, int)
            or self.global_microbatches_per_step < 1
        ):
            raise ValueError("global_microbatches_per_step must be a positive integer")
        weights = _validated_rank_weights(self.rank_weights)
        weighted_partition_counts(
            self.global_microbatches_per_step,
            weights,
            require_nonempty=True,
        )
        if self.partial_window not in {"error", "fixed"}:
            raise ValueError("partial_window must be 'error' or 'fixed'")
        object.__setattr__(self, "rank_weights", weights)

    def plan(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_count: int,
    ) -> AccumulationPlan:
        """Return one rank's weighted plan for a shared optimizer window."""
        if world_size != len(self.rank_weights):
            raise ValueError(
                f"world_size={world_size} does not match {len(self.rank_weights)} rank weights"
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be between 0 and {world_size - 1}")
        if (
            isinstance(local_batch_count, bool)
            or not isinstance(local_batch_count, int)
            or local_batch_count < 0
        ):
            raise ValueError("local_batch_count must be a non-negative integer")
        local_microbatches = weighted_partition_counts(
            self.global_microbatches_per_step,
            self.rank_weights,
            require_nonempty=True,
        )[rank]
        loss_scale = world_size / self.global_microbatches_per_step
        full_windows, remainder = divmod(local_batch_count, local_microbatches)
        window_loss_scales = None
        incomplete_window: IncompleteWindow = "error"
        if remainder and self.partial_window == "fixed":
            window_loss_scales = (loss_scale,) * (full_windows + 1)
            incomplete_window = "step"
        return AccumulationPlan(
            local_microbatches_per_step=local_microbatches,
            loss_scale=loss_scale,
            incomplete_window=incomplete_window,
            window_loss_scales=window_loss_scales,
        )
