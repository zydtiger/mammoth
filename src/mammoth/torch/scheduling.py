"""Project-neutral gradient-accumulation plans for the generic trainer.

The trainer calls a consumer-supplied policy once per epoch. Consuming projects
may vary local work by rank without moving their scheduling rules into Mammoth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

IncompleteWindow = Literal["step", "error"]


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
