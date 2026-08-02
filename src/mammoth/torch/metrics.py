"""Project-named scalar aggregation policies for the optional trainer.

Step functions compute metrics; this module only performs mean, sum, or last
retention and optional standard distributed reduction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed

Reduction = Literal["mean", "sum", "last"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Aggregation policy for one opaque project metric name."""

    reduction: Reduction = "mean"
    distributed: bool = True

    def __post_init__(self) -> None:
        if self.reduction not in {"mean", "sum", "last"}:
            raise ValueError(f"Unsupported metric reduction: {self.reduction!r}")


@dataclass(slots=True)
class MetricValue:
    """Mutable accumulator state for one metric."""

    total: float = 0.0
    weight: float = 0.0
    last: float = 0.0
    seen: bool = False


class MetricAccumulator:
    """Aggregate scalar metrics according to explicit per-name policies."""

    def __init__(
        self,
        specs: Mapping[str, MetricSpec] | None = None,
        *,
        default: MetricSpec | None = None,
    ) -> None:
        self.specs = dict(specs or {})
        self.default = default or MetricSpec()
        self.values: dict[str, MetricValue] = {}

    def update(self, metrics: Mapping[str, float], *, weight: float = 1.0) -> None:
        """Accumulate one step's already-computed finite scalar metrics."""
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("metric weight must be positive and finite")
        for name, raw_value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int | float)
                or not math.isfinite(raw_value)
            ):
                raise ValueError(f"metric {name!r} must be finite")
            value = float(raw_value)
            state = self.values.setdefault(name, MetricValue())
            state.total += value * weight
            state.weight += weight
            state.last = value
            state.seen = True

    def compute(
        self,
        *,
        device: torch.device | None = None,
        distributed: bool = False,
    ) -> dict[str, float]:
        """Return reduced metrics, optionally combining initialized DDP ranks."""
        results: dict[str, float] = {}
        use_distributed = distributed and torch.distributed.is_initialized()
        for name in sorted(self.values):
            state = self.values[name]
            spec = self.specs.get(name, self.default)
            total, weight, last = state.total, state.weight, state.last
            if use_distributed and spec.distributed:
                reduction_device = device or torch.device("cpu")
                if spec.reduction == "last":
                    tensor = torch.tensor(last, dtype=torch.float64, device=reduction_device)
                    torch.distributed.broadcast(tensor, src=0)
                    last = float(tensor.item())
                else:
                    tensor = torch.tensor(
                        [total, weight], dtype=torch.float64, device=reduction_device
                    )
                    torch.distributed.all_reduce(tensor)
                    total, weight = (float(value) for value in tensor.tolist())
            if spec.reduction == "mean":
                results[name] = total / weight
            elif spec.reduction == "sum":
                results[name] = total
            else:
                results[name] = last
        return results
