"""Project-named scalar aggregation policies for the optional trainer.

Step functions compute metrics; this module only performs mean, sum, or last
retention and optional standard distributed reduction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.distributed

Reduction = Literal["mean", "sum", "last"]
MetricCadence = Literal["batch", "epoch"]
MetricScalar = float | torch.Tensor


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Aggregation policy for one opaque project metric name."""

    reduction: Reduction = "mean"
    distributed: bool = True

    def __post_init__(self) -> None:
        if self.reduction not in {"mean", "sum", "last"}:
            raise ValueError(f"Unsupported metric reduction: {self.reduction!r}")


@dataclass(frozen=True, slots=True)
class MetricRoute:
    """Map one opaque metric to optional batch and epoch sink names."""

    batch_name: str | None
    epoch_name: str | None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("batch_name", self.batch_name),
            ("epoch_name", self.epoch_name),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or None")
        if self.batch_name is not None and self.batch_name == self.epoch_name:
            raise ValueError("batch_name and epoch_name must use distinct sink names")


class StatefulMetric(Protocol):
    """Accumulate project values into additive tensors computed by Mammoth."""

    def reset(self) -> None:
        """Clear state before one trainer epoch."""
        ...

    def update(self, value: Any) -> None:
        """Consume one project-defined transient update value."""
        ...

    def state_tensors(self) -> Mapping[str, torch.Tensor]:
        """Return additive tensor state without transferring ownership."""
        ...

    def compute(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, float | torch.Tensor]:
        """Compute project-named scalars from local or globally summed state."""
        ...


@dataclass(slots=True)
class MetricValue:
    """Mutable accumulator state for one metric."""

    total: MetricScalar = 0.0
    weight: float = 0.0
    last: MetricScalar = 0.0
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

    def update(
        self,
        metrics: Mapping[str, float | torch.Tensor],
        *,
        weight: float = 1.0,
    ) -> None:
        """Accumulate detached scalar metrics without forcing host materialization."""
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("metric weight must be positive and finite")
        for name, value in prepared_scalar_metrics(metrics).items():
            state = self.values.setdefault(name, MetricValue())
            state.total = _add_metric_scalars(state.total, _scale_metric_scalar(value, weight))
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
        results: dict[str, MetricScalar] = {}
        use_distributed = distributed and torch.distributed.is_initialized()
        local_names = sorted(self.values)
        names = local_names
        if use_distributed:
            gathered_names: list[list[str]] = [
                list() for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(gathered_names, local_names)
            union_names = sorted(
                {name for rank_names in gathered_names for name in rank_names}
            )
            local_specs = {
                name: (
                    self.specs.get(name, self.default).reduction,
                    self.specs.get(name, self.default).distributed,
                )
                for name in union_names
            }
            gathered_specs: list[dict[str, tuple[Reduction, bool]]] = [
                {} for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(gathered_specs, local_specs)
            if any(specs != local_specs for specs in gathered_specs):
                raise ValueError("metric reduction specifications differ across ranks")
            distributed_names = [
                name for name in union_names if self.specs.get(name, self.default).distributed
            ]
            local_only_names = [
                name
                for name in local_names
                if not self.specs.get(name, self.default).distributed
            ]
            names = distributed_names + local_only_names
        for name in names:
            state = self.values.get(name, MetricValue())
            spec = self.specs.get(name, self.default)
            total = state.total
            reduced_weight: MetricScalar = state.weight
            last = state.last
            if use_distributed and spec.distributed:
                reduction_device = device or torch.device("cpu")
                if spec.reduction == "last":
                    if name not in gathered_names[0]:
                        raise ValueError(
                            f"distributed last metric {name!r} was not reported on rank 0"
                        )
                    tensor = _distributed_metric_tensor(last, reduction_device)
                    torch.distributed.broadcast(tensor, src=0)
                    last = tensor
                else:
                    total_tensor = _distributed_metric_tensor(total, reduction_device)
                    tensor = torch.stack(
                        (
                            total_tensor,
                            torch.tensor(
                                state.weight,
                                dtype=torch.float64,
                                device=reduction_device,
                            ),
                        )
                    )
                    torch.distributed.all_reduce(tensor)
                    total, reduced_weight = tensor[0], tensor[1]
            if spec.reduction == "mean":
                if not isinstance(reduced_weight, torch.Tensor) and reduced_weight <= 0:
                    raise ValueError(f"metric {name!r} has no globally reduced weight")
                results[name] = total / reduced_weight
            elif spec.reduction == "sum":
                results[name] = total
            else:
                results[name] = last
        return scalar_metrics(results)


def reset_stateful_metrics(metrics: Mapping[str, StatefulMetric]) -> None:
    """Reset every registered stateful metric in stable name order."""
    for name in sorted(metrics):
        validate_metric_name(name)
        metrics[name].reset()


def update_stateful_metrics(
    metrics: Mapping[str, StatefulMetric],
    updates: Mapping[str, Any],
) -> None:
    """Route transient step values to registered project metrics."""
    for name, value in updates.items():
        validate_metric_name(name)
        if name not in metrics:
            raise KeyError(f"step updated unregistered stateful metric {name!r}")
        metrics[name].update(value)


def compute_stateful_metrics(
    metrics: Mapping[str, StatefulMetric],
    *,
    device: torch.device,
    distributed: bool,
    baseline: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """Sum complete or since-baseline additive state and compute finite scalars."""
    use_distributed = distributed and torch.distributed.is_initialized()
    metric_names = sorted(metrics)
    snapshot: dict[str, dict[str, torch.Tensor]] | None = None
    preparation_error: BaseException | None = None
    try:
        snapshot = snapshot_stateful_metrics(metrics)
        if baseline is not None and sorted(baseline) != metric_names:
            raise ValueError("stateful metric baseline registrations do not match")
        if baseline is not None:
            for metric_name in metric_names:
                if sorted(baseline[metric_name]) != sorted(snapshot[metric_name]):
                    raise ValueError(
                        f"stateful metric {metric_name!r} baseline state does not match"
                    )
    except BaseException as error:
        preparation_error = error
    raise_stateful_distributed_failure(
        "preparation",
        preparation_error,
        use_distributed=use_distributed,
    )
    assert snapshot is not None
    if use_distributed:
        gathered_metric_names: list[list[str]] = [
            list() for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered_metric_names, metric_names)
        if any(names != metric_names for names in gathered_metric_names):
            raise ValueError("stateful metric registrations differ across distributed ranks")
        local_metadata = stateful_metric_metadata(snapshot)
        gathered_metadata: list[
            dict[str, dict[str, tuple[tuple[int, ...], str, str]]]
        ] = [{} for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_metadata, local_metadata)
        if any(metadata != local_metadata for metadata in gathered_metadata):
            raise ValueError("stateful metric tensor metadata differs across ranks")
        if any(
            layout != str(torch.strided)
            for metric_metadata in local_metadata.values()
            for _, _, layout in metric_metadata.values()
        ):
            raise ValueError("stateful metric tensors must use strided layout")

    prepared: dict[str, dict[str, torch.Tensor]] = {}
    tensor_error: BaseException | None = None
    try:
        for metric_name in metric_names:
            state = snapshot[metric_name]
            prepared_state: dict[str, torch.Tensor] = {}
            for state_name in sorted(state):
                value = state[state_name].to(device=device)
                if baseline is not None:
                    previous = baseline[metric_name][state_name].to(device=device)
                    if previous.shape != value.shape or previous.dtype != value.dtype:
                        raise ValueError(
                            f"stateful metric {metric_name!r} baseline tensor does not match"
                        )
                    value = value - previous
                prepared_state[state_name] = value
            prepared[metric_name] = prepared_state
    except BaseException as error:
        tensor_error = error
    raise_stateful_distributed_failure(
        "tensor preparation",
        tensor_error,
        use_distributed=use_distributed,
    )

    if use_distributed:
        for metric_name in metric_names:
            for state_name in sorted(prepared[metric_name]):
                torch.distributed.all_reduce(prepared[metric_name][state_name])

    results: dict[str, float] = {}
    compute_error: BaseException | None = None
    try:
        for metric_name in metric_names:
            computed = scalar_metrics(
                metrics[metric_name].compute(prepared[metric_name])
            )
            overlap = sorted(set(results).intersection(computed))
            if overlap:
                raise ValueError(
                    f"stateful metrics returned duplicate names: {overlap}"
                )
            results.update(computed)
    except BaseException as error:
        compute_error = error
    raise_stateful_distributed_failure(
        "computation",
        compute_error,
        use_distributed=use_distributed,
    )
    return results


def raise_stateful_distributed_failure(
    operation: str,
    local_error: BaseException | None,
    *,
    use_distributed: bool,
) -> None:
    """Propagate one rank-local stateful-metric failure before more collectives."""
    if not use_distributed:
        if local_error is not None:
            raise local_error
        return
    local_status = (
        None
        if local_error is None
        else f"{type(local_error).__name__}: {local_error}"
    )
    gathered_statuses: list[str | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered_statuses, local_status)
    failure = next((status for status in gathered_statuses if status is not None), None)
    if failure is not None:
        raise RuntimeError(
            f"stateful metric {operation} failed: {failure}"
        ) from local_error


def snapshot_stateful_metrics(
    metrics: Mapping[str, StatefulMetric],
) -> dict[str, dict[str, torch.Tensor]]:
    """Clone additive local state for later logical-window delta reduction."""
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for metric_name in sorted(metrics):
        validate_metric_name(metric_name)
        state = metrics[metric_name].state_tensors()
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"stateful metric {metric_name!r} returned no tensor state")
        metric_state: dict[str, torch.Tensor] = {}
        for state_name in sorted(state):
            validate_metric_name(state_name)
            tensor = state[state_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"stateful metric {metric_name!r} state {state_name!r} must be a tensor"
                )
            metric_state[state_name] = tensor.detach().clone()
        snapshot[metric_name] = metric_state
    return snapshot


def stateful_metric_metadata(
    snapshot: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, tuple[tuple[int, ...], str, str]]]:
    """Return collective-relevant tensor metadata for rank consensus."""
    metadata: dict[str, dict[str, tuple[tuple[int, ...], str, str]]] = {}
    for metric_name, state in snapshot.items():
        metric_metadata: dict[str, tuple[tuple[int, ...], str, str]] = {}
        for state_name, tensor in state.items():
            metric_metadata[state_name] = (
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.layout),
            )
        metadata[metric_name] = metric_metadata
    return metadata


def route_metrics(
    metrics: Mapping[str, float],
    routes: Mapping[str, MetricRoute],
    cadence: MetricCadence,
) -> dict[str, float]:
    """Apply optional sink-name routing while preserving unspecified metrics."""
    routed: dict[str, float] = {}
    for name, value in metrics.items():
        route = routes.get(name)
        if route is None:
            destination = name if cadence == "batch" else None
        else:
            destination = route.batch_name if cadence == "batch" else route.epoch_name
        if destination is None:
            continue
        if destination in routed:
            raise ValueError(f"multiple metrics route to {destination!r}")
        routed[destination] = value
    return routed


def prepared_scalar_metrics(
    metrics: Mapping[str, float | torch.Tensor],
) -> dict[str, MetricScalar]:
    """Validate scalar shapes and detach tensors without materializing them on the host."""
    if not isinstance(metrics, Mapping):
        raise TypeError("computed metric values must be a mapping")
    values: dict[str, MetricScalar] = {}
    for name, raw_value in metrics.items():
        validate_metric_name(name)
        value: MetricScalar
        if isinstance(raw_value, torch.Tensor):
            if raw_value.ndim != 0:
                raise ValueError(f"metric {name!r} tensor must be scalar")
            value = raw_value.detach()
        elif isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise TypeError(f"metric {name!r} must be a number or scalar tensor")
        else:
            value = float(raw_value)
        if not isinstance(value, torch.Tensor) and not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        values[name] = value
    return values


def scalar_metrics(metrics: Mapping[str, float | torch.Tensor]) -> dict[str, float]:
    """Materialize validated scalar metrics, batching transfers by device and dtype."""
    values = prepared_scalar_metrics(metrics)
    materialized: dict[str, float] = {}
    tensors: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            tensors.setdefault((value.device, value.dtype), []).append((name, value))
        else:
            materialized[name] = value
    for grouped_values in tensors.values():
        names, tensor_values = zip(*grouped_values, strict=True)
        transferred = torch.stack(tensor_values).cpu().tolist()
        for name, value in zip(names, transferred, strict=True):
            materialized[name] = float(value)
    for name, value in materialized.items():
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
    return {name: materialized[name] for name in values}


def _scale_metric_scalar(value: MetricScalar, weight: float) -> MetricScalar:
    """Multiply one detached scalar while preserving host-accumulator precision."""
    if isinstance(value, torch.Tensor) and not value.is_complex() and value.device.type in {
        "cpu",
        "cuda",
    }:
        return value.to(dtype=torch.float64) * weight
    return value * weight


def _add_metric_scalars(left: MetricScalar, right: MetricScalar) -> MetricScalar:
    """Add scalar values, preserving tensor residency when either value is a tensor."""
    if isinstance(left, torch.Tensor):
        if isinstance(right, torch.Tensor):
            return left + right.to(device=left.device, dtype=left.dtype)
        return left + right
    if isinstance(right, torch.Tensor):
        return right + left
    return left + right


def _distributed_metric_tensor(value: MetricScalar, device: torch.device) -> torch.Tensor:
    """Return one detached float64 scalar on the collective device."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def validate_metric_name(name: object) -> None:
    """Reject empty or non-string metric and state names."""
    if not isinstance(name, str) or not name:
        raise ValueError("metric names must be non-empty strings")
