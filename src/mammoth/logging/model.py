"""Sink-neutral observations passed from ``RunObserver`` to logging backends.

The observer constructs these immutable values. JSONL and TensorBoard sinks
retain different subsets without coupling computation code to either backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from mammoth.core.events import EventName

MediaKind = Literal["image", "text", "histogram"]


@dataclass(frozen=True, slots=True)
class Media:
    """One optional TensorBoard media value with backend keyword options."""

    kind: MediaKind
    value: Any
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"image", "text", "histogram"}:
            raise ValueError(f"Unsupported media kind: {self.kind!r}")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class Observation:
    """A lifecycle record plus optional dense metrics and media."""

    event: EventName
    fields: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    display_metrics: Mapping[str, float] = field(default_factory=dict)
    media: Mapping[str, Media] = field(default_factory=dict)
    logical_step: int | None = None

    def __post_init__(self) -> None:
        if self.logical_step is not None and (
            isinstance(self.logical_step, bool)
            or not isinstance(self.logical_step, int)
            or self.logical_step < 0
        ):
            raise ValueError("logical_step must be a non-negative integer or None")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "metrics", MappingProxyType(validate_metrics(self.metrics)))
        object.__setattr__(
            self,
            "display_metrics",
            MappingProxyType(validate_metrics(self.display_metrics)),
        )
        if not isinstance(self.media, Mapping):
            raise ValueError("media must be a mapping")
        validated_media: dict[str, Media] = {}
        for name, value in self.media.items():
            if not isinstance(name, str) or not name:
                raise ValueError("media names must be non-empty strings")
            if not isinstance(value, Media):
                raise ValueError(f"media[{name!r}] must be a Media value")
            validated_media[name] = value
        object.__setattr__(self, "media", MappingProxyType(validated_media))


def validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return finite scalar metrics without assigning domain semantics."""
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"metric {name!r} must be finite")
        validated[name] = float(value)
    return validated
