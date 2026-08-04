"""Serializable framework-neutral coordinates for the optional trainer.

The trainer and callbacks share this value; checkpoint registry publication
persists it alongside caller-provided stateful objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainerState:
    """Mutable ordinary-loop coordinates restored from Mammoth checkpoints.

    ``global_step`` counts microbatches across all ranks at completed optimizer
    windows, so checkpointed DDP coordinates remain rank-invariant.
    """

    epoch: int = -1
    global_step: int = 0
    optimizer_step: int = 0
    stopped_early: bool = False

    def state_dict(self) -> dict[str, int | bool]:
        """Return JSON-like checkpoint state."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optimizer_step": self.optimizer_step,
            "stopped_early": self.stopped_early,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Validate and restore checkpoint coordinates."""
        epoch = integer_field(payload, "epoch", minimum=-1)
        global_step = integer_field(payload, "global_step", minimum=0)
        optimizer_step = integer_field(payload, "optimizer_step", minimum=0)
        stopped_early = payload.get("stopped_early")
        if not isinstance(stopped_early, bool):
            raise ValueError("trainer stopped_early must be a boolean")
        self.epoch = epoch
        self.global_step = global_step
        self.optimizer_step = optimizer_step
        self.stopped_early = stopped_early


def integer_field(payload: Mapping[str, Any], name: str, *, minimum: int) -> int:
    """Read one bounded integer checkpoint field."""
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"trainer {name} must be an integer >= {minimum}")
    return value
