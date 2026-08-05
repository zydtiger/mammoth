"""Generic trainer callback hooks and validation-based early stopping.

Callbacks receive only trainer coordinates and project-named scalar summaries;
they never depend on a model architecture or dataset implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from mammoth.torch.state import TrainerState


class Callback:
    """No-op base class for ordinary trainer lifecycle customization."""

    def on_train_start(self, state: TrainerState) -> None:
        """Run before the first epoch."""

    def on_epoch_end(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        """Run after one training epoch."""

    def on_validation_end(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        """Run after one validation epoch."""

    def on_train_end(self, state: TrainerState) -> None:
        """Run after normal or early completion."""

    def should_stop(self, state: TrainerState) -> bool:
        """Return whether training should stop after the current epoch."""
        return False

    def state_dict(self) -> dict[str, Any]:
        """Return callback checkpoint state."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore callback checkpoint state."""


class EarlyStopping(Callback):
    """Stop after a project metric fails to improve for ``patience`` checks."""

    def __init__(
        self,
        metric: str,
        *,
        mode: Literal["min", "max"] = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        if not metric:
            raise ValueError("early-stopping metric must be non-empty")
        if mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be 'min' or 'max'")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("early-stopping patience must be a non-negative integer")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("early-stopping min_delta must be finite and non-negative")
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_checks = 0
        self.improved = False
        self._checked = False

    def on_epoch_end(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        """Clear the transient validation result before the next check."""
        self.improved = False
        self._checked = False

    def on_validation_end(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        """Update improvement state from one validation summary."""
        if self.metric not in metrics:
            raise KeyError(f"Early-stopping metric {self.metric!r} was not reported")
        value = metrics[self.metric]
        self._checked = True
        self.improved = self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if self.improved:
            self.best = value
            self.bad_checks = 0
            return
        self.bad_checks += 1

    def should_stop(self, state: TrainerState) -> bool:
        """Stop on the ``patience``-th consecutive non-improving check."""
        return self._checked and not self.improved and self.bad_checks >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """Return early-stopping checkpoint state."""
        return {
            "best": self.best,
            "bad_checks": self.bad_checks,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore early-stopping checkpoint state."""
        best = state.get("best")
        if best is not None and (
            isinstance(best, bool) or not isinstance(best, int | float) or not math.isfinite(best)
        ):
            raise ValueError("early-stopping best must be finite or null")
        bad_checks = state.get("bad_checks")
        if not isinstance(bad_checks, int) or isinstance(bad_checks, bool) or bad_checks < 0:
            raise ValueError("early-stopping bad_checks must be non-negative")
        stopped = state.get("stopped")
        if stopped is not None and not isinstance(stopped, bool):
            raise ValueError("early-stopping stopped must be boolean")
        self.best = None if best is None else float(best)
        self.bad_checks = bad_checks
        self.improved = False
        self._checked = False
