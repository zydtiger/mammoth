"""Optional TensorBoard dense-history sink for scalar and media observations.

This module belongs to the ``tensorboard`` extra. It is intentionally absent
from ``mammoth.logging`` root imports so core and JSONL users need no backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from tensorboardX import SummaryWriter  # type: ignore[import-untyped]

from mammoth.logging.model import Media, Observation


class SummaryWriterLike(Protocol):
    """Small TensorBoard writer surface used by :class:`TensorBoardSink`."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Write a scalar."""
        ...

    def add_image(self, tag: str, img_tensor: Any, global_step: int, **kwargs: Any) -> None:
        """Write an image."""
        ...

    def add_text(self, tag: str, text_string: str, global_step: int, **kwargs: Any) -> None:
        """Write text."""
        ...

    def add_histogram(self, tag: str, values: Any, global_step: int, **kwargs: Any) -> None:
        """Write a histogram."""
        ...

    def flush(self) -> None:
        """Flush buffered records."""
        ...

    def close(self) -> None:
        """Close the writer."""
        ...


class NullSummaryWriter:
    """No-op writer used by non-primary ranks."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Discard a scalar."""
        pass

    def add_image(self, tag: str, img_tensor: Any, global_step: int, **kwargs: Any) -> None:
        """Discard an image."""
        pass

    def add_text(self, tag: str, text_string: str, global_step: int, **kwargs: Any) -> None:
        """Discard text."""
        pass

    def add_histogram(self, tag: str, values: Any, global_step: int, **kwargs: Any) -> None:
        """Discard a histogram."""
        pass

    def flush(self) -> None:
        """Do nothing."""
        pass

    def close(self) -> None:
        """Do nothing."""
        pass


class TensorBoardSink:
    """Retain dense metrics and explicit media on a rank-aware logical clock."""

    def __init__(
        self,
        log_dir: Path,
        *,
        rank: int = 0,
        primary_rank: int = 0,
        enabled_on_secondary: bool = False,
        flush_seconds: int = 10,
        writer: SummaryWriterLike | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.rank = rank
        self.primary_rank = primary_rank
        self.enabled = rank == primary_rank or enabled_on_secondary
        self._step = 0
        if writer is not None:
            self.writer = writer
        elif self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(logdir=str(self.log_dir), flush_secs=flush_seconds)
        else:
            self.writer = NullSummaryWriter()

    def observe(self, observation: Observation) -> None:
        """Write dense scalar and media history while ignoring lifecycle-only events."""
        if not self.enabled or (not observation.metrics and not observation.media):
            return
        step = self._logical_step(observation.fields.get("coordinates"))
        for name, value in observation.metrics.items():
            self.writer.add_scalar(name, value, step)
        for name, media in observation.media.items():
            self._write_media(name, media, step)

    def flush(self) -> None:
        """Flush TensorBoard's event writer."""
        self.writer.flush()

    def close(self) -> None:
        """Flush and close TensorBoard's event writer."""
        self.writer.close()

    def _logical_step(self, coordinates: object) -> int:
        if isinstance(coordinates, Mapping):
            for name in ("global_step", "optimizer_step", "step", "epoch", "batch"):
                value = coordinates.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    self._step = max(self._step, value + 1)
                    return value
        step = self._step
        self._step += 1
        return step

    def _write_media(self, name: str, media: Media, step: int) -> None:
        options = dict(media.options)
        if media.kind == "image":
            self.writer.add_image(name, media.value, step, **options)
        elif media.kind == "text":
            self.writer.add_text(name, str(media.value), step, **options)
        else:
            self.writer.add_histogram(name, media.value, step, **options)
