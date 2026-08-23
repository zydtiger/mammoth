"""Recursive device transfer for ordinary nested PyTorch batches.

The trainer calls this default mover unless a consuming project injects its
own function for custom batch containers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Final

import torch

type BatchMover = Callable[[Any, torch.device], Any]
_EXHAUSTED: Final = object()


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    """Move tensors in mappings, tuples, and lists while retaining structure."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(
            *(move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch)
        )
    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch
        )
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch]
    return batch


class CudaPrefetchingBatchIterator(Iterator[Any]):
    """Move eligible batches on one CUDA copy stream ahead of their consumption.

    The iterator owns at most one device-resident prefetched batch and one raw
    batch that was not eligible for asynchronous transfer.  It deliberately
    leaves caller-supplied movers on the synchronous path because their stream
    and ownership behavior is project-defined.
    """

    def __init__(
        self,
        batches: Iterator[Any],
        device: torch.device,
        mover: BatchMover,
        *,
        enabled: bool,
        prefetch_mover: BatchMover | None = None,
    ) -> None:
        self._batches = batches
        self._device = device
        self._mover = mover
        self._prefetch_mover = prefetch_mover or mover
        self._stream = (
            torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            if enabled and device.type == "cuda"
            else None
        )
        self._prefetched: Any = _EXHAUSTED
        self._pending_raw: Any = _EXHAUSTED
        self._exhausted = False

    def __next__(self) -> Any:
        """Return one moved batch and stage at most one following batch."""
        if self._prefetched is not _EXHAUSTED:
            moved = self._prefetched
            self._prefetched = _EXHAUSTED
            self._wait_for_prefetched_batch(moved)
        else:
            raw = self._next_raw_batch()
            moved = self._mover(raw, self._device)
        if self._stream is not None:
            self._stage_following_batch()
        return moved

    def close(self) -> None:
        """Release retained batches and the owned stream without synchronizing CUDA."""
        self._prefetched = _EXHAUSTED
        self._pending_raw = _EXHAUSTED
        self._stream = None

    def _next_raw_batch(self) -> Any:
        if self._pending_raw is not _EXHAUSTED:
            raw = self._pending_raw
            self._pending_raw = _EXHAUSTED
            return raw
        if self._exhausted:
            raise StopIteration
        try:
            return next(self._batches)
        except StopIteration:
            self._exhausted = True
            raise

    def _stage_following_batch(self) -> None:
        if self._stream is None or self._exhausted or self._prefetched is not _EXHAUSTED:
            return
        try:
            raw = next(self._batches)
        except StopIteration:
            self._exhausted = True
            return
        if not batch_is_cuda_prefetch_eligible(raw, self._device):
            self._pending_raw = raw
            return
        with torch.cuda.stream(self._stream):
            self._prefetched = self._prefetch_mover(raw, self._device)

    def _wait_for_prefetched_batch(self, batch: Any) -> None:
        stream = self._stream
        if stream is None:
            raise RuntimeError("CUDA prefetch stream was released before batch consumption")
        compute_stream = torch.cuda.current_stream(self._device)
        compute_stream.wait_stream(stream)
        record_batch_stream(batch, compute_stream)


def batch_is_cuda_prefetch_eligible(batch: Any, device: torch.device) -> bool:
    """Return whether all CPU tensor leaves are pinned for safe asynchronous H2D copy."""
    if device.type != "cuda":
        return False
    has_cpu_tensor = False

    def inspect(value: Any) -> bool:
        nonlocal has_cpu_tensor
        if isinstance(value, torch.Tensor):
            if value.device.type == "cpu":
                has_cpu_tensor = True
                return value.is_pinned()
            return value.device == device
        if isinstance(value, Mapping):
            return all(inspect(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(inspect(item) for item in value)
        return True

    supported = inspect(batch)
    return has_cpu_tensor and supported


def record_batch_stream(batch: Any, stream: torch.cuda.Stream) -> None:
    """Keep CUDA tensor storage alive until ``stream`` has consumed the batch."""
    if isinstance(batch, torch.Tensor):
        if batch.device.type == "cuda":
            batch.record_stream(stream)
        return
    if isinstance(batch, Mapping):
        for value in batch.values():
            record_batch_stream(value, stream)
        return
    if isinstance(batch, (tuple, list)):
        for value in batch:
            record_batch_stream(value, stream)
