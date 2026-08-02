"""Recursive device transfer for ordinary nested PyTorch batches.

The trainer calls this default mover unless a consuming project injects its
own function for custom batch containers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


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
        return [
            move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch
        ]
    return batch
