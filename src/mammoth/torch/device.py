"""Resolve project-selected PyTorch device strings without owning runtime state.

The execution runtime and generic Trainer both use this helper before applying
their separate process-group and model-lifecycle policies.
"""

from __future__ import annotations

import torch


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` or one explicit torch device string."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device
