"""Optional generic PyTorch trainer, metrics, callbacks, and checkpoints.

Install the ``torch`` extra before importing this package.
"""

from __future__ import annotations

from mammoth.torch.batch import move_batch_to_device
from mammoth.torch.callbacks import Callback, EarlyStopping
from mammoth.torch.checkpoint import (
    AsyncCheckpointPublisher,
    StateRegistry,
    checkpoint_payload,
    restore_checkpoint,
)
from mammoth.torch.metrics import MetricAccumulator, MetricSpec
from mammoth.torch.state import TrainerState
from mammoth.torch.trainer import (
    StepContext,
    StepOutput,
    Trainer,
    TrainerConfig,
    TrainerResult,
)

__all__ = [
    "AsyncCheckpointPublisher",
    "Callback",
    "EarlyStopping",
    "MetricAccumulator",
    "MetricSpec",
    "StateRegistry",
    "StepContext",
    "StepOutput",
    "Trainer",
    "TrainerConfig",
    "TrainerResult",
    "TrainerState",
    "checkpoint_payload",
    "move_batch_to_device",
    "restore_checkpoint",
]
