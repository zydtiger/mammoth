"""Optional generic PyTorch runtime, trainer, profiler, and state utilities.

Install the ``torch`` extra before importing this package.
"""

from __future__ import annotations

from mammoth.torch.batch import move_batch_to_device
from mammoth.torch.callbacks import Callback, EarlyStopping
from mammoth.torch.checkpoint import (
    AsyncCheckpointPublisher,
    CheckpointArtifact,
    CheckpointPlan,
    CheckpointPublication,
    StateRegistry,
    TrainerCheckpointContext,
    TrainerCheckpointPolicy,
    checkpoint_payload,
    publish_checkpoint_plan,
    restore_checkpoint,
)
from mammoth.torch.metrics import (
    MetricAccumulator,
    MetricRoute,
    MetricSpec,
    StatefulMetric,
)
from mammoth.torch.profiling import (
    CudaMemoryStats,
    LatencySummary,
    OperationProfile,
    ProfileConfig,
    ProfileReport,
    ProfileTiming,
    ThroughputSummary,
    TorchRuntimeOptions,
    TorchRuntimeState,
    current_torch_runtime_state,
    profile_callable,
    summarize_latency,
    summarize_output_value,
    torch_runtime_options,
    write_profile_report,
)
from mammoth.torch.runtime import (
    TorchExecutionRequest,
    TorchExecutionRuntime,
    TorchRuntimeConfig,
    initialize_torch_runtime,
)
from mammoth.torch.scheduling import (
    AccumulationPlan,
    AccumulationPolicy,
    UniformAccumulationPolicy,
)
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
    "AccumulationPlan",
    "AccumulationPolicy",
    "Callback",
    "CheckpointArtifact",
    "CheckpointPlan",
    "CheckpointPublication",
    "CudaMemoryStats",
    "EarlyStopping",
    "LatencySummary",
    "MetricAccumulator",
    "MetricRoute",
    "MetricSpec",
    "OperationProfile",
    "ProfileConfig",
    "ProfileReport",
    "ProfileTiming",
    "StateRegistry",
    "StatefulMetric",
    "StepContext",
    "StepOutput",
    "Trainer",
    "TrainerConfig",
    "TrainerCheckpointContext",
    "TrainerCheckpointPolicy",
    "TrainerResult",
    "TrainerState",
    "TorchExecutionRequest",
    "TorchExecutionRuntime",
    "TorchRuntimeConfig",
    "TorchRuntimeOptions",
    "TorchRuntimeState",
    "ThroughputSummary",
    "UniformAccumulationPolicy",
    "checkpoint_payload",
    "current_torch_runtime_state",
    "initialize_torch_runtime",
    "move_batch_to_device",
    "publish_checkpoint_plan",
    "profile_callable",
    "restore_checkpoint",
    "summarize_latency",
    "summarize_output_value",
    "torch_runtime_options",
    "write_profile_report",
]
