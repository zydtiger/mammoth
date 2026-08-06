"""Project-neutral process-global PyTorch backend and seed configuration.

Training, inference, and profiling entry points use this module to apply
caller-selected numerical backend settings without importing project policy.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import torch

type MatmulPrecision = Literal["highest", "high", "medium"]

_TORCH_MIN_SEED = -(2**63)
_TORCH_MAX_SEED = 2**64 - 1


@dataclass(frozen=True, slots=True)
class TorchBackendConfig:
    """Optional process-global PyTorch numerical backend settings."""

    matmul_precision: MatmulPrecision | None = None
    cuda_matmul_allow_tf32: bool | None = None
    cudnn_allow_tf32: bool | None = None
    cudnn_benchmark: bool | None = None
    cudnn_deterministic: bool | None = None
    deterministic_algorithms: bool | None = None
    deterministic_warn_only: bool = False

    def __post_init__(self) -> None:
        if self.matmul_precision not in {None, "highest", "high", "medium"}:
            raise ValueError(f"Unsupported matmul precision: {self.matmul_precision!r}")
        for name in (
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None")
        if not isinstance(self.deterministic_warn_only, bool):
            raise ValueError("deterministic_warn_only must be a boolean")
        if self.deterministic_warn_only and self.deterministic_algorithms is None:
            raise ValueError(
                "deterministic_warn_only requires deterministic_algorithms to be set"
            )


@dataclass(frozen=True, slots=True)
class TorchBackendState:
    """Capture the effective process-global PyTorch numerical backend state."""

    matmul_precision: MatmulPrecision
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool


@dataclass(frozen=True, slots=True)
class TorchSeedPolicy:
    """Select which generic random-number generators receive one integer seed."""

    seed: int
    python_random: bool = True
    torch_cpu: bool = True
    torch_cuda: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("python_random", "torch_cpu", "torch_cuda"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (self.torch_cpu or self.torch_cuda) and not (
            _TORCH_MIN_SEED <= self.seed <= _TORCH_MAX_SEED
        ):
            raise ValueError(
                "seed must be between -(2**63) and 2**64 - 1 when seeding Torch"
            )


def apply_torch_seed_policy(policy: TorchSeedPolicy) -> None:
    """Seed the caller-selected Python, Torch CPU, and available CUDA generators."""
    if not isinstance(policy, TorchSeedPolicy):
        raise TypeError("policy must be TorchSeedPolicy")
    if policy.python_random:
        random.seed(policy.seed)
    if policy.torch_cpu:
        torch.default_generator.manual_seed(policy.seed)
    if policy.torch_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy.seed)


def apply_torch_backend_config(config: TorchBackendConfig) -> TorchBackendState:
    """Apply caller-selected backend settings and return their effective state."""
    if not isinstance(config, TorchBackendConfig):
        raise TypeError("config must be TorchBackendConfig")
    previous = current_torch_backend_state()
    matmul_precision = _normalized_matmul_precision(config, previous)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    if config.cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32
    if config.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if config.cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
    if config.deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(
            config.deterministic_algorithms,
            warn_only=config.deterministic_warn_only,
        )
    return current_torch_backend_state()


@contextmanager
def configured_torch_backend(config: TorchBackendConfig) -> Iterator[TorchBackendState]:
    """Apply backend settings temporarily and restore every captured value."""
    if not isinstance(config, TorchBackendConfig):
        raise TypeError("config must be TorchBackendConfig")
    previous = current_torch_backend_state()
    try:
        yield apply_torch_backend_config(config)
    finally:
        apply_torch_backend_config(
            TorchBackendConfig(
                matmul_precision=previous.matmul_precision,
                cudnn_allow_tf32=previous.cudnn_allow_tf32,
                cudnn_benchmark=previous.cudnn_benchmark,
                cudnn_deterministic=previous.cudnn_deterministic,
                deterministic_algorithms=previous.deterministic_algorithms,
                deterministic_warn_only=previous.deterministic_warn_only,
            )
        )


def current_torch_backend_state() -> TorchBackendState:
    """Return the effective process-global PyTorch numerical backend state."""
    cuda_matmul_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    return TorchBackendState(
        matmul_precision=_current_matmul_precision(cuda_matmul_allow_tf32),
        cuda_matmul_allow_tf32=cuda_matmul_allow_tf32,
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _current_matmul_precision(cuda_matmul_allow_tf32: bool) -> MatmulPrecision:
    """Read precision, normalizing a mixed legacy/new TF32 state by its effective flag."""
    try:
        return cast(MatmulPrecision, torch.get_float32_matmul_precision())
    except RuntimeError as exc:
        if "mix of the legacy and new APIs" not in str(exc):
            raise
        return "high" if cuda_matmul_allow_tf32 else "highest"


def _normalized_matmul_precision(
    config: TorchBackendConfig,
    previous: TorchBackendState,
) -> MatmulPrecision | None:
    """Express the legacy CUDA TF32 choice through the readable precision API."""
    precision = config.matmul_precision
    allow_tf32 = config.cuda_matmul_allow_tf32
    if allow_tf32 is None:
        return precision
    if not allow_tf32:
        return "highest"
    if precision == "medium" or (precision is None and previous.matmul_precision == "medium"):
        return "medium"
    return "high"
