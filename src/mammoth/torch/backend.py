"""Project-neutral process-global PyTorch backend and seed configuration.

Training, inference, and profiling entry points use this module to apply
caller-selected numerical backend settings without importing project policy.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

type MatmulPrecision = Literal["highest", "high", "medium"]
type TorchAutocastDtype = Literal["bf16", "fp16"]
type TorchSDPABackend = Literal["default", "flash", "mem-efficient", "math", "cudnn"]

_TORCH_MIN_SEED = -(2**63)
_TORCH_MAX_SEED = 2**64 - 1

TORCH_LOG_COMPONENTS: frozenset[str] = frozenset(
    {
        "all",
        "dynamo",
        "aot",
        "autograd",
        "dynamic",
        "inductor",
        "distributed",
        "c10d",
        "ddp",
        "fsdp",
        "dtensor",
        "onnx",
        "export",
    }
)
TORCH_LOG_ARTIFACTS: frozenset[str] = frozenset(
    {
        "bytecode",
        "aot_graphs",
        "aot_joint_graph",
        "ddp_graphs",
        "graph",
        "graph_code",
        "graph_code_verbose",
        "graph_breaks",
        "graph_sizes",
        "guards",
        "recompiles",
        "recompiles_verbose",
        "trace_source",
        "trace_call",
        "trace_bytecode",
        "output_code",
        "kernel_code",
        "schedule",
        "perf_hints",
        "pre_grad_graphs",
        "post_grad_graphs",
        "ir_pre_fusion",
        "ir_post_fusion",
        "onnx_diagnostics",
        "fusion",
        "overlap",
        "cudagraphs",
        "sym_node",
        "compiled_autograd",
        "compiled_autograd_verbose",
        "cudagraph_static_inputs",
        "benchmarking",
        "autotuning",
        "graph_region_expansion",
        "inductor_metrics",
        "hierarchical_compile",
        "compute_dependencies",
    }
)


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


def parse_torch_logs(requested: str) -> tuple[dict[str, Any], list[str]]:
    """Translate a small ``TORCH_LOGS``-style request into ``set_logs`` kwargs.

    Recognized component tokens (optionally prefixed with ``+``/``-`` for
    debug/error verbosity) map to ``logging`` levels; recognized artifact
    tokens map to booleans. Unrecognized tokens are returned separately so
    callers can decide how to report them. This function only parses; callers
    apply the returned kwargs through ``torch._logging.set_logs`` themselves.
    """
    if not isinstance(requested, str):
        raise TypeError("requested must be a string")
    kwargs: dict[str, Any] = {}
    unsupported: list[str] = []
    for raw_token in requested.split(","):
        token = raw_token.strip()
        if not token:
            continue
        prefix = token[0] if token[0] in {"+", "-"} else ""
        name = token[1:] if prefix else token
        name = name.replace("-", "_")
        if name in TORCH_LOG_COMPONENTS:
            if prefix == "+":
                kwargs[name] = logging.DEBUG
            elif prefix == "-":
                kwargs[name] = logging.ERROR
            else:
                kwargs[name] = logging.INFO
            continue
        if name in TORCH_LOG_ARTIFACTS:
            kwargs[name] = prefix != "-"
            continue
        unsupported.append(token)
    return kwargs, unsupported


def autocast_dtype_for_device(
    device: str,
    requested: TorchAutocastDtype | None,
) -> torch.dtype:
    """Resolve an autocast dtype for a device, defaulting by device type."""
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float16


@contextmanager
def autocast_context(
    device: str,
    *,
    enabled: bool,
    dtype: TorchAutocastDtype | None,
) -> Iterator[None]:
    """Yield an autocast context for one device with an explicit dtype override."""
    device_type = torch.device(device).type
    resolved_dtype = autocast_dtype_for_device(device, dtype)
    with torch.autocast(device_type=device_type, dtype=resolved_dtype, enabled=enabled):
        yield


@contextmanager
def sdpa_backend_context(backend: TorchSDPABackend) -> Iterator[None]:
    """Yield a scaled dot-product attention backend selection context."""
    if backend == "default":
        yield
        return

    backend_map: dict[TorchSDPABackend, Any] = {
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "mem-efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        "math": torch.nn.attention.SDPBackend.MATH,
        "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    }
    with torch.nn.attention.sdpa_kernel(backend_map[backend]):
        yield


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
