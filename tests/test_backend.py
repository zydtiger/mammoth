"""Tests for project-neutral PyTorch backend and seed configuration."""

from __future__ import annotations

import logging
import random

import pytest
import torch

from mammoth.torch import (
    TORCH_LOG_ARTIFACTS,
    TORCH_LOG_COMPONENTS,
    TorchBackendConfig,
    TorchBackendState,
    TorchSeedPolicy,
    apply_torch_backend_config,
    apply_torch_seed_policy,
    autocast_context,
    autocast_dtype_for_device,
    configured_torch_backend,
    current_torch_backend_state,
    parse_torch_logs,
    sdpa_backend_context,
)


def backend_config_from_state(state: TorchBackendState) -> TorchBackendConfig:
    """Return an explicit configuration that restores one captured state."""
    return TorchBackendConfig(
        matmul_precision=state.matmul_precision,
        cudnn_allow_tf32=state.cudnn_allow_tf32,
        cudnn_benchmark=state.cudnn_benchmark,
        cudnn_deterministic=state.cudnn_deterministic,
        deterministic_algorithms=state.deterministic_algorithms,
        deterministic_warn_only=state.deterministic_warn_only,
    )


def test_backend_config_applies_persistent_caller_selected_values() -> None:
    before = current_torch_backend_state()
    try:
        state = apply_torch_backend_config(
            TorchBackendConfig(
                matmul_precision="high",
                cuda_matmul_allow_tf32=True,
                cudnn_allow_tf32=not before.cudnn_allow_tf32,
                cudnn_benchmark=not before.cudnn_benchmark,
                cudnn_deterministic=not before.cudnn_deterministic,
                deterministic_algorithms=not before.deterministic_algorithms,
                deterministic_warn_only=True,
            )
        )

        assert state.matmul_precision == "high"
        assert state.cuda_matmul_allow_tf32 is True
        assert state.cudnn_allow_tf32 is not before.cudnn_allow_tf32
        assert state.cudnn_benchmark is not before.cudnn_benchmark
        assert state.cudnn_deterministic is not before.cudnn_deterministic
        assert state.deterministic_algorithms is not before.deterministic_algorithms
        assert state.deterministic_warn_only is True
    finally:
        apply_torch_backend_config(backend_config_from_state(before))


def test_configured_backend_restores_state_after_failure() -> None:
    before = current_torch_backend_state()

    with pytest.raises(RuntimeError, match="backend failed"), configured_torch_backend(
        TorchBackendConfig(cudnn_benchmark=not before.cudnn_benchmark)
    ) as state:
        assert state.cudnn_benchmark is not before.cudnn_benchmark
        raise RuntimeError("backend failed")

    assert current_torch_backend_state() == before


def test_backend_state_normalizes_mixed_legacy_tf32_configuration() -> None:
    before = current_torch_backend_state()
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = False

        state = current_torch_backend_state()
        assert state.matmul_precision == "highest"
        assert state.cuda_matmul_allow_tf32 is False

        with configured_torch_backend(
            TorchBackendConfig(cudnn_benchmark=not state.cudnn_benchmark)
        ):
            pass

        assert current_torch_backend_state().matmul_precision == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        apply_torch_backend_config(backend_config_from_state(before))


def test_backend_config_rejects_invalid_warn_only_before_mutation() -> None:
    before = current_torch_backend_state()

    with pytest.raises(ValueError, match="deterministic_warn_only must be a boolean"):
        apply_torch_backend_config(
            TorchBackendConfig(
                matmul_precision="medium",
                deterministic_algorithms=True,
                deterministic_warn_only=None,  # type: ignore[arg-type]
            )
        )

    assert current_torch_backend_state() == before


def test_backend_config_rejects_warn_only_without_deterministic_mode() -> None:
    with pytest.raises(ValueError, match="requires deterministic_algorithms"):
        TorchBackendConfig(deterministic_warn_only=True)


def test_seed_policy_reproduces_python_and_torch_sequences() -> None:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    policy = TorchSeedPolicy(27, torch_cuda=False)
    try:
        apply_torch_seed_policy(policy)
        first = (random.random(), torch.rand(3))
        apply_torch_seed_policy(policy)
        second = (random.random(), torch.rand(3))

        assert first[0] == second[0]
        assert torch.equal(first[1], second[1])
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


def test_seed_policy_can_target_available_cuda_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", observed.append)

    apply_torch_seed_policy(
        TorchSeedPolicy(31, python_random=False, torch_cpu=False, torch_cuda=True)
    )

    assert observed == [31]


def test_seed_policy_can_seed_torch_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    torch_state = torch.get_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", observed.append)
    try:
        apply_torch_seed_policy(
            TorchSeedPolicy(37, python_random=False, torch_cpu=True, torch_cuda=False)
        )

        assert observed == []
        assert torch.initial_seed() == 37
    finally:
        torch.set_rng_state(torch_state)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "7"],
)
def test_seed_policy_rejects_non_integer_seed(value: object) -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        TorchSeedPolicy(value)  # type: ignore[arg-type]


def test_seed_policy_rejects_out_of_range_torch_seed_without_mutation() -> None:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()

    with pytest.raises(ValueError, match="when seeding Torch"):
        TorchSeedPolicy(2**100, torch_cuda=False)

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_seed_policy_allows_arbitrary_python_only_seed() -> None:
    python_state = random.getstate()
    policy = TorchSeedPolicy(2**100, torch_cpu=False, torch_cuda=False)
    try:
        apply_torch_seed_policy(policy)
        first = random.random()
        apply_torch_seed_policy(policy)

        assert random.random() == first
    finally:
        random.setstate(python_state)


def test_torch_log_component_and_artifact_tables_are_disjoint_frozensets() -> None:
    assert isinstance(TORCH_LOG_COMPONENTS, frozenset)
    assert isinstance(TORCH_LOG_ARTIFACTS, frozenset)
    assert TORCH_LOG_COMPONENTS.isdisjoint(TORCH_LOG_ARTIFACTS)


def test_parse_torch_logs_maps_known_components_and_artifacts_with_prefixes() -> None:
    kwargs, unsupported = parse_torch_logs("+dynamo,-aot,inductor,graph_breaks,-recompiles")

    assert kwargs == {
        "dynamo": logging.DEBUG,
        "aot": logging.ERROR,
        "inductor": logging.INFO,
        "graph_breaks": True,
        "recompiles": False,
    }
    assert unsupported == []


def test_parse_torch_logs_normalizes_hyphens_and_collects_unsupported_tokens() -> None:
    kwargs, unsupported = parse_torch_logs("compiled-autograd, not_a_real_token, ,")

    assert kwargs == {"compiled_autograd": True}
    assert unsupported == ["not_a_real_token"]


def test_parse_torch_logs_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="requested must be a string"):
        parse_torch_logs(None)  # type: ignore[arg-type]


def test_autocast_dtype_for_device_prefers_explicit_request() -> None:
    assert autocast_dtype_for_device("cpu", "bf16") is torch.bfloat16
    assert autocast_dtype_for_device("cpu", "fp16") is torch.float16


def test_autocast_dtype_for_device_defaults_to_fp16_off_cuda() -> None:
    assert autocast_dtype_for_device("cpu", None) is torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_autocast_dtype_for_device_defaults_to_bf16_on_cuda() -> None:
    assert autocast_dtype_for_device("cuda", None) is torch.bfloat16


def test_autocast_context_casts_cpu_matmul_to_requested_dtype() -> None:
    left = torch.randn(4, 4, dtype=torch.float32)
    right = torch.randn(4, 4, dtype=torch.float32)

    with autocast_context("cpu", enabled=True, dtype="bf16"):
        result = left @ right

    assert result.dtype == torch.bfloat16


def test_autocast_context_disabled_keeps_float32() -> None:
    left = torch.randn(4, 4, dtype=torch.float32)
    right = torch.randn(4, 4, dtype=torch.float32)

    with autocast_context("cpu", enabled=False, dtype="bf16"):
        result = left @ right

    assert result.dtype == torch.float32


def test_sdpa_backend_context_default_is_a_noop() -> None:
    with sdpa_backend_context("default"):
        pass


def test_sdpa_backend_context_selects_math_backend_on_cpu() -> None:
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)

    with sdpa_backend_context("math"):
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert result.shape == (1, 1, 4, 8)


def test_sdpa_backend_context_rejects_unrecognized_backend_name() -> None:
    with pytest.raises(KeyError), sdpa_backend_context("bogus"):  # type: ignore[arg-type]
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sdpa_backend_context_selects_flash_backend_on_cuda() -> None:
    query = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)
    value = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)

    with sdpa_backend_context("flash"):
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert result.shape == (1, 1, 4, 8)
