"""Tests for project-neutral PyTorch backend and seed configuration."""

from __future__ import annotations

import random

import pytest
import torch

from mammoth.torch import (
    TorchBackendConfig,
    TorchBackendState,
    TorchSeedPolicy,
    apply_torch_backend_config,
    apply_torch_seed_policy,
    configured_torch_backend,
    current_torch_backend_state,
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
