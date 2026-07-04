# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    MambaModelConfig,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_hybrid_dcp_config(**overrides):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            cache_dtype="auto",
            calculate_kv_scales=False,
            kv_offloading_size=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            dcp_kv_cache_interleave_size=1,
            dcp_comm_backend="ag_rs",
        ),
        model_config=SimpleNamespace(use_mla=True, model="test-model"),
        attention_config=SimpleNamespace(backend=None),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    for path, value in overrides.items():
        target = config
        parts = path.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return config


def test_hybrid_mla_dcp_selects_triton_mla(monkeypatch):
    monkeypatch.setattr(
        MambaModelConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    config = _make_hybrid_dcp_config()

    HybridAttentionMambaModelConfig.verify_and_update_config(config)

    assert config.attention_config.backend == AttentionBackendEnum.TRITON_MLA


@pytest.mark.parametrize(
    "backend",
    [AttentionBackendEnum.TRITON_MLA, AttentionBackendEnum.FLASHINFER_MLA],
)
def test_hybrid_mla_dcp_accepts_lse_backend(monkeypatch, backend):
    monkeypatch.setattr(
        MambaModelConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    config = _make_hybrid_dcp_config(attention_config__backend=backend)

    HybridAttentionMambaModelConfig.verify_and_update_config(config)

    assert config.attention_config.backend == backend


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.ROCM_AITER_MLA,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
    ],
)
def test_hybrid_mla_dcp_rejects_unvalidated_backend(monkeypatch, backend):
    monkeypatch.setattr(
        MambaModelConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    config = _make_hybrid_dcp_config(attention_config__backend=backend)

    with pytest.raises(ValueError, match="TRITON_MLA and FLASHINFER_MLA"):
        HybridAttentionMambaModelConfig.verify_and_update_config(config)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"parallel_config__dcp_kv_cache_interleave_size": 2},
            "CP KV cache interleave size",
        ),
        ({"cache_config__kv_offloading_size": 4.0}, "KV offloading"),
    ],
)
def test_hybrid_dcp_rejects_deferred_unsupported_options(
    monkeypatch, overrides, error
):
    monkeypatch.setattr(
        MambaModelConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    config = _make_hybrid_dcp_config(**overrides)

    with pytest.raises(ValueError, match=error):
        HybridAttentionMambaModelConfig.verify_and_update_config(config)
