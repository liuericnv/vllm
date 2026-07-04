# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.platforms import current_platform

FLASHINFER_LSE_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        reason="FlashInfer MLA requires compute capability family 10.x.",
        allow_module_level=True,
    )
else:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.v1.attention.backends.mla import (
        flashinfer_mla as flashinfer_mla_module,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla import FlashInferMLAImpl


def ref_mla(
    out: Tensor,  # (bs, num_heads, v_head_dim)
    query: Tensor,  # (bs, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]
    lse = torch.empty((bs, num_heads), dtype=torch.float32, device=query.device)

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

        # FlashInfer's TRTLLM-gen MLA kernel returns log2 LSE. DCP uses this
        # value to normalize and combine attention outputs across KV shards.
        logits = torch.matmul(query[i].float(), kv[0].float().transpose(0, 1)) * scale
        lse[i] = torch.logsumexp(logits, dim=-1) / math.log(2.0)

    return out, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
def test_flashinfer_mla_decode(dtype: torch.dtype, bs: int, block_size: int):
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Deepseek R1 config
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    MAX_SEQ_LEN = 1024

    seq_lens = [torch.randint(2, MAX_SEQ_LEN, (1,)).item() for _ in range(bs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Generate block tables with random but unique block IDs
    # From https://github.com/flashinfer-ai/flashinfer/pull/1222
    blocks_per_seq = (seq_lens_tensor + block_size - 1) // block_size
    max_num_blocks_per_seq = max(blocks_per_seq.max().item(), 4)
    total_blocks_needed = sum(blocks_per_seq)
    # Get random unique IDs for all blocks
    all_block_ids = torch.randperm(total_blocks_needed)

    block_id = 0
    block_tables = torch.zeros(
        (bs, max_num_blocks_per_seq),
        dtype=torch.int32,
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(bs):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    kv_cache = torch.randn(block_tables.numel(), block_size, qk_head_dim).to(dtype)
    q = torch.randn(bs, num_heads, qk_head_dim).to(dtype)

    out_ref = q.new_zeros(bs, num_heads, kv_lora_rank)
    out_ref, lse_ref = ref_mla(
        out_ref, q, kv_cache, scale, block_tables, seq_lens_tensor
    )

    workspace_buffer = torch.zeros(
        FLASHINFER_LSE_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device=q.device,
    )
    # Flashinfer MLA expects the query to be of shape
    # (bs, q_len_per_request, num_heads, qk_head_dim),
    # where q_len_per_request is the MTP query length (=1 without MTP)
    q = q.unsqueeze(1)

    out_ans, lse_ans = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale,
        return_lse=True,
    )
    out_ans = out_ans.squeeze(1)
    assert lse_ans.dtype == torch.float32
    assert lse_ans.shape == (bs, num_heads)
    torch.testing.assert_close(out_ans, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_ans, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("query_len", [1, 2])
@pytest.mark.parametrize("empty_index", [0, 1])
@pytest.mark.parametrize("empty_payload", [123.0, float("nan")])
def test_flashinfer_mla_dcp_masks_empty_local_sequences(
    monkeypatch: pytest.MonkeyPatch,
    query_len: int,
    empty_index: int,
    empty_payload: float,
):
    """Empty DCP shards must be neutral even if FlashInfer returns garbage.

    DCP currently routes only Q=1 through FlashInfer MLA decode. Q=2 covers
    output/LSE mask broadcasting, not speculative-decode causality.
    """
    device = "cuda"
    batch_size = 2
    num_heads = 4
    kv_lora_rank = 6
    qk_nope_head_dim = 4
    qk_rope_head_dim = 2
    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    impl = object.__new__(FlashInferMLAImpl)
    impl.qk_nope_head_dim = qk_nope_head_dim
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = qk_rope_head_dim
    impl.kv_cache_dtype = "auto"
    impl.scale = 0.5
    impl.bmm1_scale = None
    impl.bmm2_scale = None
    impl.need_to_return_lse_for_decode = True

    # This is the per-rank local length passed to FlashInfer by the MLA
    # metadata builder, not the request's global sequence length.
    local_lengths = [5, 5]
    local_lengths[empty_index] = 0
    seq_lens = torch.tensor(local_lengths, dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        num_decode_tokens=batch_size * query_len,
        num_decodes=batch_size,
        max_seq_len=5,
        decode=SimpleNamespace(
            block_table=torch.zeros((batch_size, 1), dtype=torch.int32, device=device),
            seq_lens=seq_lens,
        ),
    )
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)

    kernel_args = {}

    def fake_decode_kernel(**kwargs):
        kernel_args.update(kwargs)
        out = torch.full(
            (batch_size, query_len, num_heads, kv_lora_rank),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        )
        lse = torch.full(
            (batch_size * query_len, num_heads),
            9.0,
            dtype=torch.float32,
            device=device,
        )
        out[empty_index].fill_(empty_payload)
        empty_start = empty_index * query_len
        lse[empty_start : empty_start + query_len].fill_(empty_payload)
        return out, lse

    monkeypatch.setattr(
        flashinfer_mla_module,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_decode_kernel,
    )
    monkeypatch.setattr(
        flashinfer_mla_module,
        "_get_workspace_buffer",
        lambda return_lse: torch.empty(1, dtype=torch.uint8, device=device),
    )

    query = torch.zeros(
        (batch_size * query_len, num_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.zeros((1, 1, qk_head_dim), dtype=torch.bfloat16, device=device)
    out, lse = impl.forward_mqa(query, kv_cache, metadata, layer)

    assert kernel_args["seq_lens"].data_ptr() == seq_lens.data_ptr()
    assert out.shape == (
        batch_size * query_len,
        num_heads,
        kv_lora_rank,
    )
    out_by_request = out.view(batch_size, query_len, num_heads, kv_lora_rank)
    lse_by_request = lse.view(batch_size, query_len, num_heads)
    assert torch.count_nonzero(out_by_request[empty_index]).item() == 0
    assert torch.isneginf(lse_by_request[empty_index]).all()
    nonempty_index = 1 - empty_index
    torch.testing.assert_close(
        out_by_request[nonempty_index],
        torch.full_like(out_by_request[nonempty_index], 7.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lse_by_request[nonempty_index],
        torch.full_like(lse_by_request[nonempty_index], 9.0),
        rtol=0,
        atol=0,
    )
