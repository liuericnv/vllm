# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


@pytest.mark.parametrize(
    (
        "manager_block_size",
        "kernel_block_size",
        "manager_block_count",
        "actual_kernel_blocks",
        "expected_table_width",
    ),
    [
        (960, 64, 1, 15, 16),
        (480, 32, 3, 45, 48),
        (256, 64, 3, 12, 12),
    ],
)
def test_block_tables_align_expanded_kernel_width(
    manager_block_size: int,
    kernel_block_size: int,
    manager_block_count: int,
    actual_kernel_blocks: int,
    expected_table_width: int,
):
    block_tables = BlockTables(
        block_sizes=[manager_block_size],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[manager_block_count],
        device=torch.device("cuda"),
        kernel_block_sizes=[kernel_block_size],
    )

    assert block_tables.block_tables[0].gpu.shape[1] == expected_table_width
    assert block_tables.input_block_tables[0].shape[1] == expected_table_width
    assert expected_table_width % (128 // kernel_block_size) == 0

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=(list(range(manager_block_count)),),
        overwrite=True,
    )
    assert block_tables.num_blocks.np[0, 0] == actual_kernel_blocks


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_slot_mappings_use_per_group_cp_layout(dcp_rank: int):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[4, 4],
        max_num_reqs=1,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4, 4],
        device=device,
        kernel_block_sizes=[4, 4],
        cp_size=2,
        cp_rank=dcp_rank,
        cp_sizes=[2, 1],
        cp_ranks=[dcp_rank, 0],
    )
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([10, 11], [20, 21, 22]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 10], dtype=torch.int32, device=device),
        positions=torch.arange(10, dtype=torch.int64, device=device),
        num_tokens_padded=12,
    )
    torch.accelerator.synchronize()

    expected_attention = []
    attention_blocks = [10, 11]
    for position in range(10):
        virtual_offset = position % 8
        if virtual_offset % 2 != dcp_rank:
            expected_attention.append(PAD_SLOT_ID)
        else:
            block_number = attention_blocks[position // 8]
            expected_attention.append(block_number * 4 + virtual_offset // 2)
    expected_attention.extend([PAD_SLOT_ID, PAD_SLOT_ID])

    recurrent_blocks = [20, 21, 22]
    expected_recurrent = [
        recurrent_blocks[position // 4] * 4 + position % 4 for position in range(10)
    ]
    expected_recurrent.extend([PAD_SLOT_ID, PAD_SLOT_ID])

    assert slot_mappings.tolist() == [expected_attention, expected_recurrent]
