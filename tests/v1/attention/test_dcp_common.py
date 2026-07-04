# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out


@pytest.mark.parametrize("invalid_lse", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_zeros_empty_dcp_partition(
    invalid_lse: float, is_lse_base_on_e: bool
):
    out = torch.full((1, 2, 8), float("nan"), device="cuda")
    lses = torch.tensor(
        [[[0.0, 0.0]], [[invalid_lse, invalid_lse]]],
        device="cuda",
        dtype=torch.float32,
    )

    corrected, final_lse = correct_attn_out(
        out,
        lses,
        cp_rank=1,
        ctx=CPTritonContext(),
        is_lse_base_on_e=is_lse_base_on_e,
    )

    torch.testing.assert_close(corrected, torch.zeros_like(corrected))
    torch.testing.assert_close(final_lse, torch.zeros_like(final_lse))
