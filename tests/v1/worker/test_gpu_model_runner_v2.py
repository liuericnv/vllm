# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_has_fresh_single_token_prefill() -> None:
    assert GPUModelRunner._has_fresh_single_token_prefill(
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
    )
    assert GPUModelRunner._has_fresh_single_token_prefill(
        np.array([4, 1], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([4, 8], dtype=np.int32),
    )
    assert not GPUModelRunner._has_fresh_single_token_prefill(
        np.array([1, 1], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([2, 8], dtype=np.int32),
    )
    assert not GPUModelRunner._has_fresh_single_token_prefill(
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([2], dtype=np.int32),
    )
    assert not GPUModelRunner._has_fresh_single_token_prefill(
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    assert not GPUModelRunner._has_fresh_single_token_prefill(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )


def test_fresh_single_token_prefill_disables_uniform_decode() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(has_inner_state=True)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"request": 0},
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        prompt_len=SimpleNamespace(np=np.array([1], dtype=np.int32)),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request": 1},
        total_num_scheduled_tokens=1,
    )

    assert runner._get_uniform_token_count(scheduler_output, dummy_run=False) is None

    runner.req_states.num_computed_tokens_np[0] = 1
    assert runner._get_uniform_token_count(scheduler_output, dummy_run=False) == 1
    runner.req_states.num_computed_tokens_np[0] = 0
    assert runner._get_uniform_token_count(scheduler_output, dummy_run=True) == 1
