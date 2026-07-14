import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def test_postprocess_state_accepts_int32_mapping_and_skips_sentinels() -> None:
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full((5,), 7, dtype=torch.int32)
    state._align_mode = False
    state._mamba_ctx = None

    idx_mapping = torch.tensor([0, -1, 3], dtype=torch.int32)
    state.postprocess_state(idx_mapping, num_sampled=0)

    torch.testing.assert_close(
        state.num_accepted_tokens_gpu,
        torch.tensor([1, 7, 7, 1, 7], dtype=torch.int32),
    )
