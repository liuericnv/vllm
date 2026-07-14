from types import SimpleNamespace

import torch

from vllm.model_executor.models.kimi_linear import KimiLinearForCausalLM


def test_make_empty_intermediate_tensors() -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=16)

    intermediate_tensors = model.make_empty_intermediate_tensors(
        batch_size=7,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert set(intermediate_tensors.tensors) == {"hidden_states", "residual"}
    for tensor in intermediate_tensors.tensors.values():
        assert tensor.shape == (7, 16)
        assert tensor.dtype == torch.bfloat16
        assert tensor.device.type == "cpu"
        assert torch.count_nonzero(tensor) == 0
