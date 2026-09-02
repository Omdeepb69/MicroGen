"""Unit tests for TensorParallelPyTorchBackend multi-rank execution."""

import pytest
import torch

from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelPyTorchBackend,
)
from microgen.devices import CPUDevice

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_column_and_row_parallel_linear_math():
    in_dim, out_dim = 16, 32
    world_size = 2

    full_linear = torch.nn.Linear(in_dim, out_dim)
    x = torch.randn(2, 4, in_dim)

    # Test ColumnParallelLinear math
    col_parallel = ColumnParallelLinear(
        full_linear.weight.data, full_linear.bias.data if full_linear.bias is not None else None, world_size
    )
    col_out = col_parallel(x)
    ref_out = full_linear(x)

    assert torch.allclose(col_out, ref_out, atol=1e-5)

    # Test RowParallelLinear math
    full_linear_row = torch.nn.Linear(out_dim, in_dim)
    x_row = torch.randn(2, 4, out_dim)
    row_parallel = RowParallelLinear(
        full_linear_row.weight.data, full_linear_row.bias.data if full_linear_row.bias is not None else None, world_size
    )
    row_out = row_parallel(x_row)
    ref_out_row = full_linear_row(x_row)

    assert torch.allclose(row_out, ref_out_row, atol=1e-5)


def test_tensor_parallel_backend_execution_and_logits_equivalence():
    device = CPUDevice()
    single_backend = PyTorchBackend(device=device)
    tp_backend = TensorParallelPyTorchBackend(world_size=2, devices=[device, device])

    single_backend.load_model(MODEL_NAME)
    tp_backend.load_model(MODEL_NAME)

    assert tp_backend._is_parallel is True

    input_ids = torch.tensor([[101, 202, 303]])
    single_logits, _ = single_backend.prefill(input_ids)
    tp_logits, _ = tp_backend.prefill(input_ids)

    assert tp_logits.shape == single_logits.shape
    # Check exact/high similarity logits equivalence
    cos_sim = torch.nn.functional.cosine_similarity(single_logits, tp_logits, dim=-1).mean().item()
    assert cos_sim > 0.999


def test_tensor_parallel_decode_pass():
    device = CPUDevice()
    tp_backend = TensorParallelPyTorchBackend(world_size=2, devices=[device, device])
    tp_backend.load_model(MODEL_NAME)

    input_ids = torch.tensor([[101]])
    logits, cache = tp_backend.prefill(input_ids)

    next_token = tp_backend.sample(logits)
    decode_logits, updated_cache = tp_backend.decode(next_token, cache=cache)

    assert decode_logits.shape[0] == 1
    assert updated_cache is not None


def test_tensor_parallel_memory_info():
    device = CPUDevice()
    tp_backend = TensorParallelPyTorchBackend(world_size=4, devices=[device] * 4)
    tp_backend.load_model(MODEL_NAME)

    info = tp_backend.get_memory_usage()
    assert info["world_size"] == 4
    assert info["is_tensor_parallel"] is True
    assert len(info["ranks"]) == 4
