"""Unit tests for QuantizedPyTorchBackend INT8 weight quantization."""

import pytest
import torch

from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.devices.cpu import CPUDevice

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_quantized_backend_loading_and_execution():
    device = CPUDevice()
    fp_backend = PyTorchBackend(device=device)
    quant_backend = QuantizedPyTorchBackend(device=device, quant_type="int8")

    fp_backend.load_model(MODEL_NAME)
    quant_backend.load_model(MODEL_NAME)

    assert quant_backend._quantized is True

    input_ids = torch.tensor([[100, 200, 300]])
    fp_logits, _ = fp_backend.prefill(input_ids)
    quant_logits, _ = quant_backend.prefill(input_ids)

    assert fp_logits.shape == quant_logits.shape
    # Check high cosine similarity between float and INT8 quantized logits
    cos_sim = torch.nn.functional.cosine_similarity(fp_logits, quant_logits, dim=-1).mean().item()
    assert cos_sim > 0.98


def test_quantized_decode_pass():
    device = CPUDevice()
    backend = QuantizedPyTorchBackend(device=device)
    backend.load_model(MODEL_NAME)

    input_ids = torch.tensor([[101]])
    logits, cache = backend.prefill(input_ids)

    next_token = backend.sample(logits)
    decode_logits, updated_cache = backend.decode(next_token, cache=cache)

    assert decode_logits.shape[0] == 1
    assert updated_cache is not None


def test_quantized_memory_info():
    device = CPUDevice()
    backend = QuantizedPyTorchBackend(device=device, quant_type="int8")
    backend.load_model(MODEL_NAME)

    info = backend.get_memory_usage()
    assert info["is_quantized"] is True
    assert info["quant_type"] == "int8"
