"""Unit tests for microgen PyTorch execution backend."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from microgen.devices import CPUDevice
from microgen.backends import InferenceBackend, PyTorchBackend
from minigen.cache import SimpleKVCache

MODEL_NAME = "sshleifer/tiny-gpt2"


@pytest.fixture
def cpu_backend():
    backend = PyTorchBackend(device=CPUDevice())
    backend.load_model(MODEL_NAME)
    return backend


def test_pytorch_backend_protocol(cpu_backend):
    assert isinstance(cpu_backend, InferenceBackend)
    assert cpu_backend.device.name == "cpu"
    assert cpu_backend.model is not None


def test_pytorch_backend_prefill_and_decode(cpu_backend):
    input_ids = torch.tensor([[101, 200, 300]])
    cache = SimpleKVCache()

    # Prefill pass
    logits, updated_cache = cpu_backend.prefill(input_ids, cache=cache)
    assert logits.dim() == 2
    assert logits.size(0) == 1
    assert updated_cache is not None

    # Decode pass
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    decode_logits, updated_cache2 = cpu_backend.decode(next_token, cache=updated_cache)
    assert decode_logits.dim() == 2
    assert decode_logits.size(0) == 1


def test_pytorch_backend_sampling(cpu_backend):
    logits = torch.tensor([[1.0, 5.0, 2.0]])

    # Greedy sampling
    token_greedy = cpu_backend.sample(logits, temperature=0.0)
    assert token_greedy.item() == 1  # Index of max value (5.0)

    # Temperature & top_k sampling
    token_sampled = cpu_backend.sample(logits, temperature=0.7, top_k=2)
    assert token_sampled.size() == (1, 1)


def test_pytorch_backend_memory_usage(cpu_backend):
    mem = cpu_backend.get_memory_usage()
    assert isinstance(mem, dict)
    assert "total_bytes" in mem
