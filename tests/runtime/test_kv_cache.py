"""Unit tests for microgen runtime KV cache state and lifecycle manager."""

import pytest
import torch
from microgen.runtime import KVCacheState, KVCacheManager
from microgen.backends import PyTorchBackend
from microgen.devices import CPUDevice

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_kv_cache_state_operations():
    cache = KVCacheState(max_seq_len=1024)
    assert cache.get_seq_length(0) == 0
    assert cache.get_memory_usage_bytes() == 0

    key_layer0 = torch.randn(1, 4, 3, 16)  # [batch, heads, seq_len, head_dim]
    val_layer0 = torch.randn(1, 4, 3, 16)

    updated_k, updated_v = cache.update(key_layer0, val_layer0, layer_idx=0)
    assert updated_k.shape == (1, 4, 3, 16)
    assert cache.get_seq_length(0) == 3
    assert cache.get_mask_sizes(query_length=1, layer_idx=0) == (4, 0)
    assert cache.get_memory_usage_bytes() > 0

    # Append 1 token
    key_step = torch.randn(1, 4, 1, 16)
    val_step = torch.randn(1, 4, 1, 16)
    updated_k2, updated_v2 = cache.update(key_step, val_step, layer_idx=0)
    assert updated_k2.shape == (1, 4, 4, 16)
    assert cache.get_seq_length(0) == 4

    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache.get_memory_usage_bytes() == 0


def test_kv_cache_manager_lifecycle():
    manager = KVCacheManager()
    assert manager.active_requests_count() == 0

    req1_cache = manager.allocate("req-1")
    req2_cache = manager.allocate("req-2")

    assert manager.active_requests_count() == 2
    assert manager.get("req-1") is req1_cache
    assert manager.get("req-2") is req2_cache

    # Update cache tensors
    req1_cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), layer_idx=0)
    req2_cache.update(torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8), layer_idx=0)

    assert manager.get_total_memory_usage_bytes() > 0

    # Free req-1
    freed = manager.free("req-1")
    assert freed is True
    assert manager.get("req-1") is None
    assert manager.active_requests_count() == 1

    manager.clear()
    assert manager.active_requests_count() == 0
    assert manager.get_total_memory_usage_bytes() == 0


def test_kv_cache_state_integration_with_pytorch_backend():
    backend = PyTorchBackend(device=CPUDevice())
    backend.load_model(MODEL_NAME)

    manager = KVCacheManager()
    cache = manager.allocate("req-e2e")

    input_ids = torch.tensor([[100, 200, 300]])
    logits, updated_cache = backend.prefill(input_ids, cache=cache)

    assert logits.shape == (1, 50257)
    assert updated_cache.get_seq_length(0) == 3
    assert updated_cache.get_memory_usage_bytes() > 0

    manager.free("req-e2e")
    assert manager.get_total_memory_usage_bytes() == 0


def test_sliding_window_eviction():
    cache = KVCacheState(max_seq_len=1024, sliding_window_size=4)
    key_initial = torch.randn(1, 2, 5, 8)  # seq_len=5 > window_size=4
    val_initial = torch.randn(1, 2, 5, 8)

    k_out, v_out = cache.update(key_initial, val_initial, layer_idx=0)
    assert k_out.shape == (1, 2, 4, 8)
    assert cache.get_seq_length(0) == 4

    # Append 2 more tokens -> seq_len=6 cut to 4
    k_step = torch.randn(1, 2, 2, 8)
    v_step = torch.randn(1, 2, 2, 8)
    k_out2, v_out2 = cache.update(k_step, v_step, layer_idx=0)
    assert k_out2.shape == (1, 2, 4, 8)
    assert cache.get_seq_length(0) == 4


def test_repeat_kv_gqa():
    from microgen.runtime.kv_cache import repeat_kv

    # 2 KV heads, n_rep=4 -> 8 query heads total
    kv_states = torch.randn(2, 2, 10, 16)  # [bsz=2, num_kv_heads=2, seq_len=10, head_dim=16]
    repeated = repeat_kv(kv_states, n_rep=4)
    assert repeated.shape == (2, 8, 10, 16)

    # n_rep=1 (MHA case) should return unchanged tensor
    single = repeat_kv(kv_states, n_rep=1)
    assert single.shape == (2, 2, 10, 16)


def test_quantized_kv_cache_state():
    # Compare memory usage between unquantized float32 KV cache and INT8 quantized KV cache
    float_cache = KVCacheState(quantize_kv=False)
    int8_cache = KVCacheState(quantize_kv=True)

    k_tensor = torch.randn(1, 4, 64, 32, dtype=torch.float32)
    v_tensor = torch.randn(1, 4, 64, 32, dtype=torch.float32)

    float_k, float_v = float_cache.update(k_tensor, v_tensor, layer_idx=0)
    int8_k, int8_v = int8_cache.update(k_tensor, v_tensor, layer_idx=0)

    # Returned output float shapes and values match
    assert float_k.shape == int8_k.shape
    cos_sim = torch.nn.functional.cosine_similarity(float_k, int8_k, dim=-1).mean().item()
    assert cos_sim > 0.98

    # Memory usage of INT8 cache should be significantly smaller (nearly 2x-4x memory reduction)
    float_bytes = float_cache.get_memory_usage_bytes()
    int8_bytes = int8_cache.get_memory_usage_bytes()
    assert int8_bytes < float_bytes * 0.6  # Verify > 1.6x - 2x compression ratio
