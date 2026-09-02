"""Unit tests for SimpleKVCache."""

import torch
from minigen.cache import SimpleKVCache


def test_initialization() -> None:
    cache = SimpleKVCache()
    assert cache.is_empty()
    assert cache.num_layers == 0
    assert cache.to_legacy_cache() is None

    cache_fixed = SimpleKVCache(num_layers=4)
    assert cache_fixed.num_layers == 4
    assert cache_fixed.is_empty()


def test_update_layer_concatenation() -> None:
    cache = SimpleKVCache()
    batch_size, num_heads, head_dim = 1, 2, 8

    # Step 1: prompt sequence of length 3
    k1 = torch.randn(batch_size, num_heads, 3, head_dim)
    v1 = torch.randn(batch_size, num_heads, 3, head_dim)

    updated_k, updated_v = cache.update(k1, v1, layer_idx=0)
    assert updated_k.shape == (batch_size, num_heads, 3, head_dim)
    assert updated_v.shape == (batch_size, num_heads, 3, head_dim)
    assert cache.get_seq_length(0) == 3

    # Step 2: new token sequence of length 1
    k2 = torch.randn(batch_size, num_heads, 1, head_dim)
    v2 = torch.randn(batch_size, num_heads, 1, head_dim)

    updated_k2, updated_v2 = cache.update(k2, v2, layer_idx=0)
    assert updated_k2.shape == (batch_size, num_heads, 4, head_dim)
    assert updated_v2.shape == (batch_size, num_heads, 4, head_dim)
    assert cache.get_seq_length(0) == 4


def test_update_multiple_layers_and_legacy_cache() -> None:
    cache = SimpleKVCache()
    b, h, d = 1, 4, 16

    l0_k = torch.randn(b, h, 5, d)
    l0_v = torch.randn(b, h, 5, d)
    l1_k = torch.randn(b, h, 5, d)
    l1_v = torch.randn(b, h, 5, d)

    cache.update(l0_k, l0_v, layer_idx=0)
    cache.update(l1_k, l1_v, layer_idx=1)

    legacy = cache.to_legacy_cache()
    assert legacy is not None
    assert len(legacy) == 2
    assert legacy[0][0].shape == (b, h, 5, d)
    assert legacy[1][0].shape == (b, h, 5, d)
    assert cache.num_layers == 2
    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 5


def test_reset() -> None:
    cache = SimpleKVCache()
    cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), layer_idx=0)
    assert not cache.is_empty()

    cache.reset()
    assert cache.is_empty()
    assert cache.num_layers == 0
    assert cache.to_legacy_cache() is None
