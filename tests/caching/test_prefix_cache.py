"""Unit tests for microgen prefix KV cache manager."""

import pytest
from microgen.runtime.kv_cache import KVCacheState
from microgen.caching import PrefixKVCache, compute_token_hash


def test_token_hash_deterministic():
    tokens1 = [101, 200, 300]
    tokens2 = [101, 200, 300]
    tokens3 = [101, 200, 301]

    hash1 = compute_token_hash(tokens1)
    hash2 = compute_token_hash(tokens2)
    hash3 = compute_token_hash(tokens3)

    assert hash1 == hash2
    assert hash1 != hash3


def test_prefix_cache_insert_and_get():
    cache = PrefixKVCache(max_capacity=10)
    kv_state = KVCacheState(max_seq_len=50)

    prompt_tokens = [1, 2, 3, 4, 5]
    key = cache.insert(prompt_tokens, kv_state)

    assert cache.size == 1
    retrieved = cache.get(key)
    assert retrieved is kv_state


def test_prefix_cache_match_prefix():
    cache = PrefixKVCache(max_capacity=10)
    kv_state_short = KVCacheState(max_seq_len=50)
    kv_state_long = KVCacheState(max_seq_len=50)

    # Insert short system prompt prefix [10, 20]
    cache.insert([10, 20], kv_state_short)
    # Insert longer system prompt + context prefix [10, 20, 30, 40]
    cache.insert([10, 20, 30, 40], kv_state_long)

    # Query with prompt starting with [10, 20, 30, 40, 50, 60] -> should match long prefix (length 4)
    match = cache.match_prefix([10, 20, 30, 40, 50, 60])
    assert match is not None
    matched_len, matched_kv = match
    assert matched_len == 4
    assert matched_kv is kv_state_long

    # Query with prompt starting with [10, 20, 99, 100] -> should match short prefix (length 2)
    match_short = cache.match_prefix([10, 20, 99, 100])
    assert match_short is not None
    matched_len2, matched_kv2 = match_short
    assert matched_len2 == 2
    assert matched_kv2 is kv_state_short

    # Query with completely non-matching prompt [99, 99] -> should return None
    assert cache.match_prefix([99, 99]) is None


def test_prefix_cache_eviction():
    cache = PrefixKVCache(max_capacity=2)
    kv1 = KVCacheState(max_seq_len=10)
    kv2 = KVCacheState(max_seq_len=10)
    kv3 = KVCacheState(max_seq_len=10)

    k1 = cache.insert([1], kv1)
    k2 = cache.insert([2], kv2)
    assert cache.size == 2

    # Inserting 3rd entry should evict oldest (k1)
    k3 = cache.insert([3], kv3)
    assert cache.size == 2
    assert cache.get(k1) is None
    assert cache.get(k2) is kv2
    assert cache.get(k3) is kv3


def test_prefix_cache_clear():
    cache = PrefixKVCache(max_capacity=10)
    kv = KVCacheState(max_seq_len=10)
    cache.insert([1, 2], kv)

    assert cache.size == 1
    cache.clear()
    assert cache.size == 0
