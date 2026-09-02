"""Prefix KV Cache manager for prompt prefix hashing, matching, and KV cache reuse."""

import hashlib
from typing import Dict, List, Optional, Tuple
from microgen.runtime.kv_cache import KVCacheState


def compute_token_hash(token_ids: List[int]) -> str:
    """Compute a SHA256 hex digest for a sequence of token IDs."""
    token_bytes = ",".join(str(tid) for tid in token_ids).encode("utf-8")
    return hashlib.sha256(token_bytes).hexdigest()


class PrefixKVCache:
    """Manager storing precomputed KV cache states indexed by prompt token prefixes."""

    def __init__(self, max_capacity: int = 100) -> None:
        self.max_capacity = max_capacity
        self._cache: Dict[str, Tuple[List[int], KVCacheState]] = {}

    @property
    def size(self) -> int:
        """Return the number of stored prefix cache entries."""
        return len(self._cache)

    def insert(self, token_ids: List[int], kv_cache_state: KVCacheState) -> str:
        """Insert a precomputed KV cache state for a sequence of token IDs.

        Returns the computed cache key hash string.
        """
        if not token_ids:
            raise ValueError("Cannot cache empty token sequence.")

        cache_key = compute_token_hash(token_ids)

        if len(self._cache) >= self.max_capacity and cache_key not in self._cache:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = (list(token_ids), kv_cache_state)
        return cache_key

    def get(self, cache_key: str) -> Optional[KVCacheState]:
        """Lookup a cached KV state by exact hash key."""
        entry = self._cache.get(cache_key)
        return entry[1] if entry is not None else None

    def match_prefix(self, token_ids: List[int]) -> Optional[Tuple[int, KVCacheState]]:
        """Find the longest matching prefix sequence present in the cache.

        Returns (matched_prefix_length, cached_kv_state) or None if no match is found.
        """
        if not token_ids:
            return None

        best_match_len = 0
        best_match_cache: Optional[KVCacheState] = None

        for cached_tokens, kv_state in self._cache.values():
            cached_len = len(cached_tokens)
            if cached_len <= len(token_ids) and cached_len > best_match_len:
                if token_ids[:cached_len] == cached_tokens:
                    best_match_len = cached_len
                    best_match_cache = kv_state

        if best_match_cache is not None:
            return best_match_len, best_match_cache

        return None

    def clear(self) -> None:
        """Clear all entries from the prefix cache."""
        self._cache.clear()
