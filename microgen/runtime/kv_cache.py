"""Key-Value (KV) cache runtime state management and per-request lifecycle manager."""

from typing import Dict, List, Tuple, Optional, Any
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object  # type: ignore[misc,assignment]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key/Value heads n_rep times for Grouped-Query Attention (GQA).
    
    Equivalent to torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep) without memory copies.
    Expected tensor shape: (batch_size, num_kv_heads, seq_len, head_dim).
    Output tensor shape: (batch_size, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)
    )


def _quantize_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating point tensor to INT8 per-vector scale factor."""
    max_val = torch.max(torch.abs(t), dim=-1, keepdim=True).values
    scale = (max_val / 127.0).clamp(min=1e-8)
    t_int8 = torch.round(t / scale).clamp(-128, 127).to(torch.int8)
    return t_int8, scale


def _dequantize_tensor(t_int8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize INT8 tensor back to target float dtype using scale factor."""
    return t_int8.to(dtype) * scale


class KVCacheState(Cache):
    """Per-request per-layer Key-Value cache subclassing HuggingFace Cache.

    Stores key and value tensors across model layers, tracks current sequence length,
    supports sliding window eviction, dynamic INT8 quantization, and calculates exact memory footprint in bytes.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        sliding_window_size: Optional[int] = None,
        quantize_kv: bool = False,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.sliding_window_size = sliding_window_size
        self.quantize_kv = quantize_kv
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.key_scales: List[Optional[torch.Tensor]] = []
        self.value_scales: List[Optional[torch.Tensor]] = []
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update key and value tensors for specified layer index with sliding-window and INT8 quantization support."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.key_scales.append(None)
            self.value_scales.append(None)

        dtype = key_states.dtype

        if self.key_cache[layer_idx] is None:
            k_out = key_states
            v_out = value_states
        else:
            cached_k = self.key_cache[layer_idx]
            cached_v = self.value_cache[layer_idx]
            assert cached_k is not None and cached_v is not None

            if self.quantize_kv and self.key_scales[layer_idx] is not None and self.value_scales[layer_idx] is not None:
                cached_k = _dequantize_tensor(cached_k, self.key_scales[layer_idx], dtype)
                cached_v = _dequantize_tensor(cached_v, self.value_scales[layer_idx], dtype)

            k_out = torch.cat([cached_k, key_states], dim=-2)
            v_out = torch.cat([cached_v, value_states], dim=-2)

        # Enforce sliding-window context eviction if sliding_window_size is specified
        if self.sliding_window_size is not None and k_out.shape[-2] > self.sliding_window_size:
            k_out = k_out[..., -self.sliding_window_size:, :]
            v_out = v_out[..., -self.sliding_window_size:, :]

        if self.quantize_kv:
            k_int8, k_scale = _quantize_tensor(k_out)
            v_int8, v_scale = _quantize_tensor(v_out)
            self.key_cache[layer_idx] = k_int8
            self.value_cache[layer_idx] = v_int8
            self.key_scales[layer_idx] = k_scale
            self.value_scales[layer_idx] = v_scale
        else:
            self.key_cache[layer_idx] = k_out
            self.value_cache[layer_idx] = v_out

        if layer_idx == 0:
            self._seen_tokens = k_out.shape[-2]
        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the cached sequence length for the given layer index."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            k = self.key_cache[layer_idx]
            assert k is not None
            return k.shape[-2]
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        """Return max allowed cache sequence length."""
        return self.max_seq_len

    def get_mask_sizes(self, query_length: int = 1, layer_idx: int = 0) -> Tuple[int, int]:
        """Return (past_length + query_length, 0) for HuggingFace attention mask calculation."""
        past_length = self.get_seq_length(layer_idx)
        return past_length + query_length, 0

    def get_memory_usage_bytes(self) -> int:
        """Calculate total memory consumed by stored key and value tensors (and scale factors) in bytes."""
        total_bytes = 0
        for k, v, ks, vs in zip(self.key_cache, self.value_cache, self.key_scales, self.value_scales):
            if k is not None and k.numel() > 0:
                total_bytes += k.numel() * k.element_size()
            if v is not None and v.numel() > 0:
                total_bytes += v.numel() * v.element_size()
            if ks is not None and ks.numel() > 0:
                total_bytes += ks.numel() * ks.element_size()
            if vs is not None and vs.numel() > 0:
                total_bytes += vs.numel() * vs.element_size()
        return total_bytes

    def reset(self) -> None:
        """Clear all cached key and value tensors."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.key_scales.clear()
        self.value_scales.clear()
        self._seen_tokens = 0

    def rollback(self, num_tokens: int) -> None:
        """Roll back (truncate) the last `num_tokens` from all cached layer key and value tensors."""
        if num_tokens <= 0:
            return

        for i in range(len(self.key_cache)):
            k = self.key_cache[i]
            v = self.value_cache[i]
            ks = self.key_scales[i]
            vs = self.value_scales[i]

            if k is not None and v is not None:
                seq_len = k.shape[-2]
                keep_len = max(0, seq_len - num_tokens)
                if keep_len == 0:
                    self.key_cache[i] = None
                    self.value_cache[i] = None
                    self.key_scales[i] = None
                    self.value_scales[i] = None
                else:
                    self.key_cache[i] = k[..., :keep_len, :]
                    self.value_cache[i] = v[..., :keep_len, :]
                    if ks is not None:
                        self.key_scales[i] = ks[..., :keep_len, :]
                    if vs is not None:
                        self.value_scales[i] = vs[..., :keep_len, :]

        if len(self.key_cache) > 0 and self.key_cache[0] is not None:
            self._seen_tokens = self.key_cache[0].shape[-2]
        else:
            self._seen_tokens = 0


class KVCacheManager:
    """Manages allocation, retrieval, and deallocation of per-request KV cache states."""

    def __init__(self, max_total_memory_bytes: Optional[int] = None) -> None:
        self._caches: Dict[str, KVCacheState] = {}
        self.max_total_memory_bytes = max_total_memory_bytes

    def allocate(self, request_id: str, max_seq_len: int = 2048) -> KVCacheState:
        """Allocate a new KV cache state for a request ID."""
        if request_id in self._caches:
            return self._caches[request_id]

        cache = KVCacheState(max_seq_len=max_seq_len)
        self._caches[request_id] = cache
        return cache

    def get(self, request_id: str) -> Optional[KVCacheState]:
        """Retrieve existing KV cache state for request ID, or None if not found."""
        return self._caches.get(request_id)

    def free(self, request_id: str) -> bool:
        """Deallocate and remove KV cache state for request ID."""
        if request_id in self._caches:
            self._caches[request_id].reset()
            del self._caches[request_id]
            return True
        return False

    def get_total_memory_usage_bytes(self) -> int:
        """Calculate aggregate memory usage across all active request caches."""
        return sum(cache.get_memory_usage_bytes() for cache in self._caches.values())

    def active_requests_count(self) -> int:
        """Return count of active request caches."""
        return len(self._caches)

    def clear(self) -> None:
        """Free all active request caches."""
        for request_id in list(self._caches.keys()):
            self.free(request_id)
