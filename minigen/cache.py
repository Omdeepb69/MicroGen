"""Hand-built per-layer Key-Value cache implementation for autoregressive LLM generation."""

from typing import Any, List, Optional, Tuple
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object  # type: ignore[misc,assignment]


class SimpleKVCache(Cache):
    """Per-layer Key-Value Cache for causal language models.

    Manages key and value tensors per transformer layer to avoid recomputing
    past key/value states during autoregressive token generation.
    """

    def __init__(self, num_layers: Optional[int] = None) -> None:
        """Initialize empty cache or pre-allocate slots for a given number of layers."""
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        if num_layers is not None and num_layers > 0:
            self.key_cache = [None] * num_layers
            self.value_cache = [None] * num_layers

    @property
    def num_layers(self) -> int:
        """Return the number of tracked layers."""
        return len(self.key_cache)

    def is_empty(self) -> bool:
        """Check if cache contains any stored tensors."""
        return len(self.key_cache) == 0 or all(k is None for k in self.key_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current cached sequence length for a given layer."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            k = self.key_cache[layer_idx]
            if k is not None:
                return k.shape[-2]
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Get total key/value sequence length and offset for causal mask calculation."""
        past_length = self.get_seq_length(layer_idx)
        return past_length + query_length, 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer by appending new key and value states.

        Args:
            key_states: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
            value_states: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
            layer_idx: Index of the transformer layer being updated.

        Returns:
            Tuple of (concatenated_key, concatenated_value) for layer_idx.
        """
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            cached_k = self.key_cache[layer_idx]
            cached_v = self.value_cache[layer_idx]
            assert cached_k is not None and cached_v is not None
            self.key_cache[layer_idx] = torch.cat([cached_k, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([cached_v, value_states], dim=-2)

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        assert k_out is not None and v_out is not None
        return k_out, v_out

    def update_layer(
        self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicit layer update method for direct tensor manipulation."""
        return self.update(key_states, value_states, layer_idx)

    def to_legacy_cache(self) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """Format cache as a tuple of (key, value) tuples per layer."""
        if self.is_empty():
            return None
        layers = []
        for i in range(len(self.key_cache)):
            k, v = self.key_cache[i], self.value_cache[i]
            if k is not None and v is not None:
                layers.append((k, v))
        return tuple(layers) if layers else None

    def reset(self) -> None:
        """Clear all cached tensors."""
        self.key_cache = []
        self.value_cache = []
