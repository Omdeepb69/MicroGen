"""Adapter interfaces to isolate decision logic from specific execution engines."""

from typing import Protocol, Any, Tuple
import torch
from transformers import PreTrainedTokenizer

class DecisionModel(Protocol):
    """Protocol defining the interface required by the Decision Engine.
    
    This isolates the decision-making logic from the full LLMEngine implementation,
    allowing for deterministic test fixtures and easy swapping of underlying inference backends.
    """
    
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used to encode prompts and choices."""
        ...
        
    @property
    def device_type(self) -> str:
        """The string device type (e.g., 'cpu', 'cuda')."""
        ...
        
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        """Run the initial prefill pass and return (logits, updated_cache)."""
        ...
        
    def decode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        """Run a single-token decode pass and return (logits, updated_cache)."""
        ...
