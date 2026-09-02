"""Inference backend protocol abstraction."""

from typing import Protocol, Tuple, Optional, Any, Dict, runtime_checkable
import torch
from microgen.devices.base import Device


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol for LLM inference runtime backends (PyTorch, ONNX, etc.)."""

    @property
    def device(self) -> Device:
        """Return the target hardware device for this backend."""
        ...

    def load_model(self, model_name_or_path: str) -> None:
        """Load causal LM model weights onto target hardware device."""
        ...

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Perform initial prompt prefill pass, returning (logits, updated_cache)."""
        ...

    def decode(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Perform single-token decode pass, returning (logits, updated_cache)."""
        ...

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory consumption statistics for active model & backend."""
        ...
