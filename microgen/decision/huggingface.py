"""HuggingFace Transformers adapter for the Decision Engine.

Provides TransformersDecisionModel which satisfies the DecisionModel protocol
by wrapping a standard AutoModelForCausalLM and PreTrainedTokenizer.
"""

from typing import Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from microgen.decision.adapters import DecisionModel


class TransformersDecisionModel(DecisionModel):
    """Adapter for HuggingFace Transformers causal LMs."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = str(model.device)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def device_type(self) -> str:
        # device might be "cuda:0", we return just "cuda" or the whole string.
        # Actually PyTorch torch.device() handles "cuda:0" fine.
        return self._device

    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        """Run the initial prefill pass and return (logits, updated_cache)."""
        kwargs = {
            "input_ids": input_ids,
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
            
        outputs = self._model(**kwargs)
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def decode(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        """Run a single-token decode pass and return (logits, updated_cache)."""
        kwargs = {
            "input_ids": token_ids,
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
            
        outputs = self._model(**kwargs)
        return outputs.logits, outputs.past_key_values
