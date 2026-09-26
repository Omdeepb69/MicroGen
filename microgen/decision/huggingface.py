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
        self, input_ids: torch.Tensor, cache: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        """Run the initial prefill pass and return (logits, updated_cache)."""
        outputs = self._model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def decode(
        self, token_ids: torch.Tensor, cache: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        """Run a single-token decode pass and return (logits, updated_cache)."""
        # For decoding step, we just provide the new token(s) and the past_key_values cache.
        outputs = self._model(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values
