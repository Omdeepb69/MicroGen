"""PyTorch inference backend implementation."""

from typing import Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from microgen.devices import Device, get_device
from microgen.backends.base import InferenceBackend


class PyTorchBackend(InferenceBackend):
    """PyTorch execution backend for Causal LM inference across CPU and CUDA devices."""

    def __init__(self, device: Optional[Device] = None) -> None:
        self._device: Device = device if device is not None else get_device("cpu")
        self._model: Optional[PreTrainedModel] = None
        self._model_name: Optional[str] = None

    @property
    def device(self) -> Device:
        """Return the target hardware device."""
        return self._device

    @property
    def model(self) -> Optional[PreTrainedModel]:
        """Return loaded HuggingFace model instance."""
        return self._model

    def load_model(
        self,
        model_name_or_path: str,
        model_instance: Optional[PreTrainedModel] = None,
    ) -> None:
        """Load causal LM weights onto target hardware device."""
        if model_instance is not None:
            self._model = model_instance.to(self._device.torch_device)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(
                self._device.torch_device
            )

        self._model.eval()
        self._model_name = model_name_or_path

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Perform initial prompt prefill pass on hardware device."""
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        input_ids = self._device.to_device(input_ids)
        if attention_mask is not None:
            attention_mask = self._device.to_device(attention_mask)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        updated_cache = getattr(outputs, "past_key_values", cache)
        return logits, updated_cache

    def decode(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Perform single-token decode pass with KV cache on hardware device."""
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        token_ids = self._device.to_device(token_ids)
        if attention_mask is not None:
            attention_mask = self._device.to_device(attention_mask)

        with torch.no_grad():
            outputs = self._model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        updated_cache = getattr(outputs, "past_key_values", cache)
        return logits, updated_cache

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Sample next token from logits tensor (greedy decoding or temperature/top-k/top-p)."""
        if temperature <= 0.0 or (top_k == 0 and top_p == 0.0 and temperature == 1.0):
            # Deterministic greedy decoding
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-5)

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return device memory metrics."""
        return self._device.get_memory_info()
