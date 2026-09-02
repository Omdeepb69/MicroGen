"""PyTorch inference backend implementation."""

from typing import Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from microgen.devices import Device, get_device
from microgen.backends.base import InferenceBackend

# Compatibility fix for transformers SDPA causal mask handling with tensor kv_length
try:
    import transformers.masking_utils
    _orig_ignore_causal_mask_sdpa = transformers.masking_utils._ignore_causal_mask_sdpa
    _orig_prepare_padding_mask = transformers.masking_utils.prepare_padding_mask
    _orig_sdpa_mask = transformers.masking_utils.sdpa_mask

    def _coerce_kv_length(val: Any) -> Any:
        if isinstance(val, torch.Tensor) and val.dtype in (torch.int32, torch.int64, torch.long):
            return int(val.shape[-1]) if val.ndim > 0 else int(val.item())
        return val

    def _safe_ignore_causal_mask_sdpa(*args: Any, **kwargs: Any) -> bool:
        args_list = list(args)
        if "kv_length" in kwargs:
            kwargs["kv_length"] = _coerce_kv_length(kwargs["kv_length"])
        elif len(args_list) >= 3:
            args_list[2] = _coerce_kv_length(args_list[2])
        return _orig_ignore_causal_mask_sdpa(*args_list, **kwargs)

    def _safe_prepare_padding_mask(*args: Any, **kwargs: Any) -> Optional[torch.Tensor]:
        args_list = list(args)
        if "kv_length" in kwargs:
            kwargs["kv_length"] = _coerce_kv_length(kwargs["kv_length"])
        elif len(args_list) >= 2:
            args_list[1] = _coerce_kv_length(args_list[1])
        return _orig_prepare_padding_mask(*args_list, **kwargs)

    def _safe_sdpa_mask(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if "kv_length" in kwargs:
            kwargs["kv_length"] = _coerce_kv_length(kwargs["kv_length"])
        elif len(args_list) >= 3:
            args_list[2] = _coerce_kv_length(args_list[2])
        return _orig_sdpa_mask(*args_list, **kwargs)

    transformers.masking_utils._ignore_causal_mask_sdpa = _safe_ignore_causal_mask_sdpa
    transformers.masking_utils.prepare_padding_mask = _safe_prepare_padding_mask
    transformers.masking_utils.sdpa_mask = _safe_sdpa_mask
    if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS") and "sdpa" in transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS:
        transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _safe_sdpa_mask
except (ImportError, AttributeError):
    pass


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
        """Perform single token decode step using cached KV states."""
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
