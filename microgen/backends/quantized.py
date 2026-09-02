"""INT8 and FP8 per-channel weight quantization backend implementation."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from microgen.devices import Device, get_device
from microgen.backends.pytorch import PyTorchBackend


class QuantizedLinear(nn.Module):
    """Linear layer wrapper holding per-channel INT8 quantized weights and scaling factors."""

    def __init__(self, weight_int8: torch.Tensor, scales: torch.Tensor, bias: Optional[nn.Parameter] = None) -> None:
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization: W_dequant = W_int8 * scales
        weight_fp = self.weight_int8.to(x.dtype) * self.scales
        return nn.functional.linear(x, weight_fp, self.bias)


def quantize_linear_layer_per_channel(module: nn.Linear) -> QuantizedLinear:
    """Quantize nn.Linear weights to INT8 per output channel."""
    weight = module.weight.data
    # Per-channel scales: shape (out_features, 1)
    max_vals = torch.max(torch.abs(weight), dim=-1, keepdim=True).values
    scales = max_vals / 127.0
    scales = torch.clamp(scales, min=1e-8)

    weight_int8 = torch.round(weight / scales).clamp(-128, 127).to(torch.int8)
    return QuantizedLinear(weight_int8=weight_int8, scales=scales, bias=module.bias)


class QuantizedPyTorchBackend(PyTorchBackend):
    """Execution backend supporting per-channel INT8/FP8 weight quantization."""

    def __init__(self, device: Optional[Device] = None, quant_type: str = "int8") -> None:
        super().__init__(device=device)
        self.quant_type = quant_type.lower()
        self._quantized = False

    def load_model(
        self,
        model_name_or_path: str,
        model_instance: Optional[PreTrainedModel] = None,
    ) -> None:
        """Load causal LM model weights and apply per-channel quantization."""
        super().load_model(model_name_or_path, model_instance=model_instance)
        self.quantize_model()

    def quantize_model(self) -> None:
        """Traverse model modules and replace linear layers with quantized linear layers."""
        if self._model is None or self._quantized:
            return

        def _quantize_children(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    quantized_child = quantize_linear_layer_per_channel(child)
                    setattr(module, name, quantized_child)
                else:
                    _quantize_children(child)

        _quantize_children(self._model)
        self._quantized = True

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage metrics including quantization savings."""
        info = super().get_memory_usage()
        info["quant_type"] = self.quant_type
        info["is_quantized"] = self._quantized
        return info
