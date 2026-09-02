"""Tensor-parallel multi-GPU PyTorch backend wrapper."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from microgen.devices import Device, get_device
from microgen.backends.pytorch import PyTorchBackend


class ColumnParallelLinear(nn.Module):
    """Linear layer sharded across ranks along output features dimension (dim 0)."""

    def __init__(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor], world_size: int) -> None:
        super().__init__()
        self.world_size = world_size
        out_features, in_features = full_weight.shape
        shard_out = out_features // world_size

        self.shards = nn.ModuleList()
        for r in range(world_size):
            start_idx = r * shard_out
            end_idx = (r + 1) * shard_out if r < world_size - 1 else out_features
            w_shard = full_weight[start_idx:end_idx, :].clone()
            b_shard = full_bias[start_idx:end_idx].clone() if full_bias is not None else None

            linear = nn.Linear(in_features, end_idx - start_idx, bias=full_bias is not None)
            linear.weight.data.copy_(w_shard)
            if b_shard is not None:
                linear.bias.data.copy_(b_shard)
            self.shards.append(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute forward pass on each shard and concatenate along output dim
        outputs = [shard(x) for shard in self.shards]
        return torch.cat(outputs, dim=-1)


class RowParallelLinear(nn.Module):
    """Linear layer sharded across ranks along input features dimension (dim 1) with all-reduce sum."""

    def __init__(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor], world_size: int) -> None:
        super().__init__()
        self.world_size = world_size
        out_features, in_features = full_weight.shape
        shard_in = in_features // world_size

        self.shards = nn.ModuleList()
        for r in range(world_size):
            start_idx = r * shard_in
            end_idx = (r + 1) * shard_in if r < world_size - 1 else in_features
            w_shard = full_weight[:, start_idx:end_idx].clone()

            # Bias is added only once on rank 0 / after all-reduce
            linear = nn.Linear(end_idx - start_idx, out_features, bias=False)
            linear.weight.data.copy_(w_shard)
            self.shards.append(linear)

        if full_bias is not None:
            self.bias = nn.Parameter(full_bias.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input tensor x across in_features dimension for each rank shard
        in_features = x.shape[-1]
        shard_in = in_features // self.world_size

        partial_outputs = []
        for r, shard in enumerate(self.shards):
            start_idx = r * shard_in
            end_idx = (r + 1) * shard_in if r < self.world_size - 1 else in_features
            x_shard = x[..., start_idx:end_idx]
            partial_outputs.append(shard(x_shard))

        # All-reduce sum across rank outputs
        output = sum(partial_outputs)
        if self.bias is not None:
            output = output + self.bias
        return output


class TensorParallelPyTorchBackend(PyTorchBackend):
    """Multi-GPU / multi-rank tensor-parallel execution backend."""

    def __init__(
        self,
        world_size: int = 2,
        devices: Optional[List[Device]] = None,
    ) -> None:
        primary_device = devices[0] if devices is not None and len(devices) > 0 else get_device("cpu")
        super().__init__(device=primary_device)
        self.world_size = max(1, world_size)
        self.devices = devices if devices is not None else [primary_device] * self.world_size
        self._is_parallel = False

    def load_model(
        self,
        model_name_or_path: str,
        model_instance: Optional[PreTrainedModel] = None,
    ) -> None:
        """Load model and partition projection weights into Tensor-Parallel shards."""
        super().load_model(model_name_or_path, model_instance=model_instance)
        if self.world_size > 1:
            self.apply_tensor_parallelism()

    def apply_tensor_parallelism(self) -> None:
        """Partition linear layers in attention/MLP blocks across tensor-parallel ranks."""
        if self._model is None or self._is_parallel:
            return

        def _shard_module(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    # Classify projection type by name: column-parallel vs row-parallel
                    if any(proj in name.lower() for proj in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "fc1", "c_attn"]):
                        sharded_layer = ColumnParallelLinear(
                            child.weight.data, child.bias.data if child.bias is not None else None, self.world_size
                        )
                        setattr(module, name, sharded_layer)
                    elif any(proj in name.lower() for proj in ["o_proj", "down_proj", "fc2", "c_proj"]):
                        sharded_layer = RowParallelLinear(
                            child.weight.data, child.bias.data if child.bias is not None else None, self.world_size
                        )
                        setattr(module, name, sharded_layer)
                else:
                    _shard_module(child)

        _shard_module(self._model)
        self._is_parallel = True

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return aggregate memory metrics across tensor-parallel ranks."""
        info = super().get_memory_usage()
        info["world_size"] = self.world_size
        info["is_tensor_parallel"] = self._is_parallel
        info["ranks"] = [d.name for d in self.devices]
        return info
