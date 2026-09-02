"""CUDA GPU hardware device implementation."""

from typing import Dict
import torch
from microgen.devices.base import Device


class CUDADevice(Device):
    """CUDA GPU hardware device implementation using PyTorch CUDA runtime."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._device = torch.device(f"cuda:{device_index}")

    @property
    def name(self) -> str:
        return f"cuda:{self._device_index}"

    @property
    def torch_device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return torch.cuda.is_available() and self._device_index < torch.cuda.device_count()

    def synchronize(self) -> None:
        if self.is_available():
            torch.cuda.synchronize(self._device)

    def get_memory_info(self) -> Dict[str, int]:
        if not self.is_available():
            return {"total_bytes": 0, "allocated_bytes": 0, "free_bytes": 0}

        total = torch.cuda.get_device_properties(self._device_index).total_memory
        allocated = torch.cuda.memory_allocated(self._device_index)
        reserved = torch.cuda.memory_reserved(self._device_index)
        free = total - allocated
        return {
            "total_bytes": total,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "free_bytes": free,
        }

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)
