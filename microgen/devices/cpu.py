"""CPU hardware device implementation."""

import os
from typing import Dict
import torch
from microgen.devices.base import Device


class CPUDevice(Device):
    """CPU hardware device implementation using PyTorch CPU and standard OS memory tracking."""

    def __init__(self) -> None:
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def torch_device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        # Host execution is synchronous in PyTorch for CPU operations
        pass

    def get_memory_info(self) -> Dict[str, int]:
        total = 0
        free = 0
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split(":")
                        if parts[0] == "MemTotal":
                            total = int(parts[1].split()[0]) * 1024
                        elif parts[0] == "MemAvailable":
                            free = int(parts[1].split()[0]) * 1024
            except OSError:
                pass
        allocated = max(0, total - free)
        return {
            "total_bytes": total,
            "allocated_bytes": allocated,
            "free_bytes": free,
        }

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)
