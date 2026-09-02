"""Base device interface for hardware abstraction."""

from abc import ABC, abstractmethod
from typing import Dict
import torch


class Device(ABC):
    """Abstract base class representing a compute hardware device (CPU or CUDA GPU)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return canonical device identifier ('cpu', 'cuda:0', etc.)."""
        pass

    @property
    @abstractmethod
    def torch_device(self) -> torch.device:
        """Return PyTorch device instance."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the underlying hardware device is available."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize host thread with device operations."""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, int]:
        """Return device memory metadata: {'total_bytes': int, 'allocated_bytes': int, 'free_bytes': int}."""
        pass

    @abstractmethod
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to this hardware device."""
        pass
