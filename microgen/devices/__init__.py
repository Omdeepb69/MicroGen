"""Device factory and module exports."""

from microgen.devices.base import Device
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice


def get_device(device_str: str = "cpu") -> Device:
    """Factory function returning appropriate Device instance for a given string."""
    normalized = device_str.lower().strip()
    if normalized == "cpu":
        return CPUDevice()
    elif normalized.startswith("cuda"):
        index = 0
        if ":" in normalized:
            try:
                index = int(normalized.split(":")[1])
            except ValueError:
                index = 0
        return CUDADevice(device_index=index)
    else:
        raise ValueError(f"Unsupported device name: '{device_str}'. Expected 'cpu' or 'cuda[:idx]'.")


__all__ = ["Device", "CPUDevice", "CUDADevice", "get_device"]
