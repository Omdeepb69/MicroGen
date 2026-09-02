"""Unit tests for microgen hardware device abstractions and backend protocol."""

import pytest
import torch
from microgen.devices import Device, CPUDevice, CUDADevice, get_device
from microgen.backends import InferenceBackend


def test_cpu_device_properties_and_operations():
    device = CPUDevice()
    assert device.name == "cpu"
    assert device.torch_device == torch.device("cpu")
    assert device.is_available() is True

    # Test memory info returning expected dict keys
    mem_info = device.get_memory_info()
    assert "total_bytes" in mem_info
    assert "allocated_bytes" in mem_info
    assert "free_bytes" in mem_info
    assert mem_info["total_bytes"] > 0

    # Test tensor transfer
    tensor = torch.tensor([1, 2, 3])
    dev_tensor = device.to_device(tensor)
    assert dev_tensor.device == torch.device("cpu")
    device.synchronize()  # No-op on CPU


def test_cuda_device_properties():
    device = CUDADevice(device_index=0)
    assert device.name == "cuda:0"
    assert device.torch_device == torch.device("cuda:0")

    # CUDA may or may not be available depending on host hardware
    cuda_avail = torch.cuda.is_available()
    assert device.is_available() == cuda_avail

    mem_info = device.get_memory_info()
    assert "total_bytes" in mem_info
    assert "allocated_bytes" in mem_info
    assert "free_bytes" in mem_info


def test_get_device_factory():
    cpu_dev = get_device("cpu")
    assert isinstance(cpu_dev, CPUDevice)
    assert cpu_dev.name == "cpu"

    cuda_dev = get_device("cuda")
    assert isinstance(cuda_dev, CUDADevice)
    assert cuda_dev.name == "cuda:0"

    cuda1_dev = get_device("cuda:1")
    assert isinstance(cuda1_dev, CUDADevice)
    assert cuda1_dev.name == "cuda:1"

    with pytest.raises(ValueError, match="Unsupported device name"):
        get_device("invalid_device")


def test_backend_protocol_check():
    class DummyBackend:
        @property
        def device(self) -> Device:
            return CPUDevice()

        def load_model(self, model_name_or_path: str) -> None:
            pass

        def prefill(self, input_ids: torch.Tensor, cache=None):
            return input_ids, cache

        def decode(self, token_ids: torch.Tensor, cache=None):
            return token_ids, cache

        def get_memory_usage(self):
            return {}

    dummy = DummyBackend()
    assert isinstance(dummy, InferenceBackend)
