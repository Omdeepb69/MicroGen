"""Unit tests for microgen.sdk.engine module."""

import pytest
from microgen.sdk.engine import LLMEngine
from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.backends.parallel import TensorParallelPyTorchBackend

TINY_MODEL = "sshleifer/tiny-gpt2"


def test_llm_engine_default_pytorch_backend():
    engine = LLMEngine.from_pretrained(TINY_MODEL, device="cpu")
    assert isinstance(engine.backend, PyTorchBackend)
    assert engine.model_name == TINY_MODEL
    assert engine.device.name == "cpu"


def test_llm_engine_quantized_backend():
    engine = LLMEngine.from_pretrained(TINY_MODEL, quantize="int8", device="cpu")
    assert isinstance(engine.backend, QuantizedPyTorchBackend)
    assert engine.backend.quant_type == "int8"


def test_llm_engine_tensor_parallel_backend():
    engine = LLMEngine.from_pretrained(TINY_MODEL, tensor_parallel_size=2, device="cpu")
    assert isinstance(engine.backend, TensorParallelPyTorchBackend)
    assert engine.backend.world_size == 2


def test_llm_engine_validation_quantize_and_tp_conflict():
    with pytest.raises(ValueError, match="Combined INT8 quantization and Tensor Parallelism is currently unvalidated"):
        LLMEngine.from_pretrained(TINY_MODEL, quantize="int8", tensor_parallel_size=2, device="cpu")


def test_llm_engine_validation_invalid_quantize():
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        LLMEngine.from_pretrained(TINY_MODEL, quantize="invalid_mode", device="cpu")


def test_llm_engine_validation_invalid_tp_size():
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        LLMEngine.from_pretrained(TINY_MODEL, tensor_parallel_size=0, device="cpu")


def test_llm_engine_generate_full():
    engine = LLMEngine.from_pretrained(TINY_MODEL, device="cpu")
    output = engine.generate("Hello world", max_new_tokens=5, stream=False)
    assert isinstance(output, str)


def test_llm_engine_generate_stream():
    engine = LLMEngine.from_pretrained(TINY_MODEL, device="cpu")
    tokens = list(engine.generate("Hello world", max_new_tokens=5, stream=True))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_llm_engine_get_memory_usage():
    engine = LLMEngine.from_pretrained(TINY_MODEL, device="cpu")
    mem = engine.get_memory_usage()
    assert isinstance(mem, dict)
