"""High-level fluent LLMEngine SDK wrapper with automated backend dispatch and validation."""

from typing import Dict, Any, Tuple, Optional, Union, Iterator, List
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from microgen.devices import Device, get_device
from microgen.backends.base import InferenceBackend
from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.backends.parallel import TensorParallelPyTorchBackend
from microgen.runtime.kv_cache import KVCacheState


class LLMEngine:
    """High-level developer-facing LLM engine interface.

    Provides 1-line model loading (`LLMEngine.from_pretrained`), automated backend
    selection (PyTorch, INT8 Quantized, or Multi-GPU Tensor Parallel), request prefill/decode,
    and streaming token generation.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self._model_name = model_name

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    @property
    def backend(self) -> InferenceBackend:
        """Return the underlying inference backend instance."""
        return self._backend

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Return loaded HuggingFace tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> Device:
        """Return active hardware device abstraction."""
        return self._backend.device

    @property
    def model_name(self) -> str:
        """Return model identifier path."""
        return self._model_name

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quantize: Optional[str] = None,
        tensor_parallel_size: int = 1,
        device: Optional[Union[str, Device]] = None,
        trust_remote_code: bool = True,
    ) -> "LLMEngine":
        """Factory method to load an LLM with automatic backend dispatch and validation.

        Args:
            model_name_or_path: HuggingFace model ID or local directory path.
            quantize: Quantization mode ('int8' or 'fp8'). None for full precision.
            tensor_parallel_size: Number of GPU ranks for Megatron 1D tensor parallelism.
            device: Target hardware device string ('cpu', 'cuda', 'cuda:0') or Device object.
            trust_remote_code: Whether to allow remote custom code execution.

        Returns:
            Initialized LLMEngine instance ready for generation.
        """
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")

        if quantize is not None:
            quantize_lower = quantize.lower()
            if quantize_lower not in ("int8", "fp8"):
                raise ValueError(f"Unsupported quantization mode '{quantize}'. Supported: 'int8', 'fp8'")
            if tensor_parallel_size > 1:
                raise ValueError(
                    "Combined INT8 quantization and Tensor Parallelism is currently unvalidated in MicroGen. "
                    "Please specify either quantize='int8' OR tensor_parallel_size > 1."
                )

        # Resolve primary device
        if device is None:
            target_device = get_device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            target_device = get_device(device)
        else:
            target_device = device

        # Dispatch backend instantiation based on parameters
        if tensor_parallel_size > 1:
            if target_device.name == "cuda" and torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if tensor_parallel_size > num_gpus:
                    devices = [get_device(f"cuda:{i % num_gpus}") for i in range(tensor_parallel_size)]
                else:
                    devices = [get_device(f"cuda:{i}") for i in range(tensor_parallel_size)]
            else:
                devices = [target_device] * tensor_parallel_size

            backend = TensorParallelPyTorchBackend(
                world_size=tensor_parallel_size,
                devices=devices,
            )
        elif quantize is not None:
            backend = QuantizedPyTorchBackend(
                device=target_device,
                quant_type=quantize.lower(),
            )
        else:
            backend = PyTorchBackend(device=target_device)

        # Load weights and tokenizer
        backend.load_model(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        return cls(backend=backend, tokenizer=tokenizer, model_name=model_name_or_path)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        stream: bool = False,
        temperature: float = 1.0,
    ) -> Union[str, Iterator[str]]:
        """Generate text continuation for prompt with optional token streaming.

        Args:
            prompt: Input text prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            stream: If True, returns an iterator yielding generated text tokens.
            temperature: Sampling temperature scaling factor.

        Returns:
            Generated text string (if stream=False) or Iterator[str] (if stream=True).
        """
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device.torch_device)

        if stream:
            return self._generate_stream(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        return self._generate_full(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _extract_last_logits(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if logits.ndim == 3:
            step_logits = logits[0, -1, :]
        else:
            step_logits = logits[0, :]
        return step_logits / max(temperature, 1e-5)

    def _generate_full(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        cache = KVCacheState()
        logits, cache = self._backend.prefill(input_ids, cache=cache)

        generated_tokens: List[int] = []
        cur_logits = logits

        for _ in range(max_new_tokens):
            scaled_logits = self._extract_last_logits(cur_logits, temperature)
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_tokens.append(next_token_id)
            if next_token_id == self._tokenizer.eos_token_id:
                break

            step_tensor = torch.tensor([[next_token_id]], device=self.device.torch_device)
            cur_logits, cache = self._backend.decode(step_tensor, cache=cache)

        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        cache = KVCacheState()
        logits, cache = self._backend.prefill(input_ids, cache=cache)

        cur_logits = logits

        for _ in range(max_new_tokens):
            scaled_logits = self._extract_last_logits(cur_logits, temperature)
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            token_text = self._tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield token_text

            if next_token_id == self._tokenizer.eos_token_id:
                break

            step_tensor = torch.tensor([[next_token_id]], device=self.device.torch_device)
            cur_logits, cache = self._backend.decode(step_tensor, cache=cache)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return active device and backend memory consumption stats."""
        return self._backend.get_memory_usage()
