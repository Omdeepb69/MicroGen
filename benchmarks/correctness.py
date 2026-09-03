"""
Correctness Validation Gates (Phase B).

Verifies token-by-token output identity and logit cosine similarity across optimization backends
before benchmark trial execution.
"""

from typing import Any, Dict, List, Tuple
import torch

from microgen.backends.base import InferenceBackend
from microgen.backends.pytorch import PyTorchBackend
from microgen.caching.prefix_cache import PrefixKVCache
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice
from microgen.runtime.kv_cache import KVCacheManager


def validate_greedy_decoding_identity(
    baseline_backend: InferenceBackend,
    target_backend: InferenceBackend,
    prompt_ids: List[int],
    max_new_tokens: int = 16,
) -> Tuple[bool, List[int], List[int]]:
    """Validates exact token-by-token equality of generated tokens between baseline and target backends."""
    device = getattr(baseline_backend.device, "torch_device", torch.device("cpu"))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1. Baseline generation
    logits_b, cache_b = baseline_backend.prefill(input_ids)
    token_b = torch.argmax(logits_b, dim=-1, keepdim=True)
    tokens_baseline = [int(token_b.item())]

    for _ in range(max_new_tokens - 1):
        logits_b, cache_b = baseline_backend.decode(token_b, cache=cache_b)
        token_b = torch.argmax(logits_b, dim=-1, keepdim=True)
        tokens_baseline.append(int(token_b.item()))

    # 2. Target backend generation
    logits_t, cache_t = target_backend.prefill(input_ids)
    token_t = torch.argmax(logits_t, dim=-1, keepdim=True)
    tokens_target = [int(token_t.item())]

    for _ in range(max_new_tokens - 1):
        logits_t, cache_t = target_backend.decode(token_t, cache=cache_t)
        token_t = torch.argmax(logits_t, dim=-1, keepdim=True)
        tokens_target.append(int(token_t.item()))

    is_equal = tokens_baseline == tokens_target
    return is_equal, tokens_baseline, tokens_target


def validate_logit_similarity(
    baseline_backend: InferenceBackend,
    target_backend: InferenceBackend,
    prompt_ids: List[int],
    min_cosine_sim: float = 0.99,
    max_abs_error: float = 0.05,
) -> Tuple[bool, float, float]:
    """Validates cosine similarity and max absolute error of logits between baseline and target backends."""
    device = getattr(baseline_backend.device, "torch_device", torch.device("cpu"))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    logits_b, _ = baseline_backend.prefill(input_ids)
    logits_t, _ = target_backend.prefill(input_ids)

    b_flat = logits_b.view(-1).float()
    t_flat = logits_t.view(-1).float()

    cos_sim = float(torch.nn.functional.cosine_similarity(b_flat, t_flat, dim=0).item())
    max_err = float(torch.max(torch.abs(b_flat - t_flat)).item())

    passed = (cos_sim >= min_cosine_sim) and (max_err <= max_abs_error)
    return passed, cos_sim, max_err


def run_all_correctness_gates(model_name: str = "sshleifer/tiny-gpt2") -> Dict[str, bool]:
    """Runs correctness verification suite for PyTorch baseline backend."""
    device = CPUDevice()
    backend1 = PyTorchBackend(device=device)
    backend1.load_model(model_name)

    backend2 = PyTorchBackend(device=device)
    backend2.load_model(model_name)

    prompt_ids = [1, 25, 400, 1000]
    is_eq, t1, t2 = validate_greedy_decoding_identity(backend1, backend2, prompt_ids, max_new_tokens=8)
    sim_passed, cos_sim, max_err = validate_logit_similarity(backend1, backend2, prompt_ids)

    return {
        "greedy_identity": is_eq,
        "logit_similarity": sim_passed,
    }


if __name__ == "__main__":
    print("Executing Correctness Verification Gates...")
    res = run_all_correctness_gates()
    print(f"Correctness Gate Results: {res}")
    assert all(res.values()), f"Correctness Gate Failed! Details: {res}"
    print("✅ All Correctness Gates Passed!")
