"""Unit tests for Speculative Decoding execution engine."""

import pytest
import torch
from microgen.backends.pytorch import PyTorchBackend
from microgen.devices.cpu import CPUDevice
from microgen.scheduler.speculative import SpeculativeEngine, SpeculativeResult

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_speculative_engine_execution():
    device = CPUDevice()
    draft_backend = PyTorchBackend(device=device)
    target_backend = PyTorchBackend(device=device)

    draft_backend.load_model(MODEL_NAME)
    target_backend.load_model(MODEL_NAME)

    engine = SpeculativeEngine(
        draft_backend=draft_backend,
        target_backend=target_backend,
        num_draft_tokens=3,
    )

    prompt_ids = torch.tensor([[100, 200, 300]])
    result = engine.generate(prompt_ids=prompt_ids, max_new_tokens=6)

    assert isinstance(result, SpeculativeResult)
    assert len(result.output_ids) == 3 + 6  # prompt (3) + max_new_tokens (6)
    assert result.total_drafted_tokens > 0
    assert result.total_accepted_tokens > 0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.num_steps > 0


def test_rejection_sampling_helper():
    from microgen.scheduler.speculative import rejection_sample_token

    # Target probability >= Draft probability -> Must accept
    draft_probs = torch.tensor([[0.2, 0.8]])
    target_probs = torch.tensor([[0.1, 0.9]])
    is_accepted, token_id = rejection_sample_token(draft_probs, target_probs, draft_token=1)
    assert is_accepted is True
    assert token_id == 1


def test_kv_cache_rollback():
    from microgen.runtime.kv_cache import KVCacheState

    cache = KVCacheState(max_seq_len=1024)
    key_tensor = torch.randn(1, 2, 8, 16)  # seq_len=8
    val_tensor = torch.randn(1, 2, 8, 16)
    cache.update(key_tensor, val_tensor, layer_idx=0)
    assert cache.get_seq_length(0) == 8

    # Roll back 3 tokens -> seq_len=5
    cache.rollback(num_tokens=3)
    assert cache.get_seq_length(0) == 5

    # Roll back remaining 5 tokens -> empty cache
    cache.rollback(num_tokens=5)
    assert cache.get_seq_length(0) == 0
