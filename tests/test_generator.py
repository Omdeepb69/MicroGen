"""Unit tests for generate_step and generate in minigen/generator.py."""

import torch
from transformers import AutoModelForCausalLM
from minigen.cache import SimpleKVCache
from minigen.generator import generate, generate_step

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_generate_step_uncached() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompt_ids = torch.tensor([[100, 200, 300]])
    logits, cache = generate_step(model, prompt_ids, cache=None)

    assert cache is None
    assert logits.ndim == 2
    assert logits.shape[0] == 1
    assert logits.shape[1] == model.config.vocab_size


def test_generate_step_cached_and_logits_equivalence() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompt_ids = torch.tensor([[100, 200, 300]])
    next_id = torch.tensor([[400]])

    # 1. Full sequence forward pass without cache (ground truth)
    full_ids = torch.cat([prompt_ids, next_id], dim=1)
    gt_logits, _ = generate_step(model, full_ids, cache=None)

    # 2. Step-by-step forward pass with SimpleKVCache
    cache = SimpleKVCache()
    prompt_logits, cache = generate_step(model, prompt_ids, cache=cache)
    assert cache is not None
    assert cache.get_seq_length(0) == 3

    cached_next_logits, cache = generate_step(model, next_id, cache=cache)
    assert cache is not None
    assert cache.get_seq_length(0) == 4

    # 3. Assert logits from cached step match uncached full forward pass
    torch.testing.assert_close(cached_next_logits, gt_logits, rtol=1e-4, atol=1e-4)


def test_generate_cached_vs_uncached_equivalence() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompt_ids = torch.tensor([[50, 100, 150, 200]])
    max_new_tokens = 15

    cached_output = generate(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True)
    uncached_output = generate(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=False)

    assert cached_output.shape == (1, 4 + max_new_tokens)
    assert uncached_output.shape == (1, 4 + max_new_tokens)
    assert torch.equal(cached_output, uncached_output)


def test_generate_hf_model_generate_equivalence() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompt_ids = torch.tensor([[50, 100, 150, 200]])
    max_new_tokens = 10

    custom_output = generate(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True)

    with torch.no_grad():
        hf_output = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
        )

    assert torch.equal(custom_output, hf_output)


def test_generate_zero_max_new_tokens() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompt_ids = torch.tensor([[1, 2, 3]])
    out = generate(model, prompt_ids, max_new_tokens=0, use_cache=True)
    assert torch.equal(out, prompt_ids)
