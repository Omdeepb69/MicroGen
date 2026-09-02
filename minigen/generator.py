"""Single-step forward pass and autoregressive generation logic."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from minigen.cache import SimpleKVCache


def generate_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    cache: Optional[SimpleKVCache] = None,
) -> Tuple[torch.Tensor, Optional[SimpleKVCache]]:
    """Perform a single forward pass step through the model.

    Args:
        model: HuggingFace Causal LM instance (e.g., GPT2LMHeadModel).
        input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            When cache is active and populated, seq_len is typically 1 (the latest token).
        cache: Optional SimpleKVCache instance for caching key/value states.

    Returns:
        Tuple containing:
            - next_token_logits: Tensor of shape (batch_size, vocab_size) representing logits for the next position.
            - cache: The updated SimpleKVCache instance, or None if caching is disabled.
    """
    use_cache = cache is not None

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=use_cache,
        )

    logits = outputs.logits
    next_token_logits = logits[:, -1, :]

    return next_token_logits, cache


def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively using greedy decoding.

    Args:
        model: HuggingFace Causal LM instance.
        input_ids: Tensor of shape (batch_size, seq_len) containing prompt input IDs.
        max_new_tokens: Maximum number of new tokens to generate.
        use_cache: Whether to use per-layer SimpleKVCache (True) or recompute full sequence (False).
        eos_token_id: Optional EOS token ID to stop generation early if produced.

    Returns:
        Tensor of shape (batch_size, seq_len + generated_tokens) containing prompt + generated token IDs.
    """
    if max_new_tokens <= 0:
        return input_ids

    current_ids = input_ids.clone()

    if use_cache:
        cache = SimpleKVCache()
        # Step 1: Pass full prompt to populate initial key-value cache
        next_token_logits, cache = generate_step(model, current_ids, cache=cache)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token_id], dim=1)

        if eos_token_id is not None and (next_token_id == eos_token_id).all():
            return current_ids

        # Steps 2 to max_new_tokens: Pass only latest single token with cache
        step_input = next_token_id
        for _ in range(1, max_new_tokens):
            next_token_logits, cache = generate_step(model, step_input, cache=cache)
            step_input = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, step_input], dim=1)

            if eos_token_id is not None and (step_input == eos_token_id).all():
                break

        return current_ids
    else:
        # Recompute full sequence at every step
        for _ in range(max_new_tokens):
            next_token_logits, _ = generate_step(model, current_ids, cache=None)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token_id], dim=1)

            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break

        return current_ids
