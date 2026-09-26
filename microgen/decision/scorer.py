"""Candidate logit scorer for the Decision Engine.

This module contains pure scoring functions — they are stateless and operate
only on tensors and schema types. The DecisionEngine in engine.py orchestrates
the model calls and passes results here.

Task 1.2 scope: single-token candidate scoring from prefill logits.
Task 2.1 scope: multi-token sequence scoring via an injected decode callable.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Tuple

import torch

from microgen.decision.schema import CandidateStats, DecisionResult


def _extract_last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the logit vector at the final sequence position.

    Args:
        logits: Raw model output, shape [batch, seq_len, vocab_size] or
            [batch, vocab_size] (some backends collapse the seq dim).

    Returns:
        1-D tensor of shape [vocab_size].
    """
    if logits.ndim == 3:
        return logits[0, -1, :]
    # [1, vocab_size] or [vocab_size]
    return logits.view(-1)


def score_single_token_candidates(
    last_logits: torch.Tensor,
    candidates: list[CandidateStats],
    alpha: float = 1.0,
) -> DecisionResult:
    """Produce a DecisionResult by scoring single-token candidates from prefill logits.

    All candidates must resolve to exactly one token ID.  The function applies
    softmax only over the candidate subset (not the full vocabulary), which is
    the correct formulation for constrained decision scoring.

    Args:
        last_logits: 1-D float tensor of shape [vocab_size] at the final
            prompt position, as returned by `_extract_last_token_logits`.
        candidates: List of CandidateStats, each with exactly one token ID.
        alpha: Length normalization penalty (unused here since length is 1,
            but included for API parity with sequence scoring).

    Raises:
        ValueError: If any candidate has token_count != 1.  Multi-token
            candidates require sequence scoring (Task 2.1).
        ValueError: If the candidates list is empty.

    Returns:
        DecisionResult with probabilities, top_probability, and calibrated=False.
        entropy is left as None — the entropy module fills it in (Task 3.1).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty.")

    multi = [c for c in candidates if c.token_count != 1]
    if multi:
        names = [c.text for c in multi]
        raise ValueError(
            f"Single-token scoring received multi-token candidates: {names}. "
            "Use score_sequence_candidates() for multi-token options."
        )

    token_ids = [c.token_ids[0] for c in candidates]
    candidate_logits = last_logits[token_ids]
    probs = torch.softmax(candidate_logits, dim=0)

    probabilities: dict[str, float] = {
        c.text: float(p) for c, p in zip(candidates, probs)
    }

    best_text = max(probabilities, key=probabilities.__getitem__)
    top_probability = probabilities[best_text]

    updated_stats = []
    for c, p, logit in zip(candidates, probs, candidate_logits):
        raw = logit.item()  # The raw logit (log-prob before softmax)
        # S_alpha for single token is just raw / 1^alpha = raw.
        updated_stats.append(
            dataclasses.replace(c, raw_logprob=raw, normalized_score=raw)
        )

    return DecisionResult(
        choice=best_text,
        probabilities=probabilities,
        top_probability=top_probability,
        candidate_stats=updated_stats,
    )


def _sequence_log_prob(
    first_logits: torch.Tensor,
    token_ids: list[int],
    decode_fn: Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]],
    cache: Any,
    device: torch.device,
) -> Tuple[float, Any]:
    """Compute the sum of log-probabilities for a token sequence.

    For a candidate with tokens [t0, t1, t2]:
        log P(t0 | context) is read from first_logits (the prefill output).
        log P(t1 | context, t0) is computed via decode_fn(t0, cache).
        log P(t2 | context, t0, t1) is computed via decode_fn(t1, extended_cache).

    The caller is responsible for rolling back the cache after this function
    returns so that the next candidate starts from the same prefill state.

    Args:
        first_logits: 1-D float tensor [vocab_size] at the final prompt position.
        token_ids: Ordered list of token IDs for this candidate (length >= 1).
        decode_fn: Callable matching (token_ids_tensor, cache) → (logits, cache).
        cache: KV cache state after prefill, shared starting point for all candidates.
        device: Target torch device.

    Returns:
        (total_log_prob, updated_cache) — the cache has been advanced by
        len(token_ids) - 1 decode steps.
    """
    log_prob = math.log(torch.softmax(first_logits, dim=-1)[token_ids[0]].item() + 1e-45)

    cur_cache = cache
    for i, next_id in enumerate(token_ids[1:], start=1):
        prev_id = token_ids[i - 1]
        prev_token = torch.tensor([[prev_id]], device=device)
        step_logits, cur_cache = decode_fn(prev_token, cur_cache)
        step_last = _extract_last_token_logits(step_logits)
        log_prob += math.log(
            torch.softmax(step_last, dim=-1)[next_id].item() + 1e-45
        )

    return log_prob, cur_cache


def score_sequence_candidates(
    last_logits: torch.Tensor,
    candidates: list[CandidateStats],
    decode_fn: Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]],
    prefill_cache: Any,
    device: torch.device,
    alpha: float = 1.0,
) -> DecisionResult:
    """Score candidates of any token length using accumulated log-probabilities.

    Handles both single-token and multi-token candidates uniformly.  For
    single-token candidates the score is simply log P(t | context) from the
    prefill logits.  For multi-token candidates it accumulates log-probabilities
    across sequential decode passes, with the caller rolling back the cache
    between candidates via the returned cache state.

    The final scores are converted to a proper probability distribution via
    softmax over the length-normalized values (S_alpha).

    Args:
        last_logits: 1-D float tensor [vocab_size] at the final prompt position.
        candidates: List of CandidateStats (any mix of token counts).
        decode_fn: Callable matching (token_ids_tensor, cache) → (logits, cache).
        prefill_cache: KV cache state after prefill; must support rollback().
        device: Target torch device for token tensors.
        alpha: Length normalization penalty. S_alpha = sum(log_P) / (len^alpha).
            alpha=0.0 means no normalization; alpha=1.0 means mean log-prob.

    Raises:
        ValueError: If the candidates list is empty.

    Returns:
        DecisionResult with probabilities, top_probability, and calibrated=False.
        entropy is left as None.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty.")

    normalized_scores: list[float] = []
    updated_stats: list[CandidateStats] = []
    
    for candidate in candidates:
        initial_cache_len = prefill_cache.get_seq_length()
        log_prob, _ = _sequence_log_prob(
            first_logits=last_logits,
            token_ids=candidate.token_ids,
            decode_fn=decode_fn,
            cache=prefill_cache,
            device=device,
        )
        
        # Length normalization: S_alpha = log_prob / (|Y|^alpha)
        length_penalty = math.pow(candidate.token_count, alpha)
        normalized = log_prob / length_penalty
        
        normalized_scores.append(normalized)
        updated_stats.append(
            dataclasses.replace(candidate, raw_logprob=log_prob, normalized_score=normalized)
        )
        
        # Roll back the decode steps for this candidate so the next candidate
        # starts from the same prefill state.
        decode_steps = candidate.token_count - 1
        if decode_steps > 0:
            prefill_cache.rollback(decode_steps)
        assert prefill_cache.get_seq_length() == initial_cache_len, (
            "Cache rollback failed — prefill state was not restored correctly."
        )

    # Convert normalized scores to a proper distribution via softmax.
    score_tensor = torch.tensor(normalized_scores, dtype=torch.float32)
    probs = torch.softmax(score_tensor, dim=0)

    probabilities: dict[str, float] = {
        c.text: float(p) for c, p in zip(candidates, probs)
    }
    best_text = max(probabilities, key=probabilities.__getitem__)
    top_probability = probabilities[best_text]

    return DecisionResult(
        choice=best_text,
        probabilities=probabilities,
        top_probability=top_probability,
        candidate_stats=updated_stats,
    )
