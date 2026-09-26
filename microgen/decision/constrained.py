"""Constrained Candidate Decoding via Trie.

Optimizes decode-free sequence scoring by organizing candidate token sequences
into a prefix trie. This avoids redundant forward passes (cache updates and
logit generation) for candidate sequences that share common prefixes.
"""

from __future__ import annotations

import math
import dataclasses
from typing import Any, Callable, Tuple

import torch

from microgen.decision.schema import CandidateStats, DecisionResult
from microgen.decision.scorer import _extract_last_token_logits


class TrieNode:
    """A node in the candidate prefix trie.

    Attributes:
        token_id: The vocabulary ID of this node's token. -1 for root.
        children: Mapping from token_id to child TrieNode.
        candidate_indices: Indices in the original candidates list that terminate exactly at this node.
    """
    def __init__(self, token_id: int):
        self.token_id: int = token_id
        self.children: dict[int, TrieNode] = {}
        self.candidate_indices: list[int] = []


def build_candidate_trie(candidates: list[CandidateStats]) -> TrieNode:
    """Construct a prefix trie from a list of candidate token sequences."""
    root = TrieNode(token_id=-1)
    for i, candidate in enumerate(candidates):
        node = root
        for token_id in candidate.token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode(token_id)
            node = node.children[token_id]
        node.candidate_indices.append(i)
    return root


def score_trie_candidates(
    last_logits: torch.Tensor,
    candidates: list[CandidateStats],
    decode_fn: Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]],
    prefill_cache: Any,
    device: torch.device,
    alpha: float = 1.0,
) -> DecisionResult:
    """Score candidates using a prefix trie to eliminate redundant decode steps.

    Args:
        last_logits: 1-D float tensor [vocab_size] at the final prompt position.
        candidates: List of CandidateStats (any mix of token counts).
        decode_fn: Callable matching (token_ids_tensor, cache) → (logits, cache).
        prefill_cache: KV cache state after prefill; must support rollback().
        device: Target torch device for token tensors.
        alpha: Length normalization penalty. S_alpha = sum(log_P) / (|Y|^alpha).

    Returns:
        DecisionResult containing choice probabilities and updated CandidateStats.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty.")

    # 1. Build the prefix trie
    root = build_candidate_trie(candidates)

    # 2. Prepare array to store the raw accumulated log_prob for each candidate
    raw_log_probs = [0.0] * len(candidates)
    
    # Pre-calculate softmax over prefill logits for depth-1 nodes
    first_probs = torch.softmax(last_logits, dim=-1)

    # 3. Recursive DFS to score all paths
    def dfs(node: TrieNode, current_log_prob: float, cache: Any):
        # If any candidates end exactly at this node, store their log_prob
        for idx in node.candidate_indices:
            raw_log_probs[idx] = current_log_prob

        # Fast path if no children
        if not node.children:
            return

        # Record cache length before decoding children so we can rollback
        initial_cache_len = cache.get_seq_length()

        for child_token, child_node in node.children.items():
            if node.token_id == -1:
                # Depth-1: log_prob comes from prefill logits, no decoding yet.
                # No cache update required.
                child_p = first_probs[child_token].item() + 1e-45
                child_log_prob = current_log_prob + math.log(child_p)
                dfs(child_node, child_log_prob, cache)
            else:
                # Depth > 1: we must decode from the parent's token
                prev_token = torch.tensor([[node.token_id]], device=device)
                step_logits, next_cache = decode_fn(prev_token, cache)
                
                step_last = _extract_last_token_logits(step_logits)
                child_p = torch.softmax(step_last, dim=-1)[child_token].item() + 1e-45
                child_log_prob = current_log_prob + math.log(child_p)
                
                # Traverse child
                dfs(child_node, child_log_prob, next_cache)
                
                # Rollback cache so the next sibling starts from the correct parent state
                steps_added = next_cache.get_seq_length() - initial_cache_len
                if steps_added > 0:
                    next_cache.rollback(steps_added)

    # Execute DFS
    dfs(root, 0.0, prefill_cache)

    # 4. Length normalization and final softmax
    normalized_scores: list[float] = []
    updated_stats: list[CandidateStats] = []
    
    for i, candidate in enumerate(candidates):
        log_prob = raw_log_probs[i]
        length_penalty = math.pow(candidate.token_count, alpha)
        normalized = log_prob / length_penalty
        
        normalized_scores.append(normalized)
        updated_stats.append(
            dataclasses.replace(candidate, raw_logprob=log_prob, normalized_score=normalized)
        )

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
