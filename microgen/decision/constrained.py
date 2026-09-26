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
    decode_fn: Callable[..., Tuple[torch.Tensor, Any]],
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
    # last_logits is [vocab_size]
    first_probs = torch.softmax(last_logits, dim=-1)

    # 3. BFS Batched Frontier setup
    frontier_nodes: list[TrieNode] = []
    frontier_log_probs: list[float] = []

    # Process depth-1 children of the root node
    for child_token, child_node in root.children.items():
        child_p = first_probs[child_token].item() + 1e-45
        child_log_prob = math.log(child_p)
        
        # If candidates end exactly here, store their log_prob
        for idx in child_node.candidate_indices:
            raw_log_probs[idx] = child_log_prob
            
        # Add to frontier if it has further children
        if child_node.children:
            frontier_nodes.append(child_node)
            frontier_log_probs.append(child_log_prob)

    if frontier_nodes:
        # We have active branches at depth > 1. Expand cache to match frontier size.
        current_cache = prefill_cache.expand_batch(len(frontier_nodes))
        
        # BFS loop
        while frontier_nodes:
            next_nodes: list[TrieNode] = []
            next_log_probs: list[float] = []
            parent_indices: list[int] = []
            
            # Prepare batched input for the current frontier
            input_ids = torch.tensor([[node.token_id] for node in frontier_nodes], device=device)
            
            # Batched forward pass
            step_logits, next_cache = decode_fn(input_ids, cache=current_cache)
            
            # Extract last token logits for the batch: shape [B, vocab_size]
            if step_logits.ndim == 3:
                step_last = step_logits[:, -1, :]
            else:
                step_last = step_logits
                
            step_probs = torch.softmax(step_last, dim=-1)
            
            # Process children
            for batch_idx, parent_node in enumerate(frontier_nodes):
                parent_log_prob = frontier_log_probs[batch_idx]
                parent_probs = step_probs[batch_idx]
                
                for child_token, child_node in parent_node.children.items():
                    child_p = parent_probs[child_token].item() + 1e-45
                    child_log_prob = parent_log_prob + math.log(child_p)
                    
                    for idx in child_node.candidate_indices:
                        raw_log_probs[idx] = child_log_prob
                        
                    if child_node.children:
                        next_nodes.append(child_node)
                        next_log_probs.append(child_log_prob)
                        parent_indices.append(batch_idx)
                        
            if not next_nodes:
                break
                
            frontier_nodes = next_nodes
            frontier_log_probs = next_log_probs
            
            # Gather cache for the active children
            # Identify the device of the cache tensors to ensure index_select works
            if hasattr(next_cache, "key_cache") and next_cache.key_cache and next_cache.key_cache[0] is not None:
                cache_device = next_cache.key_cache[0].device
            else:
                cache_device = device
                
            indices_tensor = torch.tensor(parent_indices, device=cache_device)
            current_cache = next_cache.gather_batch(indices_tensor)

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
