"""Tests for Task 4.1: Trie-based State Space.

Verification criterion: test_constrained_trie_scoring() passes for large
candidate sets, ensuring that trie-based decoding yields the same probabilities
as independent sequence scoring while optimizing forward passes.
"""

import pytest
import torch

from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema, CandidateStats
from microgen.decision.constrained import build_candidate_trie, TrieNode, score_trie_candidates


def test_build_candidate_trie():
    """Verify prefix trie construction logic."""
    candidates = [
        CandidateStats(text=" A", token_count=1, token_ids=[1]),
        CandidateStats(text=" AB", token_count=2, token_ids=[1, 2]),
        CandidateStats(text=" AC", token_count=2, token_ids=[1, 3]),
        CandidateStats(text=" B", token_count=1, token_ids=[4]),
    ]
    
    root = build_candidate_trie(candidates)
    
    assert root.token_id == -1
    assert len(root.children) == 2
    assert 1 in root.children
    assert 4 in root.children
    
    node_1 = root.children[1]
    assert node_1.candidate_indices == [0] # " A" ends here
    assert len(node_1.children) == 2
    assert 2 in node_1.children
    assert 3 in node_1.children
    
    node_1_2 = node_1.children[2]
    assert node_1_2.candidate_indices == [1]
    assert len(node_1_2.children) == 0
    
    node_1_3 = node_1.children[3]
    assert node_1_3.candidate_indices == [2]
    
    node_4 = root.children[4]
    assert node_4.candidate_indices == [3]


class DummyDecodeCache:
    def __init__(self):
        self.seq_len = 10
        self.decode_calls = 0
        
    def get_seq_length(self):
        return self.seq_len
        
    def rollback(self, steps):
        self.seq_len -= steps


def mock_decode(tokens: torch.Tensor, cache: DummyDecodeCache) -> tuple[torch.Tensor, DummyDecodeCache]:
    cache.seq_len += 1
    cache.decode_calls += 1
    logits = torch.zeros(1, 10)
    logits[0, 5] = 10.0 # Just some mock values
    return logits, cache


def test_constrained_trie_scoring_reduces_decodes():
    """Verify that Trie decoding uses fewer forward passes than independent sequences."""
    candidates = [
        CandidateStats(text=" PREFIX A", token_count=3, token_ids=[1, 2, 3]),
        CandidateStats(text=" PREFIX B", token_count=3, token_ids=[1, 2, 4]),
    ]
    
    # If scored independently, token_ids=[1,2,3] requires 2 decodes (for 2 and 3).
    # Two such candidates = 4 decodes.
    # With a Trie:
    # root (-1) -> 1 -> 2 -> 3
    #                     -> 4
    # Depth 1: no decode
    # Depth 2 (node 2): 1 decode
    # Depth 3 (node 3 and 4): 2 decodes
    # Total decodes = 3.
    
    cache = DummyDecodeCache()
    last_logits = torch.zeros(10)
    last_logits[1] = 5.0
    
    res = score_trie_candidates(
        last_logits=last_logits,
        candidates=candidates,
        decode_fn=mock_decode,
        prefill_cache=cache,
        device=torch.device("cpu"),
    )
    
    assert cache.decode_calls == 3


def test_constrained_trie_integration(mock_decision_model):
    """End-to-end test replacing score_sequence_candidates with score_trie_candidates."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="test",
        options=[Choice("LEFT"), Choice("RETURN HOME")],
    )

    result = engine.choose(context="test", schema=schema)

    assert result.choice == "LEFT"
    assert "LEFT" in result.probabilities
    assert "RETURN HOME" in result.probabilities
    
    total_prob = sum(result.probabilities.values())
    assert total_prob == pytest.approx(1.0, abs=1e-5)
