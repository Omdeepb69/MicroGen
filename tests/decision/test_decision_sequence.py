"""Tests for Task 2.1: Multi-Token Candidate Sequence Scoring.

Verification criterion: test_decision_sequence_multi_token() passes, proving
that multi-token candidates are correctly scored using accumulated log-probs,
and test_scorer_rollback() verifies the cache is correctly restored.
"""

import math
import pytest
import torch

from typing import Any
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult, CandidateStats
from microgen.decision.scorer import _sequence_log_prob, score_sequence_candidates
from microgen.runtime.kv_cache import KVCacheState


# ---------------------------------------------------------------------------
# Scorer unit tests — pure function, explicit mock cache
# ---------------------------------------------------------------------------


class DummyCache:
    """A minimal mock cache to verify rollback is called with correct counts."""
    def __init__(self):
        self.seq_len = 10
        self.rollbacks = []

    def get_seq_length(self):
        return self.seq_len

    def rollback(self, steps: int):
        self.rollbacks.append(steps)
        # We don't actually change seq_len so the assert in scorer passes
        # because initial_cache_len == prefill_cache.get_seq_length().


def mock_decode_fn(token_ids: torch.Tensor, cache: Any) -> tuple[torch.Tensor, Any]:
    """A deterministic mock decode function for unit tests."""
    logits = torch.zeros(1, 1, 10)
    
    tid = token_ids[0, -1].item()
    if tid == 5:
        # If token is 5, next token 6 gets a high logit
        logits[0, 0, 6] = 5.0
        logits[0, 0, 7] = 2.0
    return logits, cache


def test_sequence_log_prob_accumulation():
    """Verify log-probs accumulate correctly across prefill and decode passes."""
    first_logits = torch.zeros(10)
    first_logits[5] = 4.0   # token 5
    
    log_p, _ = _sequence_log_prob(
        first_logits=first_logits,
        token_ids=[5, 6],
        decode_fn=mock_decode_fn,
        cache=DummyCache(),
        device=torch.device("cpu"),
    )
    
    # Expected log_prob:
    # log P(5 | prefill) = log(softmax(first_logits)[5])
    # log P(6 | 5) = log(softmax(mock_decode_fn(5))[6])
    p5 = torch.softmax(first_logits, dim=-1)[5].item()
    
    step_logits = torch.zeros(10)
    step_logits[6] = 5.0
    step_logits[7] = 2.0
    p6_given_5 = torch.softmax(step_logits, dim=-1)[6].item()
    
    expected_log_p = math.log(p5 + 1e-45) + math.log(p6_given_5 + 1e-45)
    
    assert log_p == pytest.approx(expected_log_p)


def test_score_sequence_candidates_rollback_called():
    """Verify that score_sequence_candidates rolls back the cache correctly."""
    first_logits = torch.zeros(10)
    cache = DummyCache()
    
    candidates = [
        CandidateStats(text=" SINGLE", token_count=1, token_ids=[1]),
        CandidateStats(text=" DOUBLE", token_count=2, token_ids=[5, 6]),
        CandidateStats(text=" TRIPLE", token_count=3, token_ids=[2, 3, 4]),
    ]
    
    score_sequence_candidates(
        last_logits=first_logits,
        candidates=candidates,
        decode_fn=mock_decode_fn,
        prefill_cache=cache,
        device=torch.device("cpu"),
    )
    
    # Single token -> 0 rollbacks (not called since decode_steps <= 0)
    # Double token -> 1 rollback
    # Triple token -> 2 rollbacks
    assert cache.rollbacks == [1, 2]


def test_score_sequence_candidates_empty_reject():
    with pytest.raises(ValueError, match="non-empty"):
        score_sequence_candidates(torch.zeros(10), [], mock_decode_fn, DummyCache(), torch.device("cpu"))


# ---------------------------------------------------------------------------
# DecisionEngine integration tests
# ---------------------------------------------------------------------------


def test_decision_sequence_multi_token(mock_decision_model):
    """End-to-end: engine.choose() correctly routes and scores multi-token candidates."""
    engine = DecisionEngine(mock_decision_model)
    
    # " LEFT" -> id 1, prefill logit = 10.0 -> P ~ 1.0 (logP ~ 0)
    # " RETURN HOME" -> id [5, 6], prefill logit 5 = 4.0, decode logit 6 = 15.0
    # Because 10.0 is very high, LEFT will still win in raw probabilities,
    # but the sequence path should execute successfully.
    
    schema = ChoiceSchema(
        name="drone_maneuver",
        options=[
            Choice("LEFT"),
            Choice("RETURN HOME"),
        ],
    )

    result = engine.choose(context="base is damaged", schema=schema)

    assert isinstance(result, DecisionResult)
    assert result.choice == "LEFT"
    assert "LEFT" in result.probabilities
    assert "RETURN HOME" in result.probabilities
    
    total_prob = sum(result.probabilities.values())
    assert total_prob == pytest.approx(1.0, abs=1e-5)


def test_decision_sequence_temperature(mock_decision_model):
    """Sequence scoring properly reflects the temperature scaling from prefill/decode."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="drone_maneuver",
        options=[Choice("LEFT"), Choice("RETURN HOME")],
    )

    result_high_temp = engine.choose(context="test", schema=schema, temperature=10.0)
    result_low_temp  = engine.choose(context="test", schema=schema, temperature=0.5)

    assert result_low_temp.top_probability > result_high_temp.top_probability


def test_decision_length_normalization(mock_decision_model):
    """Sequence scoring uses alpha-scaled length normalization (S_alpha).
    
    A long candidate with a large negative accumulated log-prob can outscore
    a shorter candidate if alpha is sufficiently high (S_alpha = log_p / len^alpha).
    """
    engine = DecisionEngine(mock_decision_model)
    # Using 'LEFT' (len=1) and 'RETURN HOME' (len=2)
    # mock_decision_model prefill: LEFT logit=10.0 (high probability -> near 0 log_p)
    # RETURN HOME: id 5 prefill logit=4.0, id 6 decode logit=15.0
    # Let's override the mock behavior with a custom schema/mock if needed, 
    # but we can just test that alpha changes the returned probabilities.
    
    schema = ChoiceSchema(
        name="drone_maneuver",
        options=[Choice("LEFT"), Choice("RETURN HOME")],
    )

    result_alpha_0 = engine.choose(context="test", schema=schema, alpha=0.0)
    result_alpha_1 = engine.choose(context="test", schema=schema, alpha=1.0)
    
    # Probabilities should shift because the length penalty for len=2 
    # is 2^0 = 1 in the first case, but 2^1 = 2 in the second case.
    # So alpha=1.0 divides the (negative) log prob of RETURN HOME by 2, 
    # making it LESS negative, thus INCREASING its probability relative to alpha=0.
    prob_return_home_alpha0 = result_alpha_0.probabilities["RETURN HOME"]
    prob_return_home_alpha1 = result_alpha_1.probabilities["RETURN HOME"]
    
    assert prob_return_home_alpha1 > prob_return_home_alpha0

    # Verify candidate stats were populated
    stats = result_alpha_1.candidate_stats
    assert len(stats) == 2
    for stat in stats:
        assert stat.raw_logprob is not None
        assert stat.normalized_score is not None
        if stat.token_count == 2:
            assert stat.normalized_score == pytest.approx(stat.raw_logprob / 2.0)
        else:
            assert stat.normalized_score == pytest.approx(stat.raw_logprob)
