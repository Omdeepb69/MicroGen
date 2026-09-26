"""Tests for Task 1.2: Core Decision Engine & Single-Token Scoring.

Verification criterion: test_decision_single_token() passes, demonstrating
the highest probability is assigned to the most logical candidate.
"""

import pytest
import torch

from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult
from microgen.decision.scorer import (
    _extract_last_token_logits,
    score_single_token_candidates,
)
from microgen.decision.schema import CandidateStats


# ---------------------------------------------------------------------------
# Scorer unit tests — pure function, no model needed
# ---------------------------------------------------------------------------


def test_extract_last_token_logits_3d():
    """3-D logit tensor [1, seq, vocab] → 1-D slice at final position."""
    logits = torch.zeros(1, 5, 10)
    logits[0, -1, 3] = 99.0
    out = _extract_last_token_logits(logits)
    assert out.shape == (10,)
    assert out[3].item() == pytest.approx(99.0)


def test_extract_last_token_logits_2d():
    """2-D logit tensor [1, vocab] → flattened 1-D slice."""
    logits = torch.zeros(1, 10)
    logits[0, 7] = 5.0
    out = _extract_last_token_logits(logits)
    assert out.shape == (10,)
    assert out[7].item() == pytest.approx(5.0)


def test_score_single_token_picks_highest_logit():
    """Candidate with the highest assigned logit should win."""
    # Vocab size 10; token IDs 1,2,3,4 for LEFT/RIGHT/BRAKE/CLIMB
    last_logits = torch.zeros(10)
    last_logits[1] = 10.0   # LEFT — clear winner
    last_logits[2] = 2.0    # RIGHT
    last_logits[3] = 5.0    # BRAKE
    last_logits[4] = 1.0    # CLIMB

    candidates = [
        CandidateStats(text=" LEFT",  token_count=1, token_ids=[1]),
        CandidateStats(text=" RIGHT", token_count=1, token_ids=[2]),
        CandidateStats(text=" BRAKE", token_count=1, token_ids=[3]),
        CandidateStats(text=" CLIMB", token_count=1, token_ids=[4]),
    ]
    result = score_single_token_candidates(last_logits, candidates)

    assert result.choice == " LEFT"
    assert result.probabilities[" LEFT"] > result.probabilities[" BRAKE"]
    assert result.probabilities[" LEFT"] > result.probabilities[" RIGHT"]
    assert result.probabilities[" LEFT"] > result.probabilities[" CLIMB"]
    assert result.top_probability == pytest.approx(result.probabilities[" LEFT"])
    assert result.calibrated is False
    assert result.entropy is None


def test_score_single_token_probabilities_sum_to_one():
    """Softmax over candidate subset must sum to 1.0."""
    last_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    candidates = [
        CandidateStats(text=" A", token_count=1, token_ids=[0]),
        CandidateStats(text=" B", token_count=1, token_ids=[1]),
        CandidateStats(text=" C", token_count=1, token_ids=[2]),
    ]
    result = score_single_token_candidates(last_logits, candidates)
    total = sum(result.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-5)


def test_score_single_token_rejects_multi_token():
    """Multi-token candidates must be rejected with a clear error."""
    last_logits = torch.zeros(10)
    candidates = [
        CandidateStats(text=" LEFT",        token_count=1, token_ids=[1]),
        CandidateStats(text=" RETURN HOME", token_count=2, token_ids=[5, 6]),
    ]
    with pytest.raises(ValueError, match="multi-token"):
        score_single_token_candidates(last_logits, candidates)


def test_score_single_token_rejects_empty_candidates():
    """Empty candidate list must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        score_single_token_candidates(torch.zeros(10), [])


# ---------------------------------------------------------------------------
# DecisionEngine integration tests — uses MockDecisionModel fixture
# ---------------------------------------------------------------------------


def test_decision_single_token(mock_decision_model):
    """End-to-end: engine.choose() returns LEFT as winner from mock logits."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="drone_maneuver",
        options=[
            Choice("LEFT"),
            Choice("RIGHT"),
            Choice("BRAKE"),
            Choice("CLIMB"),
        ],
    )

    result = engine.choose(context="obstacle ahead left is clear", schema=schema)

    assert isinstance(result, DecisionResult)
    assert result.choice == "LEFT"
    assert result.probabilities["LEFT"] > result.probabilities["BRAKE"]
    assert result.probabilities["LEFT"] > result.probabilities["RIGHT"]
    assert result.probabilities["LEFT"] > result.probabilities["CLIMB"]
    assert result.top_probability == pytest.approx(result.probabilities["LEFT"])
    assert result.calibrated is False


def test_decision_probabilities_keyed_by_choice_name(mock_decision_model):
    """Probability keys must be Choice.name labels, NOT internal text (e.g. ' LEFT')."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="maneuver",
        options=[Choice("LEFT"), Choice("RIGHT")],
    )
    result = engine.choose(context="test", schema=schema)

    assert "LEFT" in result.probabilities
    assert "RIGHT" in result.probabilities
    assert " LEFT" not in result.probabilities  # internal key must not leak
    assert " RIGHT" not in result.probabilities


def test_decision_probabilities_sum_to_one(mock_decision_model):
    """Result probabilities must sum to 1.0 over the candidate set."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="maneuver",
        options=[Choice("LEFT"), Choice("RIGHT"), Choice("BRAKE"), Choice("CLIMB")],
    )
    result = engine.choose(context="test", schema=schema)
    total = sum(result.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-5)


def test_decision_temperature_zero_raises(mock_decision_model):
    """Temperature = 0 must be rejected."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(name="m", options=[Choice("A"), Choice("B")])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        engine.choose(context="test", schema=schema, temperature=0.0)


def test_decision_temperature_sharpens_distribution(mock_decision_model):
    """Lower temperature should increase the gap between winner and runner-up."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="maneuver",
        options=[Choice("LEFT"), Choice("RIGHT"), Choice("BRAKE"), Choice("CLIMB")],
    )

    result_high_temp = engine.choose(context="test", schema=schema, temperature=10.0)
    result_low_temp  = engine.choose(context="test", schema=schema, temperature=0.1)

    # The winner is the same in both cases
    assert result_high_temp.choice == result_low_temp.choice

    # Lower temperature → more concentrated distribution → higher top_probability
    assert result_low_temp.top_probability > result_high_temp.top_probability
