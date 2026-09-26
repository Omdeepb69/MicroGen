"""Tests for Task 1.3: yes_no() and score() high-level System-One wrappers.

Verification criterion: test_decision_yes_no() and test_decision_score() pass.
"""

import pytest

from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult


# ---------------------------------------------------------------------------
# yes_no() tests
# ---------------------------------------------------------------------------


def test_decision_yes_no_returns_decision_result(mock_decision_model):
    """yes_no() must return a valid DecisionResult."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(
        context="Left corridor is 5.4m wide. Drone wingspan is 0.6m.",
        question="Is the left corridor safely passable?",
    )
    assert isinstance(result, DecisionResult)


def test_decision_yes_no_choice_is_yes_or_no(mock_decision_model):
    """yes_no() choice must be exactly 'YES' or 'NO'."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(
        context="Obstacle is 5m away and closing at 1 m/s.",
        question="Is there time to maneuver?",
    )
    assert result.choice in {"YES", "NO"}


def test_decision_yes_no_probabilities_keys(mock_decision_model):
    """yes_no() probabilities must have exactly 'YES' and 'NO' as keys."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(context="ctx", question="q?")
    assert set(result.probabilities.keys()) == {"YES", "NO"}


def test_decision_yes_no_probabilities_sum_to_one(mock_decision_model):
    """yes_no() probabilities must sum to 1.0."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(context="ctx", question="q?")
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_decision_yes_no_mock_picks_yes(mock_decision_model):
    """With mock logits (YES id=7 has logit 8.0 > NO id=8 logit 2.0), YES wins."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(context="ctx", question="q?")
    assert result.choice == "YES"
    assert result.probabilities["YES"] > result.probabilities["NO"]


def test_decision_yes_no_top_probability_matches_winner(mock_decision_model):
    """top_probability must equal the winning choice's probability."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(context="ctx", question="q?")
    assert result.top_probability == pytest.approx(result.probabilities[result.choice])


def test_decision_yes_no_calibrated_false(mock_decision_model):
    """yes_no() result is not calibrated by default."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.yes_no(context="ctx", question="q?")
    assert result.calibrated is False


def test_decision_yes_no_passes_temperature(mock_decision_model):
    """Temperature is forwarded correctly: lower temp → higher top_probability."""
    engine = DecisionEngine(mock_decision_model)
    high_t = engine.yes_no(context="ctx", question="q?", temperature=10.0)
    low_t  = engine.yes_no(context="ctx", question="q?", temperature=0.1)
    assert low_t.top_probability > high_t.top_probability


# ---------------------------------------------------------------------------
# score() tests
# ---------------------------------------------------------------------------


def test_decision_score_returns_decision_result(mock_decision_model):
    """score() must return a valid DecisionResult."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.score(
        context="Obstacle 1.2m ahead. Speed: 4 m/s.",
        criteria="collision risk",
        scale=[1, 2, 3, 4, 5],
    )
    assert isinstance(result, DecisionResult)


def test_decision_score_choice_in_scale(mock_decision_model):
    """score() choice must be one of the scale values (as a string)."""
    engine = DecisionEngine(mock_decision_model)
    scale = [1, 2, 3, 4, 5]
    result = engine.score(context="ctx", criteria="risk", scale=scale)
    assert result.choice in {str(v) for v in scale}


def test_decision_score_probability_keys_are_scale_strings(mock_decision_model):
    """score() probabilities must be keyed by the string form of each scale value."""
    engine = DecisionEngine(mock_decision_model)
    scale = [1, 2, 3, 4, 5]
    result = engine.score(context="ctx", criteria="risk", scale=scale)
    assert set(result.probabilities.keys()) == {str(v) for v in scale}


def test_decision_score_probabilities_sum_to_one(mock_decision_model):
    """score() probabilities must sum to 1.0."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.score(context="ctx", criteria="risk", scale=[1, 2, 3, 4, 5])
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_decision_score_mock_picks_4(mock_decision_model):
    """With mock logits (id 14 = ' 4' has logit 7.0, highest among digits), '4' wins."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.score(context="ctx", criteria="risk", scale=[1, 2, 3, 4, 5])
    assert result.choice == "4"


def test_decision_score_rejects_single_value_scale(mock_decision_model):
    """scale with fewer than 2 values must raise ValueError."""
    engine = DecisionEngine(mock_decision_model)
    with pytest.raises(ValueError, match="at least 2 values"):
        engine.score(context="ctx", criteria="risk", scale=[3])


def test_decision_score_custom_scale(mock_decision_model):
    """score() works for a binary scale."""
    engine = DecisionEngine(mock_decision_model)
    result = engine.score(context="ctx", criteria="urgency", scale=[1, 2])
    assert result.choice in {"1", "2"}
    assert set(result.probabilities.keys()) == {"1", "2"}
