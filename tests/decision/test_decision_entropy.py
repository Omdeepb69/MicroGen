"""Tests for Task 3.1: Entropy Metrics.

Verification criterion: test_decision_entropy() passes, verifying that Shannon
entropy is correctly calculated over the normalized candidate distribution and
populated in DecisionResult.
"""

import math
import pytest

from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.entropy import calculate_entropy


def test_calculate_entropy_deterministic():
    """Entropy of a deterministic distribution (prob=1.0) is 0.0 bits."""
    assert calculate_entropy({"A": 1.0}) == 0.0


def test_calculate_entropy_uniform_binary():
    """Entropy of a uniform binary distribution is 1.0 bits."""
    # - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (-0.5 + -0.5) = 1.0
    assert calculate_entropy({"A": 0.5, "B": 0.5}) == 1.0


def test_calculate_entropy_uniform_four():
    """Entropy of a uniform 4-option distribution is 2.0 bits."""
    probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    assert calculate_entropy(probs) == 2.0


def test_calculate_entropy_zero_prob():
    """Entropy calculation correctly handles zero probabilities."""
    # lim p->0 of p*log(p) is 0
    probs = {"A": 1.0, "B": 0.0}
    assert calculate_entropy(probs) == 0.0


def test_calculate_entropy_empty():
    """Entropy of an empty distribution is 0.0."""
    assert calculate_entropy({}) == 0.0


def test_engine_populates_entropy(mock_decision_model):
    """Verify engine.choose() populates DecisionResult.entropy."""
    engine = DecisionEngine(mock_decision_model)
    schema = ChoiceSchema(
        name="test",
        options=[Choice("LEFT"), Choice("RIGHT"), Choice("BRAKE")]
    )
    
    result = engine.choose(context="ctx", schema=schema)
    
    # We don't need to assert the exact value here (since mock logits could change),
    # just that the entropy field is properly populated and matches a manual calculation.
    assert result.entropy is not None
    assert result.entropy >= 0.0
    
    expected = calculate_entropy(result.probabilities)
    assert result.entropy == pytest.approx(expected)
