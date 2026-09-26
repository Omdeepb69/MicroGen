"""Tests for the Task 1.1 schema types: Choice, ChoiceSchema, DecisionResult."""

import pytest

from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult


# ---------------------------------------------------------------------------
# Choice
# ---------------------------------------------------------------------------


def test_choice_name_only():
    c = Choice(name="LEFT")
    assert c.name == "LEFT"
    assert c.value is None


def test_choice_with_value():
    c = Choice(name="LEFT", value=0)
    assert c.value == 0


def test_choice_is_frozen():
    c = Choice(name="LEFT")
    with pytest.raises((AttributeError, TypeError)):
        c.name = "RIGHT"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChoiceSchema
# ---------------------------------------------------------------------------


def test_choice_schema_valid():
    schema = ChoiceSchema(
        name="drone_maneuver",
        options=[
            Choice("LEFT"),
            Choice("RIGHT"),
            Choice("BRAKE"),
            Choice("CLIMB"),
        ],
    )
    assert schema.name == "drone_maneuver"
    assert len(schema.options) == 4


def test_choice_schema_minimum_two_options():
    with pytest.raises(ValueError, match="at least 2 options"):
        ChoiceSchema(name="bad", options=[Choice("ONLY")])


def test_choice_schema_exactly_two_options_is_valid():
    schema = ChoiceSchema(name="binary", options=[Choice("YES"), Choice("NO")])
    assert len(schema.options) == 2


def test_choice_schema_is_frozen():
    schema = ChoiceSchema(name="s", options=[Choice("A"), Choice("B")])
    with pytest.raises((AttributeError, TypeError)):
        schema.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DecisionResult
# ---------------------------------------------------------------------------


def test_decision_result_valid():
    result = DecisionResult(
        choice="LEFT",
        probabilities={"LEFT": 0.81, "RIGHT": 0.04, "BRAKE": 0.12, "CLIMB": 0.03},
        top_probability=0.81,
    )
    assert result.choice == "LEFT"
    assert result.top_probability == 0.81
    assert result.entropy is None
    assert result.calibrated is False


def test_decision_result_with_entropy_and_calibrated():
    result = DecisionResult(
        choice="BRAKE",
        probabilities={"LEFT": 0.3, "BRAKE": 0.7},
        top_probability=0.7,
        entropy=0.88,
        calibrated=True,
    )
    assert result.entropy == pytest.approx(0.88)
    assert result.calibrated is True


def test_decision_result_choice_not_in_probabilities():
    with pytest.raises(ValueError, match="not a key in"):
        DecisionResult(
            choice="UNKNOWN",
            probabilities={"LEFT": 0.6, "RIGHT": 0.4},
            top_probability=0.6,
        )


def test_decision_result_top_probability_out_of_range():
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        DecisionResult(
            choice="LEFT",
            probabilities={"LEFT": 1.5},
            top_probability=1.5,
        )


def test_decision_result_is_frozen():
    result = DecisionResult(
        choice="LEFT",
        probabilities={"LEFT": 1.0},
        top_probability=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.choice = "RIGHT"  # type: ignore[misc]


def test_decision_result_no_confidence_field():
    """The output type must not expose a raw 'confidence' field."""
    result = DecisionResult(
        choice="LEFT",
        probabilities={"LEFT": 0.9, "RIGHT": 0.1},
        top_probability=0.9,
    )
    assert not hasattr(result, "confidence"), (
        "DecisionResult must not expose 'confidence' — use 'top_probability' "
        "for the raw max-probability and 'calibrated' to signal calibration status."
    )
