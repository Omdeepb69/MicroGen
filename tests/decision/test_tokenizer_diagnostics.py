"""Tests for the decision module tokenizer diagnostics.

These tests verify:
  - CandidateStats is a proper frozen dataclass with the right field contract.
  - tokenize_candidate() produces correct token_count / token_ids via the mock tokenizer.
  - prefix-space behaviour is handled correctly for single- and multi-token candidates.
"""

from tests.decision.conftest import MockTokenizer
from microgen.decision.schema import CandidateStats
from microgen.decision.tokenizer import tokenize_candidate


def test_candidate_stats_is_frozen_dataclass():
    """CandidateStats must be immutable and hold exactly the specified fields."""
    stats = CandidateStats(
        text=" LEFT",
        token_count=1,
        token_ids=[1],
    )
    assert stats.text == " LEFT"
    assert stats.token_count == 1
    assert stats.token_ids == [1]
    assert stats.raw_logprob is None
    assert stats.normalized_score is None

    import pytest
    with pytest.raises((AttributeError, TypeError)):
        stats.token_count = 99  # type: ignore[misc]  # should raise FrozenInstanceError


def test_tokenize_candidate_single_token(mock_decision_model):
    """Single-word candidates should produce exactly one token ID."""
    tok = mock_decision_model.tokenizer

    stats = tokenize_candidate(tok, "LEFT")

    assert stats.text == " LEFT"       # prefix space prepended
    assert stats.token_count == 1
    assert stats.token_ids == [1]
    assert stats.raw_logprob is None   # not scored yet
    assert stats.normalized_score is None


def test_tokenize_candidate_multi_token(mock_decision_model):
    """Multi-word candidates produce the correct number of token IDs."""
    tok = mock_decision_model.tokenizer

    stats = tokenize_candidate(tok, "RETURN HOME")

    assert stats.text == " RETURN HOME"
    assert stats.token_count == 2
    assert stats.token_ids == [5, 6]


def test_tokenize_candidate_no_double_prefix(mock_decision_model):
    """If the caller already includes a leading space, no extra space is added."""
    tok = mock_decision_model.tokenizer

    stats = tokenize_candidate(tok, " LEFT", add_prefix_space=True)

    # Should still be " LEFT", not "  LEFT"
    assert stats.text == " LEFT"
    assert stats.token_count == 1
    assert stats.token_ids == [1]


def test_tokenize_candidate_all_options(mock_decision_model):
    """All four drone options tokenize to single tokens with expected IDs."""
    tok = mock_decision_model.tokenizer
    options = ["LEFT", "RIGHT", "BRAKE", "CLIMB"]
    expected_ids = [1, 2, 3, 4]

    for option, expected_id in zip(options, expected_ids):
        stats = tokenize_candidate(tok, option)
        assert stats.token_count == 1, f"{option!r} should be 1 token"
        assert stats.token_ids == [expected_id], f"{option!r} unexpected token id"
