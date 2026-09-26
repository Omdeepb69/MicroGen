"""Typed schema definitions for the MicroGen Decision Engine.

Grows incrementally across tasks:
  Task 0.2 — CandidateStats (tokenizer diagnostics)
  Task 1.1 — Choice, ChoiceSchema, DecisionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CandidateStats:
    """Diagnostic record produced when tokenizing a single decision candidate.

    Holds the raw tokenization result together with any scores assigned
    during inference so that callers can inspect why sequence scoring or
    length normalization had the effect it did.

    Attributes:
        text: The original candidate string as passed by the caller.
        token_count: Number of tokens the candidate tokenized to.
        token_ids: Ordered list of token IDs produced by the tokenizer.
        raw_logprob: Sum of per-token log-probabilities before normalization.
            None until the candidate has been scored by the engine.
        normalized_score: Length-normalized score (Sα).
            None until the candidate has been scored with normalization.
    """

    text: str
    token_count: int
    token_ids: list[int]
    raw_logprob: Optional[float] = field(default=None)
    normalized_score: Optional[float] = field(default=None)


@dataclass(frozen=True)
class Choice:
    """A single candidate option in a decision schema.

    Attributes:
        name: Human-readable label used to identify this option in
            `DecisionResult.probabilities` and `DecisionResult.choice`.
        value: Optional caller-defined payload (e.g., an integer command
            code or enum member). The engine does not interpret this.
    """

    name: str
    value: Any = field(default=None)


@dataclass(frozen=True)
class ChoiceSchema:
    """The full candidate set definition for a single decision call.

    Attributes:
        name: Identifier for this decision type (e.g., "drone_maneuver").
        options: Ordered list of candidate choices. Must contain at least
            two options — fewer options is not a decision.
    """

    name: str
    options: list[Choice]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(
                f"ChoiceSchema '{self.name}' must have at least 2 options, "
                f"got {len(self.options)}."
            )


@dataclass(frozen=True)
class DecisionResult:
    """Immutable output of a single Decision Engine inference call.

    Attributes:
        choice: The name of the winning option (highest probability).
        probabilities: Mapping from each candidate name to its probability.
            Values are in [0, 1] and sum to 1.0 over the candidate set.
        top_probability: The probability of the winning choice, i.e.
            max(probabilities.values()). Deliberately not named "confidence"
            because confidence implies calibration — see the `calibrated` flag.
        entropy: Shannon entropy of the candidate distribution (bits).
            None until computed by the entropy module.
        calibrated: True only after a TemperatureScaler has been applied to
            the candidate logits. Raw softmax outputs leave this False.
        candidate_stats: Diagnostic data for each candidate (raw logprobs,
            length-normalized scores). Only populated if tracing is enabled or
            during multi-token sequence scoring.
    """

    choice: str
    probabilities: dict[str, float]
    top_probability: float
    entropy: Optional[float] = field(default=None)
    calibrated: bool = field(default=False)
    candidate_stats: list[CandidateStats] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.choice not in self.probabilities:
            raise ValueError(
                f"DecisionResult.choice '{self.choice}' is not a key in "
                f"DecisionResult.probabilities."
            )
        if not (0.0 <= self.top_probability <= 1.0):
            raise ValueError(
                f"top_probability must be in [0, 1], got {self.top_probability}."
            )
