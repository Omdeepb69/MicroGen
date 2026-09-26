"""Candidate tokenization helpers for the Decision Engine.

Provides the single function `tokenize_candidate`, which resolves a
candidate string to a `CandidateStats` diagnostic record containing the
token IDs and token count.  This module has no inference logic; it only
converts text to token sequences and wraps the result in the typed schema.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer

from microgen.decision.schema import CandidateStats


def tokenize_candidate(
    tokenizer: PreTrainedTokenizer,
    candidate: str,
    add_prefix_space: bool = True,
) -> CandidateStats:
    """Tokenize a single candidate string and return a diagnostic record.

    For causal LMs, options such as "LEFT" are typically encoded with a
    leading space so that the byte-pair or sentencepiece encoding matches
    the mid-sentence representation (e.g., " LEFT" → single token).
    `add_prefix_space=True` (the default) prepends a space unless the
    caller has already done so.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer.
        candidate: The option text (e.g. "LEFT", "RETURN HOME").
        add_prefix_space: Prepend a space before tokenizing if the
            candidate does not already start with one.

    Returns:
        CandidateStats with `text`, `token_count`, and `token_ids` filled.
        `raw_logprob` and `normalized_score` are left as None — the scorer
        fills those in after the forward pass.
    """
    text = candidate
    if add_prefix_space and not candidate.startswith(" "):
        text = " " + candidate

    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

    return CandidateStats(
        text=text,
        token_count=len(token_ids),
        token_ids=token_ids,
    )
