"""Decision Engine — decode-free inference for constrained option selection.

Provides a System-One-style API over any DecisionModel, converting ordinary
causal LMs into constrained probabilistic decision engines without generating
any output tokens.

Task 1.2 scope: single-token candidate scoring via choose().
Task 1.3 scope: yes_no() and score() high-level wrappers over choose().
Task 2.1 scope: multi-token sequence scoring routed from choose().
"""

from __future__ import annotations

import torch

from microgen.decision.adapters import DecisionModel
from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult, CandidateStats
from microgen.decision.tokenizer import tokenize_candidate
from microgen.decision.scorer import (
    _extract_last_token_logits,
    score_single_token_candidates,
)
from microgen.decision.constrained import score_trie_candidates
from microgen.decision.entropy import calculate_entropy
from microgen.runtime.kv_cache import KVCacheState


class DecisionEngine:
    """Decode-free decision engine wrapping a DecisionModel.

    Evaluates constrained choices directly from model prefill logits —
    no output tokens are generated.

    Args:
        model: Any object satisfying the DecisionModel protocol. In
            production this is typically an LLMEngine adapter; in tests
            it is a MockDecisionModel fixture.
    """

    def __init__(self, model: DecisionModel) -> None:
        self._model = model

    def choose(
        self,
        context: str,
        schema: ChoiceSchema,
        temperature: float = 1.0,
        alpha: float = 1.0,
    ) -> DecisionResult:
        """Score each option in schema against the context and return a decision.

        Performs a single prefill pass over the context, then scores candidates
        from the resulting logits.  For single-token candidates this requires
        only the prefill pass.  For multi-token candidates, sequential decode
        passes are performed per candidate token, reusing the prefill KV cache.
        No new tokens are added to the context beyond the candidates themselves.

        Args:
            context: The prompt / situation description to condition on.
            schema: The ChoiceSchema defining the candidate options.
            temperature: Softmax temperature for the candidate distribution.
                Values < 1.0 sharpen; values > 1.0 smooth.  Must be > 0.
            alpha: Length normalization penalty applied during sequence scoring.
                S_alpha = sum(log_P) / (|Y|^alpha).
                alpha=0.0: raw sum (favors short options).
                alpha=1.0: mean log-prob (default).

        Returns:
            DecisionResult with probabilities keyed by Choice.name,
            top_probability, calibrated=False, and entropy=None.
            DecisionResult.candidate_stats will contain the diagnostic data.

        Raises:
            ValueError: If temperature is not positive.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        # Tokenise the context prompt
        tok = self._model.tokenizer
        prompt_encoding = tok(context, return_tensors="pt")
        input_ids: torch.Tensor = prompt_encoding.input_ids
        attention_mask: torch.Tensor | None = prompt_encoding.get("attention_mask")

        # Move to the model's device if needed
        device = torch.device(self._model.device_type)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Single prefill pass — decode-free
        cache = KVCacheState()
        logits, _ = self._model.prefill(
            input_ids, attention_mask=attention_mask, cache=cache
        )
        last_logits = _extract_last_token_logits(logits)

        # Apply temperature before scoring
        last_logits = last_logits / max(temperature, 1e-8)

        # Tokenise each candidate option; build a name→stats mapping so we can
        # translate internal CandidateStats.text keys back to Choice.name labels.
        name_to_stats: dict[str, CandidateStats] = {}
        stats_list: list[CandidateStats] = []
        for choice in schema.options:
            stats = tokenize_candidate(tok, choice.name)
            name_to_stats[choice.name] = stats
            stats_list.append(stats)

        # Route: all single-token → fast path; any multi-token → sequence path.
        has_multi_token = any(s.token_count > 1 for s in stats_list)

        if has_multi_token:
            internal_result = score_trie_candidates(
                last_logits=last_logits,
                candidates=stats_list,
                decode_fn=self._model.decode,
                prefill_cache=cache,
                device=device,
                alpha=alpha,
            )
        else:
            internal_result = score_single_token_candidates(
                last_logits, stats_list, alpha=alpha
            )

        # Re-key probabilities from CandidateStats.text (e.g., " LEFT") back
        # to the caller's Choice.name labels (e.g., "LEFT").
        text_to_name = {stats.text: name for name, stats in name_to_stats.items()}
        probabilities: dict[str, float] = {
            text_to_name[text]: prob
            for text, prob in internal_result.probabilities.items()
        }

        winner_name = text_to_name[internal_result.choice]
        top_probability = internal_result.top_probability
        entropy = calculate_entropy(probabilities)

        return DecisionResult(
            choice=winner_name,
            probabilities=probabilities,
            top_probability=top_probability,
            entropy=entropy,
            candidate_stats=internal_result.candidate_stats,
        )

    def yes_no(
        self,
        context: str,
        question: str,
        temperature: float = 1.0,
    ) -> DecisionResult:
        """Answer a yes/no question conditioned on context.

        Combines `context` and `question` into a single prompt and scores
        "YES" vs "NO" via decode-free candidate scoring.

        Args:
            context: The situational background to condition on.
            question: The yes/no question to answer.
            temperature: Passed through to choose().

        Returns:
            DecisionResult with choice in {"YES", "NO"} and a two-entry
            probability distribution.
        """
        prompt = f"{context.rstrip()}\nQuestion: {question}\nAnswer:"
        schema = ChoiceSchema(
            name="yes_no",
            options=[Choice("YES"), Choice("NO")],
        )
        return self.choose(context=prompt, schema=schema, temperature=temperature)

    def score(
        self,
        context: str,
        criteria: str,
        scale: list[int],
        temperature: float = 1.0,
    ) -> DecisionResult:
        """Score a situation on a discrete integer scale.

        Formats the criteria and scale into a prompt and scores each integer
        label via decode-free candidate scoring.

        Args:
            context: The situational background to condition on.
            criteria: The dimension to score (e.g., "collision risk").
            scale: Ordered list of integer values defining the rating range
                (e.g., [1, 2, 3, 4, 5]).  Must contain at least 2 values.
            temperature: Passed through to choose().

        Returns:
            DecisionResult with choice as a string digit (e.g., "3") and a
            probability entry for each scale value.

        Raises:
            ValueError: If scale contains fewer than 2 values.
        """
        if len(scale) < 2:
            raise ValueError(
                f"scale must contain at least 2 values, got {len(scale)}."
            )
        scale_str = " | ".join(str(v) for v in scale)
        prompt = (
            f"{context.rstrip()}\n"
            f"Rate the {criteria} on a scale of {scale_str}.\n"
            f"Rating:"
        )
        schema = ChoiceSchema(
            name="score",
            options=[Choice(str(v)) for v in scale],
        )
        return self.choose(context=prompt, schema=schema, temperature=temperature)
