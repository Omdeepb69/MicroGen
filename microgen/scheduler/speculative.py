"""Speculative decoding execution engine with draft model candidate generation, target model verification, rejection sampling, and KV cache rollback."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import torch

from microgen.backends.base import InferenceBackend
from microgen.runtime.kv_cache import KVCacheState


@dataclass
class SpeculativeResult:
    """Dataclass holding output token sequence and speculative execution statistics."""
    output_ids: List[int]
    total_drafted_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    num_steps: int


def rejection_sample_token(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_token: int,
) -> Tuple[bool, int]:
    """Perform speculative rejection sampling according to Leviathan et al. (2023).

    Returns:
        (is_accepted, token_id): True and draft_token if accepted; False and resampled target token if rejected.
    """
    p_draft = float(draft_probs[0, draft_token].item())
    p_target = float(target_probs[0, draft_token].item())

    if p_target >= p_draft:
        return True, draft_token

    acceptance_prob = p_target / max(p_draft, 1e-12)
    rand_val = float(torch.rand(1).item())

    if rand_val < acceptance_prob:
        return True, draft_token

    # Rejected: resample from normalized max(0, P_target - P_draft)
    diff = torch.clamp(target_probs - draft_probs, min=0.0)
    diff_sum = diff.sum()

    if diff_sum > 0:
        norm_probs = diff / diff_sum
        resampled_id = int(torch.multinomial(norm_probs, num_samples=1).item())
    else:
        resampled_id = int(torch.multinomial(target_probs, num_samples=1).item())

    return False, resampled_id


class SpeculativeEngine:
    """Speculative Decoding Engine using a small draft model and a large target model.

    Generates K candidate tokens using the fast draft backend, validates them with
    the target backend, performs rejection sampling, and executes KV cache rollback
    on token rejection.
    """

    def __init__(
        self,
        draft_backend: InferenceBackend,
        target_backend: InferenceBackend,
        num_draft_tokens: int = 4,
    ) -> None:
        self.draft_backend = draft_backend
        self.target_backend = target_backend
        self.num_draft_tokens = num_draft_tokens

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        draft_cache: Optional[KVCacheState] = None,
        target_cache: Optional[KVCacheState] = None,
    ) -> SpeculativeResult:
        """Execute speculative decoding autoregressive loop until max_new_tokens are generated."""
        current_ids = prompt_ids.clone()
        generated_count = 0
        total_drafted = 0
        total_accepted = 0
        step_count = 0

        if draft_cache is None:
            draft_cache = KVCacheState()
        if target_cache is None:
            target_cache = KVCacheState()

        # Perform initial prefill for draft and target backends
        draft_logits, draft_cache = self.draft_backend.prefill(current_ids, cache=draft_cache)
        target_logits, target_cache = self.target_backend.prefill(current_ids, cache=target_cache)

        while generated_count < max_new_tokens:
            step_count += 1
            k = min(self.num_draft_tokens, max_new_tokens - generated_count)

            # 1. Draft phase: generate K candidate tokens sequentially using draft model
            draft_candidates: List[int] = []
            draft_probs_list: List[torch.Tensor] = []
            cur_draft_logits = draft_logits

            for _ in range(k):
                draft_probs = torch.softmax(cur_draft_logits, dim=-1)
                next_token_id = int(self.draft_backend.sample(cur_draft_logits).item())
                draft_candidates.append(next_token_id)
                draft_probs_list.append(draft_probs)

                step_tensor = torch.tensor([[next_token_id]], device=cur_draft_logits.device)
                cur_draft_logits, draft_cache = self.draft_backend.decode(step_tensor, cache=draft_cache)

            total_drafted += len(draft_candidates)

            # 2. Target verification phase & rejection sampling
            accepted_in_step: List[int] = []
            cur_target_logits = target_logits
            rejected = False
            num_rejected_draft_tokens = 0

            for i, cand_id in enumerate(draft_candidates):
                target_probs = torch.softmax(cur_target_logits, dim=-1)
                draft_probs = draft_probs_list[i]

                # Perform rejection sampling check
                is_accepted, token_id = rejection_sample_token(
                    draft_probs=draft_probs,
                    target_probs=target_probs,
                    draft_token=cand_id,
                )

                if is_accepted:
                    accepted_in_step.append(cand_id)
                    cand_tensor = torch.tensor([[cand_id]], device=cur_target_logits.device)
                    cur_target_logits, target_cache = self.target_backend.decode(cand_tensor, cache=target_cache)
                else:
                    accepted_in_step.append(token_id)
                    cand_tensor = torch.tensor([[token_id]], device=cur_target_logits.device)
                    cur_target_logits, target_cache = self.target_backend.decode(cand_tensor, cache=target_cache)
                    rejected = True
                    # Number of draft tokens to roll back from draft KV cache
                    num_rejected_draft_tokens = len(draft_candidates) - i
                    break

            total_accepted += len(accepted_in_step)
            generated_count += len(accepted_in_step)

            # 3. KV Cache Rollback handling on rejection
            if rejected and draft_cache is not None:
                if hasattr(draft_cache, "rollback"):
                    draft_cache.rollback(num_rejected_draft_tokens)
                elif hasattr(draft_cache, "crop"):
                    draft_cache.crop(max(0, len(draft_cache) - num_rejected_draft_tokens))

            # Update output token sequence tensor
            new_tokens_tensor = torch.tensor([accepted_in_step], device=current_ids.device)
            current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)
            target_logits = cur_target_logits

        output_list = current_ids[0].tolist()
        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0

        return SpeculativeResult(
            output_ids=output_list,
            total_drafted_tokens=total_drafted,
            total_accepted_tokens=total_accepted,
            acceptance_rate=acc_rate,
            num_steps=step_count,
        )
