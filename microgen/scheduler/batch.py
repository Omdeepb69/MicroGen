"""Batch dataclass and batching utilities for static and dynamic LLM inference."""

from dataclasses import dataclass
import time

from typing import List, Optional, Tuple
import torch

from microgen.scheduler.queue import Request, RequestStatus


@dataclass
class Batch:
    """Dataclass holding padded batch tensors and associated requests."""

    batch_id: str
    requests: List[Request]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    is_prefill: bool
    max_seq_len: int

    @property
    def batch_size(self) -> int:
        """Return the number of requests in the batch."""
        return len(self.requests)


def create_prefill_batch(
    batch_id: str,
    requests: List[Request],
    pad_token_id: int = 0,
    device: Optional[torch.device] = None,
) -> Batch:
    """Construct a left-padded prefill batch tensor and 2D attention mask for a list of requests.

    Left padding ensures that generated new tokens begin at aligned sequence positions.
    """
    if not requests:
        raise ValueError("Cannot create prefill batch from empty requests list.")

    max_seq_len = max(req.num_prompt_tokens for req in requests)
    batch_size = len(requests)

    input_ids_tensor = torch.full(
        (batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask_tensor = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.long, device=device
    )

    for idx, req in enumerate(requests):
        seq_len = req.num_prompt_tokens
        pad_len = max_seq_len - seq_len
        input_ids_tensor[idx, pad_len:] = torch.tensor(
            req.prompt_ids, dtype=torch.long, device=device
        )
        attention_mask_tensor[idx, pad_len:] = 1

    return Batch(
        batch_id=batch_id,
        requests=requests,
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
        is_prefill=True,
        max_seq_len=max_seq_len,
    )


def create_decode_batch(
    batch_id: str,
    requests: List[Request],
    next_token_ids: List[int],
    device: Optional[torch.device] = None,
) -> Batch:
    """Construct a single-token decode batch tensor (shape: [batch_size, 1])."""
    if not requests:
        raise ValueError("Cannot create decode batch from empty requests list.")
    if len(requests) != len(next_token_ids):
        raise ValueError("Length mismatch between requests and next_token_ids.")

    batch_size = len(requests)
    input_ids_tensor = torch.tensor(
        next_token_ids, dtype=torch.long, device=device
    ).unsqueeze(1)
    attention_mask_tensor = torch.ones(
        (batch_size, 1), dtype=torch.long, device=device
    )

    return Batch(
        batch_id=batch_id,
        requests=requests,
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
        is_prefill=False,
        max_seq_len=1,
    )


def update_requests_with_sampled_tokens(
    requests: List[Request],
    sampled_token_ids: List[int],
    eos_token_id: Optional[int] = None,
) -> List[str]:
    """Append sampled tokens to requests, transition completed requests, and return completed request IDs."""
    if len(requests) != len(sampled_token_ids):
        raise ValueError("Length mismatch between requests and sampled_token_ids.")

    completed_ids: List[str] = []
    current_time = time.perf_counter()

    for req, token_id in zip(requests, sampled_token_ids):
        if req.is_finished:
            continue

        req.generated_token_ids.append(token_id)

        is_eos = eos_token_id is not None and token_id == eos_token_id
        is_max_tokens = req.num_generated_tokens >= req.max_new_tokens

        if is_eos or is_max_tokens:
            req.status = RequestStatus.COMPLETED
            req.finish_time = current_time
            completed_ids.append(req.request_id)

    return completed_ids
