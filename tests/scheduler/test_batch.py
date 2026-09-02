"""Unit tests for microgen scheduler Batch dataclass and batching utilities."""

import pytest
import torch
from microgen.scheduler import (
    Request,
    RequestStatus,
    Batch,
    create_prefill_batch,
    create_decode_batch,
    update_requests_with_sampled_tokens,
)


def test_create_prefill_batch_left_padding():
    req1 = Request("req-1", "p1", [10, 20])        # length 2
    req2 = Request("req-2", "p2", [100, 200, 300]) # length 3

    batch = create_prefill_batch("b-1", [req1, req2], pad_token_id=0)

    assert batch.batch_id == "b-1"
    assert batch.batch_size == 2
    assert batch.max_seq_len == 3
    assert batch.is_prefill is True

    # req1 should have 1 left pad token (0) then [10, 20]
    assert torch.equal(batch.input_ids[0], torch.tensor([0, 10, 20]))
    assert torch.equal(batch.attention_mask[0], torch.tensor([0, 1, 1]))

    # req2 should have 0 pad tokens and [100, 200, 300]
    assert torch.equal(batch.input_ids[1], torch.tensor([100, 200, 300]))
    assert torch.equal(batch.attention_mask[1], torch.tensor([1, 1, 1]))


def test_create_decode_batch():
    req1 = Request("req-1", "p1", [10])
    req2 = Request("req-2", "p2", [100])

    batch = create_decode_batch("b-decode", [req1, req2], next_token_ids=[42, 99])

    assert batch.batch_size == 2
    assert batch.max_seq_len == 1
    assert batch.is_prefill is False
    assert batch.input_ids.shape == (2, 1)
    assert batch.attention_mask.shape == (2, 1)

    assert torch.equal(batch.input_ids, torch.tensor([[42], [99]]))
    assert torch.equal(batch.attention_mask, torch.tensor([[1], [1]]))


def test_update_requests_with_sampled_tokens():
    req1 = Request("req-1", "p1", [1], max_new_tokens=2)
    req2 = Request("req-2", "p2", [2], max_new_tokens=5)

    req1.status = RequestStatus.RUNNING
    req2.status = RequestStatus.RUNNING

    # Step 1: append token 50 to req1, token 50256 (EOS) to req2
    completed = update_requests_with_sampled_tokens(
        [req1, req2], sampled_token_ids=[50, 50256], eos_token_id=50256
    )

    assert req1.generated_token_ids == [50]
    assert req1.status == RequestStatus.RUNNING

    assert req2.generated_token_ids == [50256]
    assert req2.status == RequestStatus.COMPLETED
    assert completed == ["req-2"]

    # Step 2: append token 51 to req1 (reaching max_new_tokens=2)
    completed_2 = update_requests_with_sampled_tokens(
        [req1], sampled_token_ids=[51], eos_token_id=50256
    )

    assert req1.generated_token_ids == [50, 51]
    assert req1.status == RequestStatus.COMPLETED
    assert completed_2 == ["req-1"]
