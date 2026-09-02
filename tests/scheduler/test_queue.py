"""Unit tests for microgen scheduler Request dataclass and thread-safe RequestQueue."""

import concurrent.futures
import time
import pytest
from microgen.scheduler import Request, RequestStatus, RequestQueue


def test_request_dataclass_properties():
    req = Request(
        request_id="req-100",
        prompt="Hello world",
        prompt_ids=[101, 102],
        max_new_tokens=20,
    )
    assert req.status == RequestStatus.PENDING
    assert req.is_finished is False
    assert req.num_prompt_tokens == 2
    assert req.num_generated_tokens == 0
    assert req.total_sequence_length == 2

    req.generated_token_ids.extend([201, 202, 203])
    assert req.num_generated_tokens == 3
    assert req.total_sequence_length == 5

    req.status = RequestStatus.COMPLETED
    assert req.is_finished is True


def test_request_queue_fifo_and_priority():
    queue = RequestQueue()
    assert queue.is_empty() is True
    assert queue.size() == 0

    req_low = Request("req-low", "low", [1], priority=1)
    req_high = Request("req-high", "high", [2], priority=10)
    req_mid = Request("req-mid", "mid", [3], priority=5)

    queue.enqueue(req_low)
    queue.enqueue(req_high)
    queue.enqueue(req_mid)

    assert queue.size() == 3

    # Dequeue should yield highest priority first (req-high)
    pop1 = queue.dequeue()
    assert pop1 is not None
    assert pop1.request_id == "req-high"
    assert pop1.status == RequestStatus.RUNNING

    # Next should be req-mid
    pop2 = queue.dequeue()
    assert pop2 is not None
    assert pop2.request_id == "req-mid"

    # Last should be req-low
    pop3 = queue.dequeue()
    assert pop3 is not None
    assert pop3.request_id == "req-low"

    assert queue.is_empty() is True


def test_request_queue_pop_batch_and_cancel():
    queue = RequestQueue()

    reqs = [
        Request(f"req-{i}", f"prompt {i}", [i], priority=i) for i in range(5)
    ]
    for r in reqs:
        queue.enqueue(r)

    # Cancel req-4 (highest priority)
    cancelled = queue.cancel("req-4")
    assert cancelled is True
    cancelled_req = queue.get("req-4")
    assert cancelled_req is not None
    assert cancelled_req.status == RequestStatus.CANCELLED

    # Pop batch of size 2 -> should get req-3, req-2
    batch = queue.pop_batch(max_batch_size=2)
    assert len(batch) == 2
    assert batch[0].request_id == "req-3"
    assert batch[1].request_id == "req-2"


def test_request_queue_multithreaded_safety():
    queue = RequestQueue()
    num_requests = 100

    def producer(i):
        req = Request(f"thread-req-{i}", f"prompt {i}", [i])
        queue.enqueue(req)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(producer, i) for i in range(num_requests)]
        concurrent.futures.wait(futures)

    assert queue.size() == num_requests

    dequeued_count = 0
    while not queue.is_empty():
        req = queue.dequeue()
        if req is not None:
            dequeued_count += 1

    assert dequeued_count == num_requests
