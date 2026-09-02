"""Integration tests for ContinuousBatchingScheduler."""

import pytest
from transformers import AutoTokenizer
from microgen.devices import CPUDevice
from microgen.backends import PyTorchBackend
from microgen.runtime import KVCacheManager
from microgen.scheduler import Request, RequestStatus, ContinuousBatchingScheduler

MODEL_NAME = "sshleifer/tiny-gpt2"


def test_continuous_batching_scheduler_e2e():
    device = CPUDevice()
    backend = PyTorchBackend(device=device)
    backend.load_model(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    kv_cache_manager = KVCacheManager()

    scheduler = ContinuousBatchingScheduler(
        backend=backend,
        kv_cache_manager=kv_cache_manager,
        max_batch_size=2,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt1 = "The quick brown fox"
    prompt2 = "Once upon a time"
    prompt3 = "Machine learning is"

    req1 = Request(
        request_id="req-1",
        prompt=prompt1,
        prompt_ids=tokenizer.encode(prompt1),
        max_new_tokens=5,
        priority=1,
    )
    req2 = Request(
        request_id="req-2",
        prompt=prompt2,
        prompt_ids=tokenizer.encode(prompt2),
        max_new_tokens=4,
        priority=2,
    )
    req3 = Request(
        request_id="req-3",
        prompt=prompt3,
        prompt_ids=tokenizer.encode(prompt3),
        max_new_tokens=3,
        priority=3,
    )

    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)

    completed = scheduler.run_until_complete()

    assert len(completed) == 3
    for req in completed:
        assert req.status == RequestStatus.COMPLETED
        assert len(req.generated_token_ids) > 0
        assert req.finish_time is not None

    # Ensure KV caches were properly cleaned up
    assert kv_cache_manager.active_requests_count() == 0
    assert kv_cache_manager.get_total_memory_usage_bytes() == 0
