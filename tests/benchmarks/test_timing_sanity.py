"""Monotonic timing and latency verification tests for single-request and scheduler execution."""

import time
import pytest

from microgen.devices.cpu import CPUDevice
from microgen.backends.pytorch import PyTorchBackend
from microgen.runtime.kv_cache import KVCacheManager
from microgen.scheduler.queue import Request, RequestQueue, RequestStatus
from microgen.scheduler.batch import (
    create_prefill_batch,
    create_decode_batch,
    update_requests_with_sampled_tokens,
)
from microgen.scheduler.scheduler import ContinuousBatchingScheduler


def test_request_monotonic_timestamps_and_properties():
    """Verify that Request timestamps are monotonic, non-epoch, and compute valid latency properties."""
    req = Request(
        request_id="req-test-1",
        prompt="Hello world",
        prompt_ids=[101, 7592, 2088],
        max_new_tokens=4,
    )
    
    # Verify default arrival_time is relative/monotonic, not epoch scale (> 1.7e9 seconds)
    assert req.arrival_time > 0
    assert req.arrival_time < 1e9  # Monotonic perf_counter uptime is typically < 1e9 seconds
    
    t0 = time.perf_counter()
    req.start_time = t0
    req.first_token_time = t0 + 0.015  # 15 ms
    req.finish_time = t0 + 0.045       # 45 ms
    req.generated_token_ids = [1000, 1001, 1002, 1003]
    req.status = RequestStatus.COMPLETED
    
    # Assert monotonic order
    assert req.arrival_time <= req.start_time <= req.first_token_time <= req.finish_time
    
    # Verify TTFT, TPOT, and Total Latency properties
    assert 0.0 <= req.ttft_ms < 1000.0
    assert 0.0 <= req.tpot_ms < 1000.0
    assert 0.0 <= req.total_latency_ms < 1000.0
    
    # Explicitly assert no epoch-scale values (> 1e9 ms)
    assert req.ttft_ms < 1e6
    assert req.tpot_ms < 1e6
    assert req.total_latency_ms < 1e6


def test_scheduler_single_request_latency_bounds():
    """Verify end-to-end continuous batching single request latency measurement produces valid bounds."""
    device = CPUDevice()
    backend = PyTorchBackend(device=device)
    backend.load_model("sshleifer/tiny-gpt2")
    
    kv_manager = KVCacheManager()
    scheduler = ContinuousBatchingScheduler(
        backend=backend,
        kv_cache_manager=kv_manager,
        max_batch_size=2,
    )
    
    req = Request(
        request_id="req-single-1",
        prompt="The capital of France is",
        prompt_ids=[464, 3139, 286, 4881, 318],
        max_new_tokens=5,
    )
    
    scheduler.add_request(req)
    completed = scheduler.run_until_complete()
    
    assert len(completed) == 1
    finished_req = completed[0]
    
    assert finished_req.status == RequestStatus.COMPLETED
    assert finished_req.start_time is not None
    assert finished_req.first_token_time is not None
    assert finished_req.finish_time is not None
    
    # Check monotonic timestamps
    assert finished_req.arrival_time <= finished_req.start_time
    assert finished_req.start_time <= finished_req.first_token_time
    assert finished_req.first_token_time <= finished_req.finish_time
    
    # Latency assertions (must be realistic ms, positive, finite, non-epoch)
    ttft = finished_req.ttft_ms
    tpot = finished_req.tpot_ms
    total_lat = finished_req.total_latency_ms
    
    assert 0.0 < ttft < 10000.0, f"TTFT out of bounds: {ttft} ms"
    assert 0.0 < tpot < 10000.0, f"TPOT out of bounds: {tpot} ms"
    assert 0.0 < total_lat < 50000.0, f"Total latency out of bounds: {total_lat} ms"
