"""Unit tests for microgen TokenBucketRateLimiter."""

import time
import concurrent.futures
import pytest
from microgen.caching import TokenBucketRateLimiter


def test_unlimited_rate_limiter():
    limiter = TokenBucketRateLimiter()

    assert limiter.check(num_requests=100, num_tokens=1000) is True
    assert limiter.acquire(num_requests=100, num_tokens=1000) is True


def test_rpm_rate_limiter_exhaustion():
    # 60 RPM = 1 request per second
    limiter = TokenBucketRateLimiter(max_rpm=2.0)

    assert limiter.acquire(num_requests=1) is True
    assert limiter.acquire(num_requests=1) is True
    # 3rd request should fail immediately
    assert limiter.check(num_requests=1) is False
    assert limiter.acquire(num_requests=1) is False


def test_tpm_rate_limiter_exhaustion():
    limiter = TokenBucketRateLimiter(max_tpm=100.0)

    assert limiter.acquire(num_tokens=60) is True
    assert limiter.acquire(num_tokens=40) is True
    # Remaining capacity 0, should fail
    assert limiter.acquire(num_tokens=10) is False


def test_rate_limiter_refill():
    # 600 RPM = 10 requests per second
    limiter = TokenBucketRateLimiter(max_rpm=600.0)

    # Exhaust 600 capacity
    assert limiter.acquire(num_requests=600) is True
    assert limiter.acquire(num_requests=1) is False

    # Sleep 0.1s -> refills 0.1 * 10 = 1 request token
    time.sleep(0.12)
    assert limiter.acquire(num_requests=1) is True


def test_rate_limiter_thread_safety():
    limiter = TokenBucketRateLimiter(max_rpm=50.0)

    def worker():
        return limiter.acquire(num_requests=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(60)]
        results = [f.result() for f in futures]

    # Exactly 50 acquisitions should succeed, 10 should fail
    assert results.count(True) == 50
    assert results.count(False) == 10
