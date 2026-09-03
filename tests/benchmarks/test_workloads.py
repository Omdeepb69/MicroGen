"""
Unit tests for benchmarks/workloads.py module.
"""

import pytest
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite, WorkloadRequest


def test_workload_generator_short():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    suite = generator.generate_short_workload(num_requests=5, seed=42)

    assert suite.name == "short"
    assert len(suite.requests) == 5
    for req in suite.requests:
        assert 32 <= req.prompt_len <= 128
        assert len(req.prompt_ids) == req.prompt_len
        assert req.max_new_tokens == 32


def test_workload_generator_medium():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    suite = generator.generate_medium_workload(num_requests=5, seed=42)

    assert suite.name == "medium"
    assert len(suite.requests) == 5
    for req in suite.requests:
        assert 256 <= req.prompt_len <= 512
        assert len(req.prompt_ids) == req.prompt_len


def test_workload_generator_determinism():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    suite1 = generator.generate_heterogeneous_workload(num_requests=5, seed=123)
    suite2 = generator.generate_heterogeneous_workload(num_requests=5, seed=123)

    assert suite1.min_prompt_len == suite2.min_prompt_len
    assert suite1.max_prompt_len == suite2.max_prompt_len
    for req1, req2 in zip(suite1.requests, suite2.requests):
        assert req1.prompt_ids == req2.prompt_ids
        assert req1.prompt_text == req2.prompt_text


def test_workload_generator_shared_prefix():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    suite = generator.generate_shared_prefix_workload(
        num_requests=5,
        total_prompt_len=100,
        prefix_ratio=0.6,
        seed=42,
    )

    prefix_len = int(100 * 0.6)
    first_req_prefix = suite.requests[0].prompt_ids[:prefix_len]
    for req in suite.requests[1:]:
        assert req.prompt_ids[:prefix_len] == first_req_prefix
