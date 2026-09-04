"""Unit tests for microgen package namespace re-exports."""

import pytest
import microgen


def test_top_level_exports():
    assert hasattr(microgen, "__version__")
    assert microgen.__version__ == "1.0.0"
    assert hasattr(microgen, "LLMEngine")
    assert hasattr(microgen, "backends")
    assert hasattr(microgen, "memory")
    assert hasattr(microgen, "caching")
    assert hasattr(microgen, "scheduler")
    assert hasattr(microgen, "engine")
    assert hasattr(microgen, "profiling")
    assert hasattr(microgen, "benchmarks")


def test_memory_namespace():
    from microgen.memory import PagedKVCacheAllocator, KVCacheState, KVCacheManager, BlockTable, PhysicalBlock
    assert PagedKVCacheAllocator is not None
    assert KVCacheState is not None
    assert KVCacheManager is not None
    assert BlockTable is not None
    assert PhysicalBlock is not None


def test_backends_namespace():
    from microgen.backends import PyTorchBackend, QuantizedPyTorchBackend, TensorParallelPyTorchBackend, InferenceBackend
    assert PyTorchBackend is not None
    assert QuantizedPyTorchBackend is not None
    assert TensorParallelPyTorchBackend is not None
    assert InferenceBackend is not None


def test_caching_namespace():
    from microgen.caching import PrefixKVCache, TokenBucketRateLimiter
    assert PrefixKVCache is not None
    assert TokenBucketRateLimiter is not None


def test_scheduler_namespace():
    from microgen.scheduler import ContinuousBatchingScheduler, RequestQueue
    assert ContinuousBatchingScheduler is not None
    assert RequestQueue is not None


def test_engine_namespace():
    from microgen.engine import LLMEngine, SpeculativeEngine, SpeculativeResult
    assert LLMEngine is not None
    assert SpeculativeEngine is not None
    assert SpeculativeResult is not None


def test_profiling_namespace():
    from microgen.profiling import Profiler, DiagnosticEngine
    assert Profiler is not None
    assert DiagnosticEngine is not None


def test_benchmarks_namespace():
    from microgen.benchmarks import WorkloadGenerator, WorkloadSuite, WorkloadRequest, ExperimentHarness, ExperimentResult
    assert WorkloadGenerator is not None
    assert WorkloadSuite is not None
    assert WorkloadRequest is not None
    assert ExperimentHarness is not None
    assert ExperimentResult is not None
