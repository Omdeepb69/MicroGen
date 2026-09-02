"""Unit tests for microgen execution profiler."""

import time
import pytest
from microgen.profiling import Profiler


def test_profiler_cpu_timing():
    profiler = Profiler(enable_cuda_sync=False)

    with profiler.profile("prefill"):
        time.sleep(0.02)  # ~20ms

    durations = profiler.get_raw_durations("prefill")
    assert len(durations) == 1
    assert durations[0] >= 15.0  # Should be at least ~15-20ms


def test_profiler_multiple_events_and_stats():
    profiler = Profiler(enable_cuda_sync=False)

    for _ in range(5):
        with profiler.profile("decode"):
            time.sleep(0.005)

    with profiler.profile("sampling"):
        time.sleep(0.01)

    stats = profiler.get_stats()
    assert "decode" in stats
    assert "sampling" in stats

    decode_stats = stats["decode"]
    assert decode_stats["count"] == 5.0
    assert decode_stats["avg_ms"] > 0.0
    assert decode_stats["min_ms"] <= decode_stats["max_ms"]

    sampling_stats = profiler.get_stats("sampling")
    assert sampling_stats["count"] == 1.0


def test_profiler_empty_stats():
    profiler = Profiler(enable_cuda_sync=False)
    stats = profiler.get_stats("nonexistent")

    assert stats["count"] == 0.0
    assert stats["total_ms"] == 0.0
    assert stats["avg_ms"] == 0.0


def test_profiler_reset():
    profiler = Profiler(enable_cuda_sync=False)

    with profiler.profile("test_block"):
        pass

    assert len(profiler.get_raw_durations("test_block")) == 1
    profiler.reset()
    assert len(profiler.get_raw_durations("test_block")) == 0
