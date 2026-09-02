"""Unit tests for microgen automated performance diagnostic engine."""

import time
import pytest
from microgen.profiling import Profiler, DiagnosticEngine


def test_diagnostics_empty_profiler():
    profiler = Profiler(enable_cuda_sync=False)
    engine = DiagnosticEngine()
    report = engine.analyze(profiler)

    assert report.primary_bottleneck == "unknown"
    assert len(report.recommendations) >= 1


def test_diagnostics_prefill_bottleneck():
    profiler = Profiler(enable_cuda_sync=False)
    engine = DiagnosticEngine()

    with profiler.profile("prefill"):
        time.sleep(0.08)  # ~80ms

    with profiler.profile("decode"):
        time.sleep(0.01)  # ~10ms

    report = engine.analyze(profiler)
    assert report.primary_bottleneck == "prefill"
    assert report.prefill_decode_ratio > 1.0
    assert any("PrefixKVCache" in rec for rec in report.recommendations)


def test_diagnostics_decode_bottleneck():
    profiler = Profiler(enable_cuda_sync=False)
    engine = DiagnosticEngine()

    with profiler.profile("prefill"):
        time.sleep(0.005)  # ~5ms

    for _ in range(10):
        with profiler.profile("decode"):
            time.sleep(0.01)  # ~100ms total

    report = engine.analyze(profiler)
    assert report.primary_bottleneck == "decode"
    assert any("memory bandwidth" in rec for rec in report.recommendations)


def test_diagnostics_sampling_bottleneck():
    profiler = Profiler(enable_cuda_sync=False)
    engine = DiagnosticEngine()

    with profiler.profile("prefill"):
        time.sleep(0.01)

    with profiler.profile("decode"):
        time.sleep(0.01)

    with profiler.profile("sampling"):
        time.sleep(0.05)

    report = engine.analyze(profiler)
    assert report.primary_bottleneck == "sampling"
