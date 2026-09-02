"""Automated performance diagnostic engine and bottleneck detector analyzing profiler traces."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from microgen.profiling.profiler import Profiler


@dataclass
class DiagnosticReport:
    """Structured report produced by the diagnostic engine."""

    primary_bottleneck: str
    prefill_decode_ratio: float
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DiagnosticEngine:
    """Analyzes execution profiler statistics to detect performance bottlenecks."""

    def analyze(self, profiler: Profiler) -> DiagnosticReport:
        """Analyze profiler stats and produce a DiagnosticReport."""
        stats = profiler.get_stats()

        prefill_ms = stats.get("prefill", {}).get("total_ms", 0.0)
        decode_ms = stats.get("decode", {}).get("total_ms", 0.0)
        sampling_ms = stats.get("sampling", {}).get("total_ms", 0.0)

        total_tracked_ms = prefill_ms + decode_ms + sampling_ms

        prefill_decode_ratio = (
            round(prefill_ms / decode_ms, 4) if decode_ms > 0 else float("inf") if prefill_ms > 0 else 0.0
        )

        recommendations: List[str] = []
        primary_bottleneck = "balanced"

        if total_tracked_ms == 0.0:
            primary_bottleneck = "unknown"
            recommendations.append("No profiler events recorded. Enable profiling around prefill and decode passes.")
            return DiagnosticReport(
                primary_bottleneck=primary_bottleneck,
                prefill_decode_ratio=prefill_decode_ratio,
                recommendations=recommendations,
                metrics=stats,
            )

        if prefill_ms / total_tracked_ms >= 0.6:
            primary_bottleneck = "prefill"
            recommendations.append("Prefill phase dominates execution runtime (>60%).")
            recommendations.append("Consider enabling PrefixKVCache to reuse prompt prefix key-value states.")
        elif decode_ms / total_tracked_ms >= 0.6:
            primary_bottleneck = "decode"
            recommendations.append("Decode phase dominates execution runtime (>60%) due to memory bandwidth bounds.")
            recommendations.append("Consider increasing batch size or continuous batching concurrency.")
        elif sampling_ms / total_tracked_ms >= 0.3:
            primary_bottleneck = "sampling"
            recommendations.append("Sampling overhead is high (>30%).")
            recommendations.append("Consider using greedy decoding or optimizing top-k/top-p sampling.")

        if not recommendations:
            recommendations.append("Execution time is evenly balanced across prefill and decode phases.")

        return DiagnosticReport(
            primary_bottleneck=primary_bottleneck,
            prefill_decode_ratio=prefill_decode_ratio,
            recommendations=recommendations,
            metrics=stats,
        )
