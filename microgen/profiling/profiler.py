"""CUDA and CPU execution event profiler for fine-grained runtime latency diagnostic tracking."""

from contextlib import contextmanager
import time
from typing import Dict, List, Optional, Generator, Any
import torch


class ProfileEvent:
    """Dataclass holding recorded execution timing for a named section."""

    def __init__(self, name: str, duration_ms: float) -> None:
        self.name = name
        self.duration_ms = duration_ms
        self.timestamp = time.time()


class Profiler:
    """Hardware-aware profiler supporting CPU and CUDA event timing."""

    def __init__(self, enable_cuda_sync: bool = False) -> None:
        self.enable_cuda_sync = enable_cuda_sync and torch.cuda.is_available()
        self._events: Dict[str, List[float]] = {}

    @contextmanager
    def profile(self, name: str) -> Generator[None, None, None]:
        """Context manager measuring execution latency of a named code block."""
        if self.enable_cuda_sync:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()
                duration_ms = start_event.elapsed_time(end_event)
                self._record_duration(name, duration_ms)
        else:
            start_time = time.perf_counter()
            try:
                yield
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000.0
                self._record_duration(name, duration_ms)

    def _record_duration(self, name: str, duration_ms: float) -> None:
        if name not in self._events:
            self._events[name] = []
        self._events[name].append(duration_ms)

    def get_raw_durations(self, name: str) -> List[float]:
        """Return list of recorded durations (in ms) for a given event name."""
        return list(self._events.get(name, []))

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregated latency statistics (total, count, avg, min, max, p95).

        If name is None, returns aggregated statistics across all recorded event categories.
        """
        if name is not None:
            durations = self._events.get(name, [])
            return self._compute_summary_stats(durations)

        summary: Dict[str, Any] = {}
        for event_name, durations in self._events.items():
            summary[event_name] = self._compute_summary_stats(durations)
        return summary

    def _compute_summary_stats(self, durations: List[float]) -> Dict[str, float]:
        if not durations:
            return {
                "count": 0.0,
                "total_ms": 0.0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
            }

        sorted_durations = sorted(durations)
        count = len(sorted_durations)
        total_ms = sum(sorted_durations)
        avg_ms = total_ms / count
        min_ms = sorted_durations[0]
        max_ms = sorted_durations[-1]

        # Calculate 95th percentile
        p95_idx = int(0.95 * (count - 1))
        p95_ms = sorted_durations[p95_idx]

        return {
            "count": float(count),
            "total_ms": round(total_ms, 4),
            "avg_ms": round(avg_ms, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "p95_ms": round(p95_ms, 4),
        }

    def reset(self) -> None:
        """Clear all recorded profiler event timings."""
        self._events.clear()
