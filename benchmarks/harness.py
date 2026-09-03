"""
Empirical Experiment Harness & Statistical Collector for LLM Inference Benchmarking.

Enforces isolated trial protocols (gc collect, CUDA empty_cache, peak memory reset),
warmup execution, host-device synchronization, percentile statistics (p50, p90, p95, p99),
and structured JSONL output logging to results/raw/experiments.jsonl.
"""

import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
import torch


def compute_percentile(data: List[float], percentile: float) -> float:
    """Computes an exact linear interpolation percentile for a list of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Computes p50, p90, p95, and p99 percentiles for a metric."""
    return {
        "p50": compute_percentile(values, 50.0),
        "p90": compute_percentile(values, 90.0),
        "p95": compute_percentile(values, 95.0),
        "p99": compute_percentile(values, 99.0),
        "mean": sum(values) / len(values) if values else 0.0,
    }


def reset_environment() -> None:
    """Resets memory allocator state and synchronizes host/device hardware."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


@dataclass
class ExperimentConfig:
    """Configuration metadata for an empirical inference experiment."""
    model_name: str
    optimization_name: str
    baseline_type: str  # 'hf_pytorch', 'microgen_unoptimized', 'microgen_optimized'
    n_trials: int = 30
    warmup_trials: int = 2
    device: str = "cpu"
    output_dir: str = "results/raw"
    jsonl_filename: str = "experiments.jsonl"


@dataclass
class TrialResult:
    """Raw performance metrics recorded for a single trial pass."""
    trial_index: int
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    tokens_per_sec: float
    generated_tokens: int
    peak_allocated_mb: float
    peak_reserved_mb: float


@dataclass
class ExperimentResult:
    """Aggregated empirical results and percentile statistics for an experiment."""
    config: ExperimentConfig
    workload_name: str
    num_requests: int
    trials: List[TrialResult]
    ttft_stats_ms: Dict[str, float]
    tpot_stats_ms: Dict[str, float]
    throughput_stats_tps: Dict[str, float]
    peak_allocated_mb_stats: Dict[str, float]
    peak_reserved_mb_stats: Dict[str, float]

    def to_json_dict(self) -> Dict[str, Any]:
        """Converts experiment result to JSON-serializable dictionary."""
        return {
            "config": asdict(self.config),
            "workload_name": self.workload_name,
            "num_requests": self.num_requests,
            "ttft_stats_ms": self.ttft_stats_ms,
            "tpot_stats_ms": self.tpot_stats_ms,
            "throughput_stats_tps": self.throughput_stats_tps,
            "peak_allocated_mb_stats": self.peak_allocated_mb_stats,
            "peak_reserved_mb_stats": self.peak_reserved_mb_stats,
            "num_trials_recorded": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
        }


class ExperimentHarness:
    """
    Isolated empirical experiment harness.
    
    Coordinates warmup passes, isolated trial execution, memory tracking,
    percentile statistics generation, and JSONL logging.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run_experiment(
        self,
        workload_name: str,
        num_requests: int,
        execution_fn: Callable[[], Dict[str, Any]],
    ) -> ExperimentResult:
        """
        Executes an empirical experiment suite.

        `execution_fn` must execute the workload once and return a dict containing:
          - 'ttft_ms': float
          - 'tpot_ms': float (Time Per Output Token / ITL)
          - 'generated_tokens': int
          - 'total_latency_ms': float
        """
        # 1. Warmup passes
        for _ in range(self.config.warmup_trials):
            reset_environment()
            try:
                execution_fn()
            except Exception:
                pass

        # 2. Benchmark trial loops
        trial_results: List[TrialResult] = []

        for trial_idx in range(self.config.n_trials):
            reset_environment()

            if torch.cuda.is_available() and "cuda" in self.config.device:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            metrics = execution_fn()
            if torch.cuda.is_available() and "cuda" in self.config.device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Record memory
            peak_alloc_bytes = 0
            peak_res_bytes = 0
            if torch.cuda.is_available() and "cuda" in self.config.device:
                peak_alloc_bytes = torch.cuda.max_memory_allocated()
                peak_res_bytes = torch.cuda.max_memory_reserved()

            ttft_ms = float(metrics.get("ttft_ms", (t1 - t0) * 1000.0))
            tpot_ms = float(metrics.get("tpot_ms", 0.0))
            gen_tokens = int(metrics.get("generated_tokens", 1))
            tot_latency_ms = float(metrics.get("total_latency_ms", (t1 - t0) * 1000.0))

            tps = (gen_tokens / (tot_latency_ms / 1000.0)) if tot_latency_ms > 0 else 0.0

            trial_res = TrialResult(
                trial_index=trial_idx,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_latency_ms=tot_latency_ms,
                tokens_per_sec=tps,
                generated_tokens=gen_tokens,
                peak_allocated_mb=peak_alloc_bytes / (1024 * 1024),
                peak_reserved_mb=peak_res_bytes / (1024 * 1024),
            )
            trial_results.append(trial_res)

        # 3. Compute statistical summaries
        ttft_vals = [t.ttft_ms for t in trial_results]
        tpot_vals = [t.tpot_ms for t in trial_results]
        tps_vals = [t.tokens_per_sec for t in trial_results]
        alloc_vals = [t.peak_allocated_mb for t in trial_results]
        res_vals = [t.peak_reserved_mb for t in trial_results]

        result = ExperimentResult(
            config=self.config,
            workload_name=workload_name,
            num_requests=num_requests,
            trials=trial_results,
            ttft_stats_ms=compute_stats(ttft_vals),
            tpot_stats_ms=compute_stats(tpot_vals),
            throughput_stats_tps=compute_stats(tps_vals),
            peak_allocated_mb_stats=compute_stats(alloc_vals),
            peak_reserved_mb_stats=compute_stats(res_vals),
        )

        # 4. Stream record to JSONL log
        self._write_to_jsonl(result)

        return result

    def _write_to_jsonl(self, result: ExperimentResult) -> None:
        """Appends experiment result as a JSON line to output log file."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, self.config.jsonl_filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_json_dict()) + "\n")
