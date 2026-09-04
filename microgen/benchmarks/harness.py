"""
Empirical Experiment Harness & Statistical Collector for LLM Inference Benchmarking.

Enforces isolated trial protocols (gc collect, CUDA empty_cache, peak memory reset),
warmup execution, host-device synchronization, percentile statistics (p50, p90, p95, p99),
and structured JSONL output logging to results/raw/experiments.jsonl.
"""

import datetime
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
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
    """Computes mean, std, min, max, iqr, and percentiles (p50, p90, p95, p99) for a list of values."""
    if not values:
        return {
            "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "iqr": 0.0,
        }
    
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    
    p25 = compute_percentile(values, 25.0)
    p75 = compute_percentile(values, 75.0)
    
    return {
        "p50": compute_percentile(values, 50.0),
        "p90": compute_percentile(values, 90.0),
        "p95": compute_percentile(values, 95.0),
        "p99": compute_percentile(values, 99.0),
        "mean": mean_val,
        "std": std_val,
        "min": min(values),
        "max": max(values),
        "iqr": max(0.0, p75 - p25),
    }


def compute_paired_p_value(sample_a: List[float], sample_b: List[float]) -> float:
    """Computes paired two-sided Wilcoxon signed-rank test p-value (or fallback paired t-test)."""
    if len(sample_a) != len(sample_b) or len(sample_a) < 2:
        return 1.0
    try:
        import scipy.stats as stats
        # Check if all paired differences are zero
        diffs = [a - b for a, b in zip(sample_a, sample_b)]
        if all(abs(d) < 1e-9 for d in diffs):
            return 1.0
        _, p_val = stats.wilcoxon(sample_a, sample_b)
        return float(p_val)
    except Exception:
        try:
            import scipy.stats as stats
            _, p_val = stats.ttest_rel(sample_a, sample_b)
            return float(p_val)
        except Exception:
            return 1.0


def reset_environment() -> None:
    """Resets memory allocator state and synchronizes host/device hardware."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_git_commit() -> str:
    """Returns current git commit hash if inside a git repository."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "git-commit-unknown"


@dataclass
class ExperimentConfig:
    """Configuration metadata and provenance for an empirical inference experiment."""
    model_name: str
    optimization_name: str
    baseline_type: str  # 'hf_pytorch', 'microgen_unoptimized', 'microgen_optimized'
    n_trials: int = 30
    warmup_trials: int = 5
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "results/raw"
    jsonl_filename: str = "experiments.jsonl"
    git_commit: str = field(default_factory=get_git_commit)
    wall_clock_timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_version: str = field(default_factory=lambda: torch.version.cuda if torch.cuda.is_available() else "none")
    hardware_name: str = field(default_factory=lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


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
    acceptance_rate: float = 1.0


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
    acceptance_rate_stats: Dict[str, float] = field(default_factory=dict)

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
            "acceptance_rate_stats": self.acceptance_rate_stats,
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
            acc_rate = float(metrics.get("acceptance_rate", 1.0))

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
                acceptance_rate=acc_rate,
            )
            trial_results.append(trial_res)

        # 3. Compute statistical summaries
        ttft_vals = [t.ttft_ms for t in trial_results]
        tpot_vals = [t.tpot_ms for t in trial_results]
        tps_vals = [t.tokens_per_sec for t in trial_results]
        alloc_vals = [t.peak_allocated_mb for t in trial_results]
        res_vals = [t.peak_reserved_mb for t in trial_results]
        acc_vals = [t.acceptance_rate for t in trial_results]

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
            acceptance_rate_stats=compute_stats(acc_vals),
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
