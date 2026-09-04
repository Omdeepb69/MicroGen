"""Paper-grade benchmark harness and workload generation tools."""

from microgen.benchmarks.workloads import WorkloadGenerator, WorkloadSuite, WorkloadRequest
from microgen.benchmarks.harness import ExperimentHarness, ExperimentResult, ExperimentConfig, TrialResult
from microgen.benchmarks.correctness import (
    validate_greedy_decoding_identity,
    validate_logit_similarity,
    run_all_correctness_gates,
)

__all__ = [
    "WorkloadGenerator",
    "WorkloadSuite",
    "WorkloadRequest",
    "ExperimentHarness",
    "ExperimentResult",
    "ExperimentConfig",
    "TrialResult",
    "validate_greedy_decoding_identity",
    "validate_logit_similarity",
    "run_all_correctness_gates",
]
