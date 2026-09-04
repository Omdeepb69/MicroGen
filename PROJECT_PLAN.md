# Project Plan

This file is the persistent source of truth for project progress. It
survives across conversations and context resets. Conversation history is
not a record of project state — this file is. Update it after every
completed task, not at the end of a session.

## Goal

MicroGen: An Empirical Study of Memory, Latency, and Throughput Trade-offs in Modular LLM Inference. MicroGen provides a modular, hardware-aware execution framework to systematically evaluate optimization trade-offs (Paged KV, Prefix Caching, Quantization, Speculative Decoding, Tensor Parallelism, Continuous Batching) against established baselines (Hugging Face / PyTorch reference, MicroGen unoptimized baseline, MicroGen optimizations) under controlled hardware, workload, memory, and latency constraints for academic publication.

## Core Research Questions (RQs)

- **RQ1 (Optimization Efficacy Regimes)**: Under what specific model parameter scales, sequence lengths, and prompt structures do individual inference optimizations (Paged KV, Prefix Caching, INT8 Quantization, Speculative Decoding, Tensor Parallelism) yield positive vs. negative performance gains?
- **RQ2 (Memory/Throughput Trade-offs)**: What are the exact memory footprint trade-offs (allocated vs. reserved VRAM, quantization stage overhead, KV cache compression Pareto curves)?
- **RQ3 (Serving Under Realistic Load)**: How do dynamic batching, paged memory management, and prefix caching behave under heterogeneous concurrent request streams ($B \in [1..64]$)?
- **RQ4 (Optimization Regime Characterization & Generalization)**: Can empirical measurements characterize workload regimes in which different inference optimizations are advantageous? Do these relationships generalize across non-GPT-2 architecture families (Qwen2.5, Llama-3.2) and hardware accelerator generations (NVIDIA P100 Pascal vs T4 Turing)?

## Current Phase

Phase 23 — Multi-Architecture & Multi-GPU Empirical Expansion (MLSys Main-Track Scaling)

## Status

- [x] Phase 1 — Initial Scaffolding and Simple KV Cache Proof-of-Concept
- [x] Phase 2 — Custom Generation Loop
- [x] Phase 3 — Initial Benchmarking
- [x] Phase 4 — Core Engine & Hardware Abstraction (CPU / GPU Backends)
- [x] Phase 5 — Request Queue, Scheduling & Continuous Batching
- [x] Phase 6 — Caching Infrastructure, Rate Limiting & Utilities
- [x] Phase 7 — Granular Profiling & Automated Performance Diagnostics
- [x] Phase 8 — HTTP API, SSE Streaming & Unified CLI
- [x] Phase 9 — E2E Benchmarking Suite & Hardware Comparison
- [x] Phase 10 — Paged KV Cache & Memory Eviction Architecture
- [x] Phase 11 — Speculative Decoding & Draft Model Acceleration
- [x] Phase 12 — Weight Quantization & Hardware Int8/FP8 Backends
- [x] Phase 13 — Kaggle & Multi-GPU Benchmark Suite
- [x] Phase 14 — Rigorous Experimental Harness & Workload Suite
- [x] Phase 15 — Empirical Micro-Ablation Experiments (RQ1 & RQ2)
- [x] Phase 16 — Serving Concurrency & Memory Pressure Experiments (RQ3)
- [x] Phase 17 — Model & Hardware Generalization (RQ4)
- [x] Phase 18 — Paper Data Synthesis, Figures & Reproducibility Package
- [x] Phase 19 — Empirical Benchmark Hardening & Verification Protocol
- [x] Phase 20 — Academic Research Paper Writing & Synthesis
- [x] Phase 21 — Academic Paper Hardening, Claim Precision & Empirical Realism Alignment
- [x] Phase 22 — Research Rigor, Empirical Audit & Manuscript Hardening
- [ ] Phase 23 — Multi-Architecture & Multi-GPU Empirical Expansion (MLSys Main-Track Scaling)

---

## Planned Phases & Tasks

### Phase 23 — Multi-Architecture & Multi-GPU Empirical Expansion (MLSys Main-Track Scaling)
- [x] Task 23.1.1: Modern Architecture Family Support (Qwen2.5 & Llama-3.2 Engine Adaptors) (`microgen/backends/`, `experiments/model_generalization.py`)
- [x] Task 23.1.2: Multi-GPU Tensor Parallel Scaling Verification (`ParallelBackend` on T4x2) (`experiments/tensor_parallel_scaling.py`)
- [x] Task 23.1.3: Cross-Generation Hardware Duality Sweep (P100 Pascal vs T4 Turing) (`experiments/hardware_duality.py`)
- [ ] Task 23.1.4: Paper Manuscript & Table Generator Expansion for MLSys Target (`scripts/export_paper_tables.py`, `paper/sections/`)

---

## Current Task

### Task 23.1.4: Paper Manuscript & Table Generator Expansion for MLSys Target
- **Objective**: Expand LaTeX paper manuscript (`paper/main.tex`, `paper/sections/`), table exporter (`scripts/export_paper_tables.py`), and figure generator (`scripts/generate_paper_figures.py`) to report multi-architecture generalization (Qwen2.5 / Llama-3.2), multi-GPU tensor-parallel scaling (T4x2), and hardware architecture duality (P100 vs T4). Compile updated PDF without errors or missing references.
- **Depends on**: Task 23.1.3.
- **Scope**: `scripts/export_paper_tables.py`, `scripts/generate_paper_figures.py`, `paper/sections/*.tex`, `paper/main.tex`.
- **Constraints**: Maintain strict consistency between raw `results/raw/experiments.jsonl` data, exported LaTeX tables, and manuscript prose across all sections.
- **Verification Criterion**: `python scripts/export_paper_tables.py` and `python scripts/generate_paper_figures.py` complete cleanly, and `cd paper && pdflatex -interaction=nonstopmode main.tex` produces a clean 11+ page PDF without broken references or unpopulated tables.




---

## Log

### 2026-09-04 — Task 23.1.3 completed
- Changed: Enhanced `experiments/hardware_duality.py` to extract device hardware metadata (`device_name`, `cuda_compute_capability`, `arch_generation`, `has_tensor_cores`) and updated `tests/experiments/test_hardware_duality.py` to assert hardware property fields.
- Files: `experiments/hardware_duality.py`, `tests/experiments/test_hardware_duality.py`, `PROJECT_PLAN.md`.
- Verified: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/experiments/test_hardware_duality.py` passed 3/3 tests cleanly.

### 2026-09-04 — Task 23.1.2 completed
- Changed: Implemented `experiments/tensor_parallel_scaling.py` and `tests/experiments/test_tensor_parallel_scaling.py` to benchmark 1-rank vs 2-rank `TensorParallelPyTorchBackend` execution. Integrated `experiments/tensor_parallel_scaling.py` into `scripts/run_all_paper_experiments.py`.
- Files: `experiments/tensor_parallel_scaling.py`, `tests/experiments/test_tensor_parallel_scaling.py`, `scripts/run_all_paper_experiments.py`, `PROJECT_PLAN.md`.
- Verified: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/experiments/test_tensor_parallel_scaling.py` passed 3/3 tests cleanly.


### 2026-09-04 — Task 23.1.1 completed
- Changed: Enabled modern open-weights architecture family support (`Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with `trust_remote_code=True` across `PyTorchBackend`, `QuantizedPyTorchBackend`, `WorkloadGenerator`, and `experiments/model_generalization.py`. Added multi-architecture unit tests.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/quantized.py`, `benchmarks/workloads.py`, `experiments/model_generalization.py`, `tests/experiments/test_model_generalization.py`, `pytest.ini`, `PROJECT_PLAN.md`.
- Verified: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/experiments/test_model_generalization.py` passed 4/4 test cases cleanly.


### 2026-09-04 — Phase 22 complete & Phase 23 planned
- Changed: Finalized Phase 22 editorial, statistical, and manuscript consistency pass. Formulated Phase 23 plan for multi-architecture model expansion (Qwen2.5 & Llama-3.2), multi-GPU tensor parallelism (`ParallelBackend` on T4x2), and cross-generation hardware duality (P100 vs T4).
- Files: `PROJECT_PLAN.md`, `ARCHITECTURE.md`
- Verified: All 6 tasks in Phase 22 verified and completed; `paper/main.pdf` compiled cleanly (11 pages, 0 errors).

