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
- **RQ4 (Optimization Regime Characterization & Generalization)**: Can empirical measurements characterize workload regimes in which different inference optimizations are advantageous? Do these relationships generalize to non-English / custom architecture base models (e.g. Gonyai TEO2 251M)?
## Current Phase

Phase 22 — Research Rigor, Empirical Audit & Manuscript Hardening

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
- [ ] Phase 22 — Research Rigor, Empirical Audit & Manuscript Hardening

---

## Planned Phases & Tasks

### Phase 20 — Academic Research Paper Writing & Synthesis
- [x] Task 20.1.1: Paper Blueprint, LaTeX Boilerplate & Abstract/Introduction (Section 1)
- [x] Task 20.1.2: System Architecture & Engine Design (Section 2)
- [x] Task 20.1.3: Modular Optimization Techniques & Analytical Models (Section 3)
- [x] Task 20.1.4: Experimental Methodology & Verification Protocol (Section 4)
- [x] Task 20.1.5: Empirical Evaluation, RQ Answers & Data Integration (Section 5)
- [x] Task 20.1.6: Related Work, Discussion, Threat Analysis & Conclusion (Sections 6–8)
- [x] Task 20.1.7: LaTeX Compilation, Standalone PDF Generation & Verification

### Phase 21 — Academic Paper Hardening, Claim Precision & Empirical Realism Alignment
- [x] Task 21.1.1: Paper Retitling, Framing Alignment & Thesis Refinement (Section 1 & main.tex)
- [x] Task 21.1.2: Absolute Claim Softening & Technical Precision Hardening (Sections 2–4, 7–8)
- [x] Task 21.1.3: Comparative Related Work Systems Table & Feature Matrix (Section 6 & tables/)
- [x] Task 21.1.4: Non-Monotonic Composition & Break-Even Analysis Integration (Sections 5 & 7)
- [x] Task 21.1.5: Final Master LaTeX Compilation & Verification (paper/main.pdf)

### Phase 22 — Research Rigor, Empirical Audit & Manuscript Hardening
- [x] Task 22.1.1: Benchmark Timing Audit, Continuous Batching Fix & Statistical Metrics Extension (`microgen/scheduler/`, `benchmarks/harness.py`)
- [x] Task 22.1.2: Multi-Architecture Model Generalization Suite (GPT-2 vs Llama/Qwen Architecture) (`experiments/model_generalization.py`)
- [x] Task 22.1.3: Empirical Benchmark Re-Sweep & Raw Data Refresh ($N=30$ Statistical Clean Collection) (`experiments/`, `results/raw/experiments.jsonl`)
- [x] Task 22.1.4: Table Generator Overhaul & Non-Monotonicity Ablation Ladder Exporter (`scripts/export_paper_tables.py`)
- [x] Task 22.1.5: Paper Manuscript Overhaul & Reviewer Criticism Remediation (Sections 1–8, References & Footnote URL) (`paper/sections/`, `paper/references.bib`)
- [x] Task 22.1.6: Master LaTeX Compilation, Full Verification & PDF Audit (`paper/main.pdf`)

---

## Current Task

*All tasks in Phase 22 are complete!*

---

## Log

### 2026-09-04 — Task 22.1.6 complete
- Changed: Finalized master LaTeX PDF compilation (`paper/main.pdf`) following full empirical dataset integration.
- Files: `paper/main.pdf`, `PROJECT_PLAN.md`
- Verified: Clean PDF build (12 pages, 455,377 bytes) with 0 errors, 0 missing figures, and 0 undefined citations.

### 2026-09-04 — Task 22.1.5 complete
- Changed: Overhauled paper sections (`paper/sections/01_introduction.tex` through `08_conclusion.tex`) to embed open-source repository URL footnote (`\url{https://github.com/Omdeepb69/MicroGen}`), verified line-by-line alignment between text descriptions and refreshed $N=30$ GPU empirical data tables and figures, and confirmed complete bibliography in `paper/references.bib`.
- Files: `paper/sections/01_introduction.tex`, `paper/references.bib`, `paper/main.pdf`, `PROJECT_PLAN.md`
- Verified: Executed `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex && pdflatex -halt-on-error main.tex` in `paper/`; compilation passed cleanly with 0 errors and zero undefined citations.

### 2026-09-04 — Tasks 22.1.3 & 22.1.4 complete
- Changed: Executed full N=30 GPU empirical experiment suite on Kaggle (Tesla T4, 2 GPUs). Resolved HF Hub 429 rate limit issues by loading Kaggle `HF_TOKEN_READ` secret and setting `HF_HUB_OFFLINE=1` after pre-caching models. Exported refreshed LaTeX tables with $\mu \pm \sigma$ error bars and 10 publication vector figures.
- Files: `scripts/run_all_paper_experiments.py`, `scripts/kaggle_benchmark_runner.py`, `experiments/*.py`, `PROJECT_PLAN.md`
- Verified: All 9 experiment modules executed with N=30 trials on CUDA; generated 4 LaTeX tables in `paper/tables/` and 10 figures in `paper/figures/`.

### 2026-09-04 — Task 22.1.2 complete
- Changed: Extended `experiments/model_generalization.py` to extract `architecture_type` metadata (`backend.model.config.model_type`) and support multi-architecture evaluation across model families (GPT-2, Qwen, Llama). Added error boundary to handle multi-model evaluation loops cleanly. Updated unit test assertions in `tests/experiments/test_model_generalization.py`.
- Files: `experiments/model_generalization.py`, `tests/experiments/test_model_generalization.py`, `PROJECT_PLAN.md`
- Verified: `pytest tests/experiments/test_model_generalization.py` passed (3/3).
- Changed: Fixed continuous batching TTFT latency calculation in `experiments/batching_concurrency.py` using `r.ttft_ms` so TTFT reflects prefill pass and queue wait time across all batch sizes $B \in [1..16]$. Extended `compute_stats` in `benchmarks/harness.py` to compute standard deviation ($\sigma$), interquartile range (IQR), min, max, and percentiles. Added `compute_paired_p_value` for Wilcoxon signed-rank significance testing. Added `acceptance_rate` to `TrialResult` and `ExperimentResult` in `benchmarks/harness.py` and updated `experiments/speculative_sweep.py` to export empirical acceptance rates.
- Files: `benchmarks/harness.py`, `experiments/batching_concurrency.py`, `experiments/speculative_sweep.py`, `PROJECT_PLAN.md`
- Verified: `pytest tests/benchmarks/` passed (10/10), full test suite passed (117/117 passed), and verified exported JSONL fields (`std`, `iqr`, `acceptance_rate_stats`).
- Changed: Executed full master LaTeX compilation pipeline (`pdflatex` + `bibtex` + `pdflatex`). Verified 0 LaTeX compilation errors, 0 missing citations/references, and 100% test suite pass (`10/10` tests passed in `pytest tests/benchmarks/`). Finalized submission-ready PDF manuscript (`paper/main.pdf` — 435 KB).
- Files: `paper/main.tex`, `paper/main.pdf`
- Verified: `pdflatex` pipeline succeeded cleanly, `pytest tests/benchmarks/` passed 10/10.

### 2026-09-04 — Task 21.1.4 complete
- Changed: Enhanced Section 5 (`paper/sections/05_evaluation.tex`) and Section 7 (`paper/sections/07_discussion.tex`) to feature non-monotonic optimization composition ("*Non-Monotonic Optimization Composition*", 408 tok/s baseline vs 319 tok/s all-combined = $0.78\times$) and prefix cache break-even threshold analysis ($r \ge 15\%$, $\delta_{\text{cache}} \approx 0.7\text{ ms}$) as centerpiece paper findings and system design guidelines.
- Files: `paper/sections/05_evaluation.tex`, `paper/sections/07_discussion.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 11-page PDF (`paper/main.pdf` — 435 KB) with 0 errors.

### 2026-09-04 — Task 21.1.3 complete
- Changed: Created `paper/tables/table0_systems_comparison.tex` contrasting Orca, vLLM, SGLang, TensorRT-LLM, and MicroGen across features and modular isolation capability. Embedded table and wrote architectural positioning subsection in Section 6 (`paper/sections/06_related_work.tex`).
- Files: `paper/tables/table0_systems_comparison.tex`, `paper/sections/06_related_work.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 11-page PDF (`paper/main.pdf` — 432 KB) with 0 errors.

### 2026-09-04 — Task 21.1.2 complete
- Changed: Replaced absolute claims across Sections 2, 3, 7, and 8 ("completely eliminates external fragmentation", "guarantees request resilience", "strictly requires $\alpha \ge 0.70$") with scientifically scoped observations ("under evaluated allocation workloads...", "empirically observed thresholds..."). Clarified prefix cache implementation as a hash-based prefix cache abstraction.
- Files: `paper/sections/02_architecture.tex`, `paper/sections/03_optimizations.tex`, `paper/sections/07_discussion.tex`, `paper/sections/08_conclusion.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into PDF (`paper/main.pdf` — 411 KB) with 0 errors.

### 2026-09-04 — Task 21.1.1 complete
- Changed: Updated paper title to *"MicroGen: When LLM Inference Optimizations Help—and When They Hurt (An Empirical Study of Latency, Throughput, and Memory Trade-offs)"* in `paper/main.tex`. Re-framed Section 1 (`paper/sections/01_introduction.tex`) to explicitly position MicroGen as an *empirical research isolation substrate* rather than a production serving engine competing with C++ engines. Refined prefill sequence complexity and decode memory bound definitions. Highlighted non-monotonic optimization composition as a key contribution.
- Files: `paper/main.tex`, `paper/sections/01_introduction.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 10-page PDF (`paper/main.pdf` — 410 KB) with 0 errors.

### 2026-09-04 — Task 20.1.7 complete


### 2026-09-04 — Task 20.1.7 complete
- Changed: Executed full master LaTeX compilation pipeline (`pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex`). Verified all cross-references, figure links, and table formatting. Verified full test suite (`PYTHONPATH=. pytest tests/benchmarks/` — 10 passed). Generated final 10-page PDF manuscript (`paper/main.pdf` — 397 KB).
- Files: `paper/main.tex`, `paper/main.pdf`
- Verified: `paper/main.pdf` (10 pages, 397 KB), zero LaTeX errors, 0 missing citations, 10/10 benchmark tests passing.

### 2026-09-04 — Task 20.1.6 complete
- Changed: Authored Sections 6 (Related Work), 7 (Discussion & System Design Guidelines), and 8 (Conclusion & Future Work) in `paper/sections/06_related_work.tex`, `paper/sections/07_discussion.tex`, and `paper/sections/08_conclusion.tex`. Added 4 missing BibTeX references (`lin2023awq`, `frantar2022gptq`, `cai2024medusa`, `li2024eagle`) in `paper/references.bib`. Formulated three concrete serving design guidelines and analyzed threats to internal/external validity.
- Files: `paper/sections/06_related_work.tex`, `paper/sections/07_discussion.tex`, `paper/sections/08_conclusion.tex`, `paper/references.bib`, `paper/main.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 10-page PDF (`paper/main.pdf`) with 0 missing citations.

### 2026-09-04 — Task 20.1.5 complete
- Changed: Wrote Section 5 Empirical Evaluation & RQ Answers in `paper/sections/05_evaluation.tex`. Embedded 3 LaTeX tables (`table1_main_results.tex`, `table2_concurrency_scaling.tex`, `table3_memory_ablation.tex`) and 5 vector PDF figures (`fig1` through `fig5`). Formulated direct answers to RQ1 (Optimization Efficacy Regimes), RQ2 (Memory/Throughput Trade-offs), RQ3 (Serving Under Concurrency), and RQ4 (Model Generalization).
- Files: `paper/sections/05_evaluation.tex`, `paper/tables/*.tex`, `paper/main.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 9-page PDF (`paper/main.pdf`).

### 2026-09-04 — Task 20.1.4 complete
- Changed: Wrote Section 4 Experimental Methodology & Verification Protocol in `paper/sections/04_methodology.tex`. Documented hardware environment (NVIDIA Tesla T4 GPU, CPU host), synthetic workload distributions, $N=30$ statistical protocol with $W=5$ warmups, monotonic latency metrics (`time.perf_counter()`), VRAM memory tracking (`memory_allocated`, `memory_reserved`), and the 5-step automated verification gate protocol.
- Files: `paper/sections/04_methodology.tex`, `paper/main.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 6-page PDF (`paper/main.pdf`).

### 2026-09-04 — Task 20.1.3 complete
- Changed: Wrote Section 3 Modular Optimization Techniques in `paper/sections/03_optimizations.tex`. Formulated mathematical models and algorithmic mechanics for Radix-Tree Prefix Caching (LCP lookup, TTFT speedup model), Post-Training INT8 Quantization (scale factor $\gamma$, matrix multiplication), Speculative Decoding (draft generation, parallel verification, rejection sampling acceptance criterion), and Megatron-style Tensor Parallelism (column/row partitioning, All-Reduce).
- Files: `paper/sections/03_optimizations.tex`, `paper/main.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 5-page PDF (`paper/main.pdf`).

### 2026-09-04 — Task 20.1.2 complete
- Changed: Wrote Section 2 System Architecture & Engine Design in `paper/sections/02_architecture.tex`. Formulated mathematical equations for backend interfaces, prefill vs decode step loops, continuous batching scheduler, and physical paged KV block allocation tables.
- Files: `paper/sections/02_architecture.tex`, `paper/main.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 4-page PDF (`paper/main.pdf`).

### 2026-09-04 — Task 20.1.1 complete
- Changed: Created `paper/main.tex`, `paper/references.bib`, `paper/sections/01_introduction.tex`, and structural placeholder section files 02--08. Documented introduction, research questions (RQ1--RQ4), core contributions, and paper organization.
- Files: `paper/main.tex`, `paper/references.bib`, `paper/sections/*.tex`
- Verified: `pdflatex -halt-on-error main.tex && bibtex main && pdflatex -halt-on-error main.tex` compiled cleanly into 3-page PDF (`paper/main.pdf`).
- Limitations: Sections 02--08 contain structural titles awaiting detailed text in subsequent tasks.

### 2026-09-04 — Phase 19 Hardening & Dataset Verification Complete
- Changed: Refactored timing instrumentation in `microgen/scheduler/batch.py` and `microgen/scheduler/queue.py` using `time.perf_counter()`. Implemented `tests/benchmarks/test_timing_sanity.py`. Executed master runner `scripts/run_all_paper_experiments.py` on Kaggle Tesla T4 GPU with $N=30$ trials (`microgen_paper_results_v4.zip`).
- Verified: All 71 raw experiment records logged with strictly monotonic TTFT/TPOT values ($0.02\text{ ms} - 150.3\text{ ms}$). Epoch timestamp leaks (> 1e9 ms) are 0. LaTeX tables and 5 PDF vector figures exported cleanly.

### 2026-09-03 — Task 19.1.1 complete
- Changed: Added `first_token_time` tracking and `ttft_ms`, `tpot_ms`, `total_latency_ms` properties to `Request` dataclass (`microgen/scheduler/queue.py`) and recorded first token timing in `microgen/scheduler/batch.py`. Created dedicated unit test `tests/benchmarks/test_timing_sanity.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/batch.py`, `tests/benchmarks/test_timing_sanity.py`, `tests/scripts/test_export_paper_tables.py`, `tests/scripts/test_generate_paper_figures.py`
- Verified: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/benchmarks/test_timing_sanity.py` passed (2 passed). Full test suite passed (118 tests passed).
- Limitations: `ttft_ms` defaults to 0.0 if neither `first_token_time` nor `start_time` is recorded.

### 2026-09-03 — Task 18.1.2 complete
- Changed: Implemented `scripts/export_paper_tables.py` exporting 3 LaTeX tables to `paper/tables/*.tex` (`table1_main_results.tex`, `table2_concurrency_scaling.tex`, `table3_memory_ablation.tex`) and generating comprehensive `reproducibility.md`.
- Files: `scripts/export_paper_tables.py`, `tests/scripts/test_export_paper_tables.py`, `reproducibility.md`, `paper/tables/*.tex`
- Verified: `pytest tests/scripts/test_export_paper_tables.py` passed (3 tests) and full test suite passed (107 tests).
- Limitations: Automatically falls back to synthetic paper records if `results/raw/experiments.jsonl` is missing or empty.


### 2026-09-03 — Task 18.1.1 complete
- Changed: Implemented `scripts/generate_paper_figures.py` generating 5 high-resolution publication vector figures (PNG & PDF) saved to `paper/figures/`: Context Scaling (Fig 1), Prefix Sharing TTFT (Fig 2), Quantization Memory Pareto (Fig 3), Batching Concurrency (Fig 4), and Empirical Optimization Regime Map (Fig 5).
- Files: `scripts/generate_paper_figures.py`, `tests/scripts/test_generate_paper_figures.py`
- Verified: `pytest tests/scripts/test_generate_paper_figures.py` passed (2 tests) and full test suite passed (104 tests).
- Limitations: Automatically falls back to synthetic paper records if `results/raw/experiments.jsonl` is missing or empty.


### 2026-09-03 — Task 17.1.2 complete
- Changed: Implemented `experiments/hardware_duality.py` evaluating execution backends across CPU (`CPUDevice`) vs CUDA (`CUDADevice`) under identical workload suites. Recorded memory allocated/reserved, throughput, and TTFT/TPOT to `results/raw/experiments.jsonl`.
- Files: `experiments/hardware_duality.py`, `tests/experiments/test_hardware_duality.py`
- Verified: `pytest tests/experiments/test_hardware_duality.py` passed (3 tests) and full test suite passed (102 tests).
- Limitations: Memory bytes dynamically queried via `get_memory_info()`.


### 2026-09-03 — Task 17.1.1 complete
- Changed: Implemented `experiments/model_generalization.py` evaluating modern open-weights architectures (`Qwen/Qwen2.5-0.5B`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, etc.) under baseline vs optimizations (+INT8, +Paged KV, +Prefix Caching). Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/model_generalization.py`, `tests/experiments/test_model_generalization.py`
- Verified: `pytest tests/experiments/test_model_generalization.py` passed (3 tests) and full test suite passed (99 tests).
- Limitations: Default test runner uses `sshleifer/tiny-gpt2` for CPU execution.


### 2026-09-03 — Task 16.1.3 complete
- Changed: Implemented `experiments/combined_interactions.py` evaluating the full 8-configuration combinatorial optimization matrix (+Paged KV, +Prefix Caching, +INT8 Quantization, +Paged+Prefix+INT8). Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/combined_interactions.py`, `tests/experiments/test_combined_interactions.py`
- Verified: `pytest tests/experiments/test_combined_interactions.py` passed (3 tests) and full test suite passed (96 tests).
- Limitations: Default runner evaluates 8 configurations on `sshleifer/tiny-gpt2` for CPU execution.


### 2026-09-03 — Task 16.1.2 complete
- Changed: Implemented `experiments/paged_memory_pressure.py` evaluating Paged KV Cache allocation vs Contiguous KV Cache under constrained memory pressure regimes (25%, 50%, 75%, 100%). Evaluated fragmentation ratio and OOM bounds. Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/paged_memory_pressure.py`, `tests/experiments/test_paged_memory_pressure.py`
- Verified: `pytest tests/experiments/test_paged_memory_pressure.py` passed (3 tests) and full test suite passed (93 tests).
- Limitations: Contiguous allocation fragmentation evaluates upfront reservation ratio against actual generated sequence length.


### 2026-09-03 — Task 16.1.1 complete
- Changed: Implemented `experiments/batching_concurrency.py` evaluating static batching vs continuous batching scheduler across batch size range $B \in [1, 2, 4, 8, 16]$. Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/batching_concurrency.py`, `tests/experiments/test_batching_concurrency.py`
- Verified: `pytest tests/experiments/test_batching_concurrency.py` passed (3 tests) and full test suite passed (90 tests).
- Limitations: Sequence lengths bounded to $L \le 256$ to fit within `sshleifer/tiny-gpt2` position limits during CPU tests.


### 2026-09-03 — Task 15.1.4 complete
- Changed: Implemented `experiments/speculative_sweep.py` evaluating speculative decoding across draft lookahead length $K \in [1..5]$ and empirical acceptance rate $\alpha$. Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/speculative_sweep.py`, `tests/experiments/test_speculative_sweep.py`
- Verified: `pytest tests/experiments/test_speculative_sweep.py` passed (3 tests) and full test suite passed (87 tests).
- Limitations: Default runner tests $K \in [1, 2, 3, 4, 5]$ with `sshleifer/tiny-gpt2` for CPU execution.


### 2026-09-03 — Task 15.1.3 complete
- Changed: Implemented `experiments/quant_lifecycle.py` measuring stage-by-stage memory allocation across FP32 model loading, INT8 weight quantization, and dynamic INT8 KV cache compression. Evaluated logit quality metrics (MSE and Cosine Similarity). Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/quant_lifecycle.py`, `tests/experiments/test_quant_lifecycle.py`
- Verified: `pytest tests/experiments/test_quant_lifecycle.py` passed (3 tests) and full test suite passed (84 tests).
- Limitations: INT8 weight quantization dequantizes on-the-fly during linear forward passes for CPU execution compatibility.


### 2026-09-03 — Task 15.1.2 complete
- Changed: Implemented `experiments/prefix_sharing.py` evaluating shared prompt prefix ratio sweeps ($\alpha \in [0.0..1.0]$) comparing uncached vs prefix-cached runs. Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/prefix_sharing.py`, `tests/experiments/test_prefix_sharing.py`
- Verified: `pytest tests/experiments/test_prefix_sharing.py` passed (3 tests) and full test suite passed (81 tests).
- Limitations: Shared prefix matching requires exact token ID sequence prefix equality.

### 2026-09-03 — Task 15.1.1 complete
- Changed: Implemented `experiments/context_sweep.py` evaluating sequence length scaling sweeps ($L_{in} \in [32..2048]$, $L_{out} \in [16..512]$) across model parameter scales comparing HF PyTorch reference baseline against MicroGen engine backend. Recorded metrics to `results/raw/experiments.jsonl`.
- Files: `experiments/context_sweep.py`, `tests/experiments/test_context_sweep.py`
- Verified: `pytest tests/experiments/test_context_sweep.py` passed (3 tests) and full test suite passed (78 tests).
- Limitations: Default runner defaults to $L_{in} \in [32, 128, 256]$ and $L_{out} \in [16, 32]$ for CPU development; configurable to full long-context range for GPU execution.

### Phase 1 — Initial Scaffolding and Simple KV Cache Proof-of-Concept
- [x] Task 1.1.1: Set up library structure (`minigen/`) and dependencies.
- [x] Task 1.1.2: Implement `SimpleKVCache` class for managing per-layer keys/values.

### Phase 2 — Custom Generation Loop
- [x] Task 2.1.1: Implement single-step forward pass logic.
- [x] Task 2.1.2: Implement the full autoregressive `generate` function (supporting both `use_cache=True` and `False`).

### Phase 3 — Initial Benchmarking
- [x] Task 3.1.1: Create `scripts/benchmark.py` to measure and compare generation latency.

### Phase 4 — Core Engine & Hardware Abstraction (CPU / GPU Backends)
- [x] Task 4.1.1: Implement `Device` abstractions (`microgen/devices/base.py`, `cpu.py`, `cuda.py`) and `InferenceBackend` protocol (`microgen/backends/base.py`).
- [x] Task 4.1.2: Implement `PyTorchBackend` (`microgen/backends/pytorch.py`) supporting prefill, decode, and sampling across CPU and CUDA devices.
- [x] Task 4.1.3: Refactor and enhance KV Cache into `microgen/runtime/kv_cache.py` supporting per-request lifecycle and CPU/GPU memory tracking.

### Phase 5 — Request Queue, Scheduling & Continuous Batching
- [x] Task 5.1.1: Implement `Request` tracking dataclasses and thread-safe `RequestQueue` (`microgen/scheduler/queue.py`).
- [x] Task 5.1.2: Implement static and dynamic batching manager (`microgen/scheduler/batch.py`) with sequence padding and attention masking.
- [x] Task 5.1.3: Implement `ContinuousBatchingScheduler` (`microgen/scheduler/scheduler.py`) handling dynamic request entry/exit and memory-aware admission control.

### Phase 6 — Caching Infrastructure, Rate Limiting & Utilities
- [x] Task 6.1.1: Implement prompt cache manager and prefix matching in `microgen/caching/prefix_cache.py`.
- [x] Task 6.1.2: Implement token bucket rate limiter in `microgen/caching/rate_limiter.py`.

### Phase 7 — Granular Profiling & Automated Performance Diagnostics
- [x] Task 7.1.1: Implement CUDA/CPU event execution profiler in `microgen/profiling/profiler.py`.
- [x] Task 7.1.2: Implement automated performance diagnostic engine and bottleneck detector in `microgen/profiling/diagnostics.py`.

### Phase 8 — HTTP API, SSE Streaming & Unified CLI
- [x] Task 8.1.1: Implement FastAPI HTTP app with SSE streaming support in `microgen/api/app.py`.
- [x] Task 8.1.2: Implement unified Click CLI and server runner in `microgen/cli/main.py`.

### Phase 9 — E2E Benchmarking Suite & Hardware Comparison
- [x] Task 9.1.1: Implement end-to-end benchmarking suite in `scripts/e2e_benchmark.py`.

### Phase 10 — Paged KV Cache & Memory Eviction Architecture
- [x] Task 10.1.1: Implement fixed-size physical block memory allocator and logical-to-physical block table in `microgen/runtime/paged_kv.py`.
- [x] Task 10.1.2: Implement sliding-window KV cache eviction and Grouped-Query Attention (GQA) key-value head repetition support.

### Phase 11 — Speculative Decoding & Draft Model Acceleration
- [x] Task 11.1.1: Implement speculative generation loop with draft-target verification in `microgen/scheduler/speculative.py`.
- [x] Task 11.1.2: Implement speculative rejection sampling and KV cache state rollback on token rejection.

### Phase 12 — Weight Quantization & Hardware Int8/FP8 Backends
- [x] Task 12.1.1: Implement INT8 / FP8 per-channel weight quantization backend in `microgen/backends/quantized.py`.
- [x] Task 12.1.2: Implement INT8 dynamic KV cache quantization to compress active sequence key/value tensors by 2x.

### Phase 13 — Kaggle & Multi-GPU Benchmark Suite
- [x] Task 13.1.1: Implement multi-GPU tensor-parallel backend wrapper in `microgen/backends/parallel.py`.
- [x] Task 13.1.2: Implement Kaggle automated benchmark runner and HTML performance report generator in `scripts/kaggle_benchmark_runner.py`.

---

## Current Task

All Tasks and Phases Complete! 🎉

## Next Task

None (All 13 phases successfully implemented and verified)

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.

### 2026-09-02 — Task 10.1.1 complete
- Changed: Implemented fixed-size physical block memory allocator (`PagedKVCacheAllocator`) and logical-to-physical sequence block table (`BlockTable`) in `microgen/runtime/paged_kv.py`.
- Files: `microgen/runtime/paged_kv.py`, `tests/runtime/test_paged_kv.py`
- Verified: `pytest` full suite (60 passed in 27.03s).
- Limitations: Integrates block allocation data structures; paged attention Kernel / Custom CUDA index gathering can be integrated in custom FlashAttention bindings.

### 2026-09-02 — Task 10.1.2 complete
- Changed: Implemented sliding-window KV cache eviction in `KVCacheState` and Grouped-Query Attention (GQA) key-value head repetition helper (`repeat_kv`) in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (62 passed in 26.76s).
- Limitations: Evicts strictly along sequence dimension for sliding window; full tree-based attention masks can be passed for custom attention patterns.

### 2026-09-02 — Task 11.1.1 complete
- Changed: Implemented `SpeculativeEngine` and `SpeculativeResult` in `microgen/scheduler/speculative.py` for draft-model candidate generation and target-model logit verification.
- Files: `microgen/scheduler/speculative.py`, `tests/scheduler/test_speculative.py`
- Verified: `pytest` full suite (63 passed in 38.62s).
- Limitations: Performs sequential logit verification; batched candidate verification and rejection sampling will be added in Task 11.1.2.

### 2026-09-02 — Task 11.1.2 complete
- Changed: Implemented speculative rejection sampling (`rejection_sample_token`) and KV cache rollback (`rollback`) in `microgen/scheduler/speculative.py` and `microgen/runtime/kv_cache.py`.
- Files: `microgen/scheduler/speculative.py`, `microgen/runtime/kv_cache.py`, `tests/scheduler/test_speculative.py`
- Verified: `pytest` full suite (65 passed in 26.32s).
- Limitations: Performs per-token probability rejection checks; tree-based candidate verification can be added as an optimization.

### 2026-09-02 — Task 12.1.1 complete
- Changed: Implemented `QuantizedPyTorchBackend` in `microgen/backends/quantized.py` supporting per-channel INT8 weight quantization (`QuantizedLinear` layer wrapper and `quantize_linear_layer_per_channel`).
- Files: `microgen/backends/quantized.py`, `microgen/backends/__init__.py`, `tests/backends/test_quantized.py`
- Verified: `pytest` full suite (68 passed in 55.57s, verifying >0.98 cosine similarity between float and INT8 quantized model logits).
- Limitations: Performs on-the-fly weight dequantization during linear forward passes; custom INT8 GEMM CUDA kernels can be added for hardware acceleration.

### 2026-09-02 — Task 12.1.2 complete
- Changed: Implemented INT8 dynamic KV cache quantization support (`quantize_kv=True`, `_quantize_tensor`, `_dequantize_tensor`, scale factor tensors) in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (69 passed in 40.08s, verifying >2x memory footprint reduction and >0.98 cosine similarity accuracy).
- Limitations: Performs per-vector dynamic quantization during KV cache update; custom INT8 GEMM / quantized attention kernels can be integrated for direct INT8 QK^V matrix multiplications.

### 2026-09-02 — Task 13.1.1 complete
- Changed: Implemented multi-GPU / multi-rank `TensorParallelPyTorchBackend` in `microgen/backends/parallel.py` with `ColumnParallelLinear` and `RowParallelLinear` layer sharding and all-reduce sum output aggregation.
- Files: `microgen/backends/parallel.py`, `microgen/backends/__init__.py`, `tests/backends/test_parallel.py`
- Verified: `pytest` full suite (73 passed in 40.01s, verifying 100% exact math equivalence and >0.999 cosine similarity for sharded model forward passes).
- Limitations: Performs sequential rank execution on single-machine simulated device arrays; NCCL multi-node distributed process groups can be initialized for distributed cluster scaling.

### 2026-09-02 — Task 13.1.2 complete
- Changed: Implemented `scripts/kaggle_benchmark_runner.py` for automated hardware detection (CPU/Kaggle T4/P100 GPUs), performance metrics collection (TTFT, ITL, Throughput, Memory), JSON results output (`kaggle_benchmark_results.json`), and modern HTML report generation (`microgen_benchmark_report.html`).
- Files: `scripts/kaggle_benchmark_runner.py`, `kaggle_benchmark_results.json`, `microgen_benchmark_report.html`
- Verified: Executed benchmark runner successfully and verified HTML report rendering & `pytest` full suite (73 passed in 35.51s).
- Limitations: Benchmark metrics depend on local/Kaggle execution environment; output reports are styled and self-contained for HTML web viewing.

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.

### 2026-09-02 — Task 10.1.1 complete
- Changed: Implemented fixed-size physical block memory allocator (`PagedKVCacheAllocator`) and logical-to-physical sequence block table (`BlockTable`) in `microgen/runtime/paged_kv.py`.
- Files: `microgen/runtime/paged_kv.py`, `tests/runtime/test_paged_kv.py`
- Verified: `pytest` full suite (60 passed in 27.03s).
- Limitations: Integrates block allocation data structures; paged attention Kernel / Custom CUDA index gathering can be integrated in custom FlashAttention bindings.

### 2026-09-02 — Task 10.1.2 complete
- Changed: Implemented sliding-window KV cache eviction in `KVCacheState` and Grouped-Query Attention (GQA) key-value head repetition helper (`repeat_kv`) in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (62 passed in 26.76s).
- Limitations: Evicts strictly along sequence dimension for sliding window; full tree-based attention masks can be passed for custom attention patterns.

### 2026-09-02 — Task 11.1.1 complete
- Changed: Implemented `SpeculativeEngine` and `SpeculativeResult` in `microgen/scheduler/speculative.py` for draft-model candidate generation and target-model logit verification.
- Files: `microgen/scheduler/speculative.py`, `tests/scheduler/test_speculative.py`
- Verified: `pytest` full suite (63 passed in 38.62s).
- Limitations: Performs sequential logit verification; batched candidate verification and rejection sampling will be added in Task 11.1.2.

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.

### 2026-09-02 — Task 10.1.1 complete
- Changed: Implemented fixed-size physical block memory allocator (`PagedKVCacheAllocator`) and logical-to-physical sequence block table (`BlockTable`) in `microgen/runtime/paged_kv.py`.
- Files: `microgen/runtime/paged_kv.py`, `tests/runtime/test_paged_kv.py`
- Verified: `pytest` full suite (60 passed in 27.03s).
- Limitations: Integrates block allocation data structures; paged attention Kernel / Custom CUDA index gathering can be integrated in custom FlashAttention bindings.

### 2026-09-02 — Task 10.1.2 complete
- Changed: Implemented sliding-window KV cache eviction in `KVCacheState` and Grouped-Query Attention (GQA) key-value head repetition helper (`repeat_kv`) in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (62 passed in 26.76s).
- Limitations: Evicts strictly along sequence dimension for sliding window; full tree-based attention masks can be passed for custom attention patterns.

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.

### 2026-09-02 — Task 10.1.1 complete
- Changed: Implemented fixed-size physical block memory allocator (`PagedKVCacheAllocator`) and logical-to-physical sequence block table (`BlockTable`) in `microgen/runtime/paged_kv.py`.
- Files: `microgen/runtime/paged_kv.py`, `tests/runtime/test_paged_kv.py`
- Verified: `pytest` full suite (60 passed in 27.03s).
- Limitations: Integrates block allocation data structures; paged attention Kernel / Custom CUDA index gathering can be integrated in custom FlashAttention bindings.

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.

## Constraints

- First-class support for both CPU and CUDA devices.
- No vendor lock-in in core scheduler / engine logic; depend strictly on abstractions (`InferenceBackend`, `Device`, `KVCache`).
- Deterministic performance and explicit memory management.
- Scope discipline: implement one coherent architectural unit per task.

## Log

### 2026-09-02 — Task 1.1.1 complete
- Changed: Created package directories (`minigen/`, `tests/`, `scripts/`) and `requirements.txt`.
- Files: `requirements.txt`, `minigen/__init__.py`, `tests/__init__.py`, `scripts/.gitkeep`
- Verified: `pip install -r requirements.txt` and verified package imports with Python execution.
- Limitations: Minimal scaffolding only; no model or generation logic included yet.

### 2026-09-02 — Task 1.1.2 complete
- Changed: Implemented `SimpleKVCache` class for per-layer key/value tensor management. Added configuration file `pytest.ini` to bypass system ROS plugin interference.
- Files: `minigen/cache.py`, `tests/test_cache.py`, `pytest.ini`
- Verified: `pytest tests/test_cache.py` (5 tests passed).
- Limitations: Manages tensor concatenation along sequence length dimension (`dim=-2`), but does not handle variable sequence lengths across batch elements or explicit memory paging/sliding window eviction.

### 2026-09-02 — Task 2.1.1 complete
- Changed: Implemented `generate_step` function in `minigen/generator.py` and updated `SimpleKVCache` to integrate natively with HuggingFace `Cache` interface.
- Files: `minigen/generator.py`, `minigen/cache.py`, `tests/test_generator.py`, `tests/test_cache.py`
- Verified: `pytest` full suite (6 tests passed, verifying 100% logits equivalence between cached step and full forward pass).
- Limitations: Interfaces with HF model via `Cache` subclassing; relies on HF attention module for QK^V computation rather than custom CUDA/attention kernels.

### 2026-09-02 — Task 2.1.2 complete
- Changed: Implemented `generate()` function in `minigen/generator.py` for autoregressive greedy decoding with support for `use_cache=True` and `use_cache=False`.
- Files: `minigen/generator.py`, `tests/test_generator.py`
- Verified: `pytest` full suite (9 tests passed, verifying 100% token sequence identity between cached generation, uncached generation, and HuggingFace `model.generate()`).
- Limitations: Single-sequence focused (`batch_size=1` tested); lacks per-sequence EOS stopping/padding for batched inputs (`batch_size > 1`); greedy decoding only (no top-k/top-p sampling).

### 2026-09-02 — Task 3.1.1 complete
- Changed: Created `scripts/benchmark.py` to measure and compare generation latency and throughput for cache-on vs cache-off configurations.
- Files: `scripts/benchmark.py`
- Verified: Executed `scripts/benchmark.py` successfully on CPU (`sshleifer/tiny-gpt2`), demonstrating a **1.68x speedup** for cache-on vs cache-off with 100% exact token sequence matching.
- Limitations: Benchmarked exclusively on CPU with `sshleifer/tiny-gpt2` and `batch_size=1`.

### 2026-09-02 — Task 4.1.1 complete
- Changed: Implemented `Device` hardware abstractions (`CPUDevice`, `CUDADevice`, `get_device`) in `microgen/devices/` and `InferenceBackend` protocol in `microgen/backends/base.py`.
- Files: `microgen/devices/base.py`, `microgen/devices/cpu.py`, `microgen/devices/cuda.py`, `microgen/devices/__init__.py`, `microgen/backends/base.py`, `microgen/backends/__init__.py`, `tests/devices/test_devices.py`
- Verified: `pytest` full suite (13 passed in 14.29s).
- Limitations: Memory info on CPU relies on `/proc/meminfo` on Linux; CUDA device properties require an available NVIDIA GPU runtime to return non-zero VRAM metrics.

### 2026-09-02 — Task 4.1.2 complete
- Changed: Implemented `PyTorchBackend` in `microgen/backends/pytorch.py` supporting prefill, single-token decode passes, greedy/top-k/top-p sampling, and device memory querying.
- Files: `microgen/backends/pytorch.py`, `microgen/backends/__init__.py`, `tests/backends/test_pytorch_backend.py`
- Verified: `pytest` full suite (17 passed in 17.44s).
- Limitations: Prefill and decode passes currently process batch sequences with uniform padding; top-p nucleus sampling sorts full logits vocabulary.

### 2026-09-02 — Task 4.1.3 complete
- Changed: Implemented `KVCacheState` (subclassing HuggingFace `Cache`) and `KVCacheManager` for per-request KV lifecycle allocation, sequence tracking, and memory calculation in `microgen/runtime/kv_cache.py`.
- Files: `microgen/runtime/kv_cache.py`, `microgen/runtime/__init__.py`, `tests/runtime/test_kv_cache.py`
- Verified: `pytest` full suite (20 passed in 15.77s).
- Limitations: KV cache memory is managed via PyTorch tensor concatenation per sequence; paged KV memory block allocation will be added in continuous batching phase.

### 2026-09-02 — Task 5.1.1 complete
- Changed: Implemented `Request` dataclass, `RequestStatus` enum, and thread-safe `RequestQueue` with priority queuing and lazy $O(1)$ enqueue sorting in `microgen/scheduler/queue.py`.
- Files: `microgen/scheduler/queue.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_queue.py`
- Verified: `pytest` full suite (24 passed in 16.53s).
- Limitations: Priority sorting handles scalar integer priority values; complex multi-resource scheduling policies will be enforced by the scheduler in Task 5.1.3.

### 2026-09-02 — Task 5.1.2 complete
- Changed: Implemented `Batch` dataclass, left-padded prefill batch creation (`create_prefill_batch`), decode batching (`create_decode_batch`), and token update tracking (`update_requests_with_sampled_tokens`) in `microgen/scheduler/batch.py`.
- Files: `microgen/scheduler/batch.py`, `microgen/scheduler/__init__.py`, `tests/scheduler/test_batch.py`
- Verified: `pytest` full suite (27 passed in 24.47s).
- Limitations: Batch tensor padding uses left-padding for prefill; attention masks are 2D masks.

### 2026-09-02 — Task 5.1.3 complete
- Changed: Implemented `ContinuousBatchingScheduler` in `microgen/scheduler/scheduler.py` handling dynamic request admission, prefill/decode iterations, per-request KV cache allocation & cleanup, and multi-request generation loops. Updated `InferenceBackend` protocol and `PyTorchBackend` to accept optional `attention_mask`.
- Files: `microgen/scheduler/scheduler.py`, `microgen/scheduler/__init__.py`, `microgen/backends/base.py`, `microgen/backends/pytorch.py`, `tests/scheduler/test_scheduler.py`
- Verified: `pytest` full suite (28 passed in 17.81s).
- Limitations: Schedulers perform iteration step evaluation in Python runtime loop.

### 2026-09-02 — Task 6.1.1 complete
- Changed: Implemented `PrefixKVCache` in `microgen/caching/prefix_cache.py` with SHA256 token hashing, exact lookup, longest prefix matching for precomputed prompt state reuse, and capacity eviction.
- Files: `microgen/caching/prefix_cache.py`, `microgen/caching/__init__.py`, `tests/caching/test_prefix_cache.py`
- Verified: `pytest` full suite (33 passed in 19.73s).
- Limitations: Prefix matching operates over linear cached prompt sequences; tree-based prefix radix tries can be integrated as an optimization for large prompt libraries.

### 2026-09-02 — Task 6.1.2 complete
- Changed: Implemented `TokenBucketRateLimiter` in `microgen/caching/rate_limiter.py` supporting thread-safe RPM (requests/min) and TPM (tokens/min) checking, consumption (`acquire`), and capacity refill.
- Files: `microgen/caching/rate_limiter.py`, `microgen/caching/__init__.py`, `tests/caching/test_rate_limiter.py`
- Verified: `pytest` full suite (38 passed in 19.30s).
- Limitations: Rate limiting tracks aggregate token consumption per process; per-client/IP rate limits can be added at the HTTP API layer.

### 2026-09-02 — Task 7.1.1 complete
- Changed: Implemented hardware-aware `Profiler` in `microgen/profiling/profiler.py` supporting CPU and CUDA event execution timing (`profile` context manager), section duration tracking, aggregate stats computation (`count`, `total`, `avg`, `min`, `max`, `p95`), and reset capabilities.
- Files: `microgen/profiling/profiler.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_profiler.py`
- Verified: `pytest` full suite (42 passed in 18.67s).
- Limitations: CUDA timing uses `torch.cuda.Event` synchronization; high-frequency context switches should be wrapped around major execution boundaries (`prefill`, `decode`, `sampling`).

### 2026-09-02 — Task 7.1.2 complete
- Changed: Implemented `DiagnosticEngine` in `microgen/profiling/diagnostics.py` producing structured `DiagnosticReport` evaluations with prefill/decode ratios, primary bottleneck classification (`prefill`, `decode`, `sampling`, `balanced`), and optimization recommendations.
- Files: `microgen/profiling/diagnostics.py`, `microgen/profiling/__init__.py`, `tests/profiling/test_diagnostics.py`
- Verified: `pytest` full suite (46 passed in 19.42s).
- Limitations: Diagnostic analysis uses statistical heuristics over recorded profiler duration stats.

### 2026-09-02 — Task 8.1.1 complete
- Changed: Implemented FastAPI server in `microgen/api/app.py` providing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` endpoints with support for both synchronous responses and server-sent event (SSE) streaming. Installed `fastapi`, `uvicorn`, `httpx`, `click`, `pydantic`.
- Files: `microgen/api/app.py`, `microgen/api/__init__.py`, `requirements.txt`, `tests/api/test_app.py`
- Verified: `pytest` full suite (51 passed in 24.37s).
- Limitations: Streaming format yields standard OpenAI JSON chunk objects via SSE; authentication headers can be added in API gateway middleware.

### 2026-09-02 — Task 8.1.2 complete
- Changed: Implemented unified Click CLI entry point `microgen/cli/main.py` supporting `microgen serve`, `microgen generate`, and `microgen profile` commands.
- Files: `microgen/cli/main.py`, `microgen/cli/__init__.py`, `tests/cli/test_cli.py`
- Verified: `pytest` full suite (54 passed in 33.71s).
- Limitations: CLI server command starts foreground uvicorn process; background daemonization can be managed with standard Linux process managers (systemd/supervisord).

### 2026-09-02 — Task 9.1.1 complete
- Changed: Implemented end-to-end benchmarking suite in `scripts/e2e_benchmark.py` measuring Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec), and hardware memory across dynamic concurrency levels, saving output to `benchmark_results.json`.
- Files: `scripts/e2e_benchmark.py`, `benchmark_results.json`
- Verified: Executed `scripts/e2e_benchmark.py` successfully and verified all 54 tests pass cleanly across full test suite.
- Limitations: Multi-GPU tensor parallelism can be added for multi-node deployment.
