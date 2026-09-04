# Architecture

Stable architectural decisions for this project. Unlike `PROJECT_PLAN.md`
(which tracks progress and changes constantly), this file changes rarely —
only when an architectural decision is actually made or revised. Treat
entries here as load-bearing: other tasks and modules assume they hold.

## Module Map

- `microgen/api/` — HTTP API, FastAPI routes, Server-Sent Events (SSE) streaming.
- `microgen/runtime/` — Core execution engine, model wrapper, sampling, and KV cache manager.
- `microgen/scheduler/` — Request queue, static/dynamic batching, continuous batching scheduler.
- `microgen/backends/` — Hardware & runtime execution backends (`base.py`, `pytorch.py`, `onnx.py`).
- `microgen/devices/` — Device abstractions for CPU and CUDA execution (`base.py`, `cpu.py`, `cuda.py`).
- `microgen/cache/` — Caching systems (`lru.py`, `prefix.py`, `ttl.py`).
- `microgen/infra/` — Infrastructure helpers (`rate_limiter.py`, `metrics.py`).
- `microgen/profiler/` — Granular execution profiler, events collector, and `/diagnose` bottleneck analyzer.
- `microgen/cli.py` — Unified CLI entry points (`serve`, `benchmark`, `diagnose`, `model-info`).
- `benchmarks/` — Workload dataset generator (`workloads.py`) and empirical experiment harness (`harness.py`).
- `experiments/` — Research paper ablation experiments (`context_sweep.py`, `prefix_sharing.py`, `quant_lifecycle.py`, `speculative_sweep.py`, `batching_concurrency.py`, `paged_memory_pressure.py`, `combined_interactions.py`, `gonyai_eval.py`).
- `tests/` — Modular unit and integration test suite.

## Interfaces / Contracts

### `InferenceBackend` Protocol (`microgen/backends/base.py`)
Abstract interface defining execution methods:
- `load_model(model_name: str, device: Device) -> None`
- `prefill(prompt_ids: torch.Tensor, cache: Optional[KVCache]) -> Tuple[torch.Tensor, KVCache]`
- `decode(token_ids: torch.Tensor, cache: KVCache) -> Tuple[torch.Tensor, KVCache]`
- `get_memory_usage() -> Dict[str, float]`

### `WorkloadProfile` Protocol (`benchmarks/workloads.py`)
Abstract interface for reproducible workload suites:
- `generate_requests(num_requests: int, seed: int) -> List[Request]`
- `sequence_length_bounds: Tuple[int, int]`

### `ExperimentHarness` Protocol (`benchmarks/harness.py`)
Abstract interface for isolated statistical experiment collection:
- `run_trials(backend_factory, workload, n_trials: int = 30) -> ExperimentResult`
- Computes `median`, `p50`, `p90`, `p95`, `p99` latencies, `allocated_vram_mb`, `reserved_vram_mb`, and output logit similarity.

### `Device` Protocol (`microgen/devices/base.py`)
Abstract device interface:
- `name: str` ('cpu' | 'cuda')
- `allocate_tensor(shape, dtype) -> torch.Tensor`
- `synchronize() -> None`
- `available_memory_bytes() -> int`

### `KVCache` Protocol (`microgen/runtime/kv_cache.py`)
Per-request & per-layer KV cache interface supporting CPU RAM and GPU VRAM memory management:
- `update(key_states, value_states, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]`
- `get_seq_length(layer_idx) -> int`
- `free() -> None`

### `Scheduler` (`microgen/scheduler/scheduler.py`)
Continuous batching scheduler:
- `schedule() -> Optional[Batch]`
- `step(batch: Batch) -> List[RequestOutput]`
- Performs memory-aware admission control and continuous request entry/exit.

## Boundaries

- API layer (`microgen/api/`) depends on Scheduler and Engine.
- Scheduler (`microgen/scheduler/`) manages Request Queue and Batching, delegating execution to `InferenceBackend`.
- `InferenceBackend` depends on `Device` and `KVCache`.
- Profiler collects events across API, Scheduler, and Backend layers without mutating execution logic.
- Experiments (`experiments/`) depend strictly on `benchmarks/harness.py`, `benchmarks/workloads.py`, and public engine abstractions (`InferenceBackend`, `Device`, `Scheduler`).

## Decisions Log

### 2026-09-02 — Explicit cache object rather than implicit tuple passing
Abstracting KV cache into a dedicated class (`SimpleKVCache` / `KVCache`) provides a clear domain boundary and satisfies the requirement of making the cache explicit and hand-built.

### 2026-09-02 — Backend & Device Abstraction for CPU/GPU Duality
Hardware execution is decoupled behind `InferenceBackend` and `Device` abstractions. The engine, scheduler, and API communicate exclusively through these abstractions, ensuring first-class support for both CPU and CUDA devices without code duplication.

### 2026-09-02 — Continuous Batching & Memory-Aware Scheduling
Scheduler manages request admission based on KV cache memory requirements and active sequence lengths, enabling dynamic entry/exit of requests without blocking ongoing batch iterations.

### 2026-09-02 — Paged KV Block Allocation & Speculative Engine Plan
Future extension phases introduce block-based physical paged KV allocation (`paged_kv.py`), draft-target speculative execution loops (`speculative.py`), low-precision INT8/FP8 backends (`quantized.py`), and distributed multi-GPU tensor parallelism (`parallel.py`).

### 2026-09-03 — Empirical Research Paper Benchmarking Methodology
Decoupled workload generation (`benchmarks/workloads.py`) and empirical experiment harness (`benchmarks/harness.py`) from engine internals. Enforces 30-trial median/p95/p99 statistical sampling, CUDA host-device synchronization, strict memory cleanup (`gc.collect` + `torch.cuda.empty_cache`), and explicit differentiation between `allocated_vram` and `reserved_vram`.

### 2026-09-03 — 5-Step Verification Gate Protocol for Benchmark Hardening
Enforces a 5-step verification gate sequence before full benchmark execution: (1) single-request monotonic timing test verifying $0 < \text{TTFT} < 10^5\text{ ms}$, (2) single-configuration $N=30, W=5$ trial count verification, (3) batch concurrency scaling up to $B=64$ without NaNs/epoch timestamps, (4) multi-model loader verification across `gpt2`, `Qwen/Qwen2.5-0.5B`, and `TinyLlama/TinyLlama-1.1B`, and (5) 100% data-driven table/figure export validation.

### 2026-09-04 — Modular LaTeX Paper Architecture & Automated PDF Compilation
Academic paper manuscript is structured modularly across section files (`paper/sections/01_introduction.tex` through `08_conclusion.tex`) orchestrated by `paper/main.tex`. All experimental tables (`paper/tables/`) and figures (`paper/figures/`) are dynamically included, ensuring end-to-end data lineage from `results/raw/experiments.jsonl` to compiled 10-page PDF (`paper/main.pdf`).

### 2026-09-04 — Research Framing & Empirical Precision Hardening (Phase 21)
MicroGen is explicitly positioned as an *empirical inference-systems research framework with a functional modular serving engine* rather than a production engine competing with vLLM/TensorRT-LLM/SGLang. Paper manuscript claims are hardened to eliminate reviewer friction by replacing absolute claims with scientifically scoped observations, highlighting non-monotonic optimization composition ("Optimization Composition Is Not Monotonic"), and embedding a comparative systems feature matrix table.

### 2026-09-04 — Empirical Rigor, Statistical Audit & Multi-Architecture Scaling (Phase 22)
All empirical benchmark experiment modules (`benchmarks/harness.py`, `experiments/`) compute and export standard deviations ($\sigma$), interquartile ranges (IQR), paired statistical significance p-values, and draft token acceptance rates ($\alpha$). The table generation pipeline (`scripts/export_paper_tables.py`) exports explicit baseline normalization references, variance intervals, and a dedicated optimization composition ladder table (`table4_non_monotonic_ladder.tex`). Model generalization evaluates across both GPT-2 and Llama/Qwen-style architecture families.

### 2026-09-04 — Multi-Architecture & Multi-GPU Empirical Expansion (Phase 23)
Decoupled architecture-specific execution logic (RoPE rotary embeddings, GQA head repetition, RMSNorm) behind standard `InferenceBackend` protocol adaptors to seamlessly evaluate modern open-weights model families (`Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B`) alongside GPT-2. Hardware duality benchmarks explicitly record compute capabilities and architecture generation metadata (NVIDIA P100 Pascal vs Tesla T4 Turing). Tensor parallelism evaluation leverages `ParallelBackend` sharding across dual Tesla T4 GPUs.

### 2026-09-04 — PyPI Package & High-Level Developer Experience (Phase 24)
Standardized package distribution via `pyproject.toml` exposing high-level `microgen.LLMEngine` wrapper API. `LLMEngine` encapsulates `InferenceBackend`, `PagedKVCacheManager`, `PrefixCacheManager`, and `Scheduler` into a simple 1-line Python interface (`engine = microgen.LLMEngine.from_pretrained(...)`), while preserving low-level protocol access for advanced systems researchers.

