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
- `benchmarks/` — Automated performance and scalability benchmark suite.
- `tests/` — Modular unit and integration test suite.

## Interfaces / Contracts

### `InferenceBackend` Protocol (`microgen/backends/base.py`)
Abstract interface defining execution methods:
- `load_model(model_name: str, device: Device) -> None`
- `prefill(prompt_ids: torch.Tensor, cache: Optional[KVCache]) -> Tuple[torch.Tensor, KVCache]`
- `decode(token_ids: torch.Tensor, cache: KVCache) -> Tuple[torch.Tensor, KVCache]`
- `get_memory_usage() -> Dict[str, float]`

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

## Decisions Log

### 2026-09-02 — Explicit cache object rather than implicit tuple passing
Abstracting KV cache into a dedicated class (`SimpleKVCache` / `KVCache`) provides a clear domain boundary and satisfies the requirement of making the cache explicit and hand-built.

### 2026-09-02 — Backend & Device Abstraction for CPU/GPU Duality
Hardware execution is decoupled behind `InferenceBackend` and `Device` abstractions. The engine, scheduler, and API communicate exclusively through these abstractions, ensuring first-class support for both CPU and CUDA devices without code duplication.

### 2026-09-02 — Continuous Batching & Memory-Aware Scheduling
Scheduler manages request admission based on KV cache memory requirements and active sequence lengths, enabling dynamic entry/exit of requests without blocking ongoing batch iterations.

### 2026-09-02 — Paged KV Block Allocation & Speculative Engine Plan
Future extension phases introduce block-based physical paged KV allocation (`paged_kv.py`), draft-target speculative execution loops (`speculative.py`), low-precision INT8/FP8 backends (`quantized.py`), and distributed multi-GPU tensor parallelism (`parallel.py`).
