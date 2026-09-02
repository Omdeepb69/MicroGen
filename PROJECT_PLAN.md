# Project Plan

This file is the persistent source of truth for project progress. It
survives across conversations and context resets. Conversation history is
not a record of project state — this file is. Update it after every
completed task, not at the end of a session.

## Goal

MicroGen is a modular, hardware-aware LLM inference engine built from scratch to understand and optimize modern model serving, featuring first-class CPU and GPU execution, scheduling, batching, continuous batching, KV caching, streaming, profiling, and performance diagnostics.

## Current Phase

Phase 12 — Weight Quantization & Hardware Int8/FP8 Backends

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

---

## Completed Phases

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
