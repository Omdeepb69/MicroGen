Absolutely. Here is the **full project specification for MicroGen**, structured as an actual serious engineering project rather than just a collection of inference features.

# MicroGen

### A tiny, hardware-aware LLM inference engine for CPU and GPU

**MicroGen** is a from-scratch, modular LLM inference engine designed to teach and demonstrate how modern inference systems such as vLLM, Ollama, TensorRT-LLM, and similar runtimes work internally.

The project takes a Hugging Face causal language model and builds the complete inference stack around it:

**request handling → scheduling → batching → model execution → KV caching → token generation → streaming → profiling → optimization**

The defining requirement is:

> **MicroGen MUST support both CPU-only and GPU-accelerated inference.**

CPU is **not** a fallback mode. It is a first-class execution target.

The same MicroGen engine and API should be able to run:

```bash
microgen serve --model sshleifer/tiny-gpt2 --device cpu
```

and:

```bash
microgen serve --model sshleifer/tiny-gpt2 --device cuda
```

without requiring two completely different implementations.

---

# 1. The Problem

Calling:

```python
model.generate(...)
```

hides almost everything that actually happens inside an inference system.

A real inference engine has to answer questions like:

* Where does a request wait?
* When should it execute?
* Can multiple requests be batched?
* When should a request enter or leave a batch?
* How are tokens generated?
* Where is the KV cache stored?
* How much memory does each request consume?
* Should computation happen on CPU or GPU?
* How much time is spent transferring tensors?
* Why is one request slow?
* Why does throughput collapse when batch size increases?
* What happens when GPU memory is insufficient?
* How do multiple users share the same model?
* How can generated tokens be streamed?
* How do we measure TTFT vs decode latency?
* Which optimization actually improved performance?

MicroGen exists to expose and solve these problems.

---

# 2. High-Level Architecture

```text
                         Clients
                            │
                            ▼
                  ┌──────────────────┐
                  │   HTTP API       │
                  │   /generate      │
                  │   /stream        │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Rate Limiter    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Request Queue   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    Scheduler     │
                  │                  │
                  │ Admission        │
                  │ Batching         │
                  │ Continuous Batch │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    MicroGen      │
                  │      Engine      │
                  └────────┬─────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
     ┌─────────────────┐       ┌─────────────────┐
     │  CPU Backend    │       │  GPU Backend    │
     │                 │       │                 │
     │ PyTorch CPU     │       │ PyTorch CUDA    │
     └────────┬────────┘       └────────┬────────┘
              │                         │
              └──────────┬──────────────┘
                         ▼
                  ┌──────────────┐
                  │  KV Cache    │
                  └──────────────┘

 Supporting systems:

 ┌────────────┐ ┌──────────────┐ ┌─────────────┐
 │ LRU Cache  │ │ Prefix Cache │ │ TTL Store   │
 └────────────┘ └──────────────┘ └─────────────┘

 ┌──────────────┐ ┌────────────────┐
 │ Rate Limiter │ │ Profiler       │
 └──────────────┘ │ + Diagnostics  │
                  └────────────────┘

 ┌──────────────┐
 │ ONNX Backend │
 └──────────────┘
```

---

# 3. Core Design Philosophy

MicroGen should be designed around **abstraction**, not a giant collection of `if/elif` statements.

For example, the engine should not become:

```python
if device == "cpu":
    ...
elif device == "cuda":
    ...
elif device == "something":
    ...
```

everywhere.

Instead:

```text
                InferenceBackend
                       │
             ┌─────────┴─────────┐
             │                   │
       PyTorchBackend       ONNXBackend
             │
       ┌─────┴─────┐
       │           │
   CPUDevice   CUDADevice
```

The engine interacts with interfaces.

For example:

```python
backend.generate(...)
backend.prefill(...)
backend.decode(...)
backend.memory_available()
```

The backend decides how that actually happens.

This makes MicroGen extensible.

Adding another backend should mean implementing an interface, **not rewriting the scheduler**.

---

# 4. Supported Hardware

## CPU

MicroGen must work on a machine with:

```text
CPU
RAM
No GPU
```

Example:

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cpu
```

The CPU implementation should support:

* model execution
* batching
* KV cache
* streaming
* profiling
* caching
* scheduling

---

# 5. GPU

GPU support initially targets CUDA through PyTorch.

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cuda
```

The GPU backend should eventually support:

* CUDA execution
* GPU memory tracking
* GPU batching
* KV cache in VRAM
* CPU ↔ GPU transfer measurement
* CUDA synchronization-aware profiling
* GPU throughput measurements

---

# 6. Model Support

The initial model should deliberately be tiny.

### Primary integration model

```text
sshleifer/tiny-gpt2
```

This keeps development fast and makes debugging practical.

Eventually MicroGen should support the general interface:

```text
Hugging Face Causal LM
        ↓
MicroGen Model Adapter
        ↓
MicroGen Engine
```

The engine shouldn't be permanently coupled to GPT-2.

---

# 7. Phase 1 — Build the Inference Engine

First, stop relying on:

```python
model.generate()
```

and understand what generation actually requires.

Implement:

```text
prefill()
decode()
sample()
generate()
```

Conceptually:

```text
Prompt
  │
  ▼
Tokenization
  │
  ▼
Prefill
  │
  ├── K
  └── V
  │
  ▼
KV Cache
  │
  ▼
Decode
  │
  ▼
Logits
  │
  ▼
Sampling
  │
  ▼
Next Token
  │
  └──────► Decode again
```

This becomes the heart of MicroGen.

---

# 8. Prefill

Given:

```text
"I went to the beach"
```

the model processes the prompt and generates the initial attention states.

The engine should measure:

```text
prefill latency
input tokens
memory usage
```

This is especially important because prefill behaves differently from decode.

---

# 9. Decode

After the prompt has been processed, MicroGen generates one token at a time.

Conceptually:

```text
token
 ↓
model
 ↓
logits
 ↓
sampling
 ↓
next token
 ↓
model
 ↓
...
```

Decode performance should be measured in:

```text
tokens / second
```

---

# 10. KV Cache

One of the most important components.

Without KV caching:

```text
Every new token
    ↓
recompute previous tokens
    ↓
waste computation
```

With KV caching:

```text
Prompt
 ↓
K/V stored
 ↓
new token
 ↓
only necessary computation
 ↓
append new K/V
```

MicroGen should implement its own KV cache abstraction.

Something conceptually like:

```python
cache.allocate(request_id)
cache.append(request_id, key, value)
cache.get(request_id)
cache.free(request_id)
```

The cache must work on:

```text
CPU RAM
GPU VRAM
```

depending on execution configuration.

---

# 11. KV Cache Memory Management

The engine should know approximately:

```text
How much KV memory does one request consume?
How much memory is available?
How many requests can fit?
```

Eventually this allows the scheduler to make decisions such as:

```text
GPU has 2 GB available
        ↓
Current request needs 300 MB
        ↓
Can accept another request?
        ↓
YES / NO
```

This is the beginning of **memory-aware scheduling**.

---

# 12. Phase 2 — Request Queue

Instead of executing every request immediately:

```text
Request
   ↓
Queue
   ↓
Scheduler
   ↓
Execution
```

The queue should support:

```python
enqueue()
dequeue()
cancel()
length()
```

Each request gets timestamps such as:

```text
arrival_time
queue_start
execution_start
first_token
completion
```

This lets us calculate:

```text
queue latency
TTFT
generation latency
total latency
```

---

# 13. Phase 3 — Batching

Suppose five users send requests:

```text
A
B
C
D
E
```

Instead of:

```text
A → GPU
B → GPU
C → GPU
D → GPU
E → GPU
```

MicroGen should eventually do:

```text
A ┐
B │
C ├──► GPU
D │
E ┘
```

This is **batch inference**.

Benchmark:

```text
batch = 1
batch = 2
batch = 4
batch = 8
```

Measure:

```text
latency
throughput
GPU utilization
tokens/sec
memory
```

Important distinction:

> Larger batch size does not automatically mean lower latency.

Batching primarily improves **hardware utilization and throughput**.

---

# 14. Phase 4 — Continuous Batching

Static batching:

```text
Batch
A
B
C
D

wait until everyone finishes
```

is inefficient.

Instead:

```text
Time →

A ────────────────┐
B ────────────┐   │
C ────────┐   │   │
D ────────┼───┼───┼──►
          │   │   │
E ────────┘   │   │
              │   │
F ────────────┘   │
```

Requests can:

* enter a running batch
* generate tokens
* finish
* leave

without stopping the entire batch.

This is one of the most important concepts MicroGen is intended to teach.

---

# 15. Scheduler

The scheduler becomes the brain of MicroGen.

It decides:

```text
What runs?
When does it run?
Which requests are batched?
Can another request enter?
Should a request wait?
Is there enough memory?
```

Eventually:

```python
scheduler.schedule()
```

could produce something conceptually like:

```text
ExecutionPlan

device: CUDA
batch_size: 8
requests: [12, 14, 17, 19, 20, 21, 22, 25]
kv_memory_required: 640 MB
estimated_latency: 42 ms
```

This is particularly relevant to your Nerva work later.

---

# 16. Streaming API

The API shouldn't wait for the entire generation.

Instead:

```text
Request
   ↓
"Hello"
   ↓
"Hello world"
   ↓
"Hello world from"
   ↓
"Hello world from Goa"
```

The client receives tokens as they're generated.

For example:

```text
POST /generate/stream
```

using Server-Sent Events or another streaming mechanism.

---

# 17. API

MicroGen should expose a simple interface.

For example:

```text
POST /generate
POST /generate/stream
GET  /health
GET  /metrics
GET  /diagnose
```

A request could look conceptually like:

```json
{
  "prompt": "Explain neural networks",
  "max_tokens": 100,
  "temperature": 0.7
}
```

The response should include useful metadata.

For example:

```text
generated_text
input_tokens
output_tokens
ttft_ms
total_latency_ms
tokens_per_second
```

---

# 18. LRU Cache

Implement an application-level LRU cache.

Potential use:

```text
prompt/result
   ↓
LRU
```

Repeated requests can avoid unnecessary work.

The implementation should expose concepts like:

```python
cache.get(key)
cache.put(key, value)
cache.remove(key)
cache.clear()
```

This teaches a fundamental systems concept while also providing a useful optimization.

---

# 19. Prefix Cache

More interesting than ordinary result caching.

Suppose:

```text
User A:
"Explain the history of Goa and..."

User B:
"Explain the history of Goa and..."
```

The prefix is identical.

Instead of recomputing the prefix KV states:

```text
Shared prefix
      ↓
Prefix KV Cache
      ↓
 ┌────┴────┐
 A         B
```

MicroGen can reuse the prefix.

This introduces concepts related to modern LLM serving systems.

---

# 20. TTL Key-Value Store

Build a small time-based store:

```python
store.set(key, value, ttl=60)
store.get(key)
store.delete(key)
```

Entries automatically expire.

Potential uses:

```text
temporary sessions
request metadata
temporary cached results
client state
```

This is deliberately a systems-infrastructure component rather than an LLM-specific component.

---

# 21. Rate Limiter

Implement a sliding-window rate limiter.

Example:

```text
10 requests / 60 seconds / client
```

Interface:

```python
rate_limiter.allow(client_id)
```

The system should reject excessive requests before they consume inference resources.

This teaches an important distinction:

```text
API infrastructure
        ≠
Inference infrastructure
```

---

# 22. CPU vs GPU Execution

This is a **core MicroGen feature**, not an optional extension.

MicroGen should allow:

```text
                 MicroGen Engine
                       │
              ┌────────┴────────┐
              │                 │
             CPU               GPU
              │                 │
          PyTorch CPU       PyTorch CUDA
```

The same:

```text
scheduler
queue
cache
API
generation logic
profiler
```

should operate on both.

Only the execution-specific implementation should differ.

---

# 23. CPU/GPU Benchmarking

MicroGen should have a benchmark suite comparing:

| Metric           | CPU | GPU |
| ---------------- | --: | --: |
| TTFT             |   ✓ |   ✓ |
| Decode tok/s     |   ✓ |   ✓ |
| p50 latency      |   ✓ |   ✓ |
| p95 latency      |   ✓ |   ✓ |
| Total latency    |   ✓ |   ✓ |
| Throughput       |   ✓ |   ✓ |
| Batch efficiency |   ✓ |   ✓ |
| KV memory        |   ✓ |   ✓ |
| Queue latency    |   ✓ |   ✓ |

This turns MicroGen into an actual experimental platform.

---

# 24. CPU-Specific Investigation

The CPU path should investigate:

```text
CPU utilization
memory bandwidth
thread count
batch size
RAM usage
KV cache size
```

Questions to answer experimentally:

> Why does increasing batch size eventually stop helping?

> Is the workload compute-bound or memory-bound?

> How does KV cache affect CPU inference?

---

# 25. GPU-Specific Investigation

GPU profiling should investigate:

```text
VRAM
GPU utilization
kernel execution
H2D transfers
D2H transfers
batch size
KV cache
```

For example:

```text
Request
  ↓
CPU tokenization
  ↓
CPU → GPU transfer
  ↓
GPU prefill
  ↓
GPU decode
  ↓
GPU → CPU
  ↓
Network
```

MicroGen should be able to tell you where time went.

---

# 26. ONNX Backend

ONNX introduces a second execution path.

The conceptual pipeline:

```text
PyTorch Model
     ↓
Export
     ↓
ONNX
     ↓
ONNX Runtime
```

MicroGen should implement an abstraction such as:

```text
InferenceBackend
       │
 ┌─────┴────────┐
 │              │
PyTorch        ONNX
```

The ONNX backend doesn't need to replace the primary LLM backend.

Its purpose is to teach:

```text
model representation
       ↓
deployment format
       ↓
execution runtime
```

and demonstrate that **ONNX is not itself an inference engine**.

---

# 27. Profiler

The profiler is one of MicroGen's most important components.

Every request should produce something like:

```text
queue_ms
tokenization_ms
cache_lookup_ms
prefill_ms
decode_ms
gpu_compute_ms
transfer_ms
network_ms
total_ms
input_tokens
output_tokens
```

This allows actual performance engineering.

Instead of saying:

> "The model is slow."

MicroGen should tell you:

```text
Total:             421 ms

Queue:              18 ms
Tokenization:        3 ms
Prefill:             71 ms
Decode:             294 ms
Transfers:           21 ms
Networking:          14 ms
```

Now you know what to optimize.

---

# 28. Latency Metrics

MicroGen should distinguish:

### TTFT

**Time To First Token**

```text
request
   ↓
first generated token
```

### Decode latency

Time spent generating subsequent tokens.

### Total latency

```text
request → final token
```

### Throughput

```text
tokens / second
```

### Tail latency

```text
p50
p95
p99
```

This matters because:

```text
average latency ≠ user experience
```

---

# 29. `/diagnose`

This can become MicroGen's signature feature.

After running a workload:

```bash
microgen diagnose
```

MicroGen analyzes the profiler data.

Example:

```text
MicroGen Performance Diagnosis
──────────────────────────────

Primary bottleneck:
    GPU decode

Evidence:
    71% of request time spent in decode

Secondary bottleneck:
    CPU → GPU transfers

Recommended experiments:
    1. Increase batch size
    2. Enable KV caching
    3. Reduce CPU/GPU synchronization
    4. Test quantized execution
```

Another workload might produce:

```text
Primary bottleneck:
    Queueing

Evidence:
    43% of total latency is queue time

Recommendation:
    Increase scheduler capacity or
    reduce admission pressure.
```

Or:

```text
Primary bottleneck:
    CPU memory bandwidth

Recommendation:
    Reduce batch size or investigate
    KV-cache memory pressure.
```

---

# 30. Optimization Experiments

MicroGen should not merely contain optimizations.

It should **prove whether they work**.

Every optimization follows:

```text
Baseline
   ↓
Implement optimization
   ↓
Benchmark
   ↓
Compare
   ↓
Keep / revert
```

Examples:

### Experiment 1

```text
No KV cache
vs
KV cache
```

### Experiment 2

```text
Batch size 1
vs
Batch size 4
vs
Batch size 8
```

### Experiment 3

```text
CPU
vs
GPU
```

### Experiment 4

```text
Static batching
vs
Continuous batching
```

### Experiment 5

```text
No prefix cache
vs
Prefix cache
```

---

# 31. Benchmark Suite

MicroGen should have:

```text
benchmarks/
├── latency.py
├── throughput.py
├── batching.py
├── kv_cache.py
├── cpu_vs_gpu.py
├── prefix_cache.py
└── end_to_end.py
```

Each benchmark should produce reproducible results.

For example:

```text
MicroGen Benchmark
────────────────────────

Model: tiny-gpt2
Device: CUDA
Prompt: 128 tokens
Generation: 128 tokens

Batch     TTFT       tok/s      p95
────────────────────────────────────
1         18 ms      42         31 ms
2         21 ms      76         39 ms
4         28 ms      133        51 ms
8         42 ms      221        73 ms
```

---

# 32. Project Structure

A clean implementation could look like:

```text
microgen/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── microgen/
│   │
│   ├── api/
│   │   ├── server.py
│   │   ├── routes.py
│   │   └── streaming.py
│   │
│   ├── runtime/
│   │   ├── engine.py
│   │   ├── generation.py
│   │   ├── model.py
│   │   ├── sampling.py
│   │   └── kv_cache.py
│   │
│   ├── scheduler/
│   │   ├── queue.py
│   │   ├── scheduler.py
│   │   ├── batch.py
│   │   └── request.py
│   │
│   ├── backends/
│   │   ├── base.py
│   │   ├── pytorch.py
│   │   └── onnx.py
│   │
│   ├── devices/
│   │   ├── base.py
│   │   ├── cpu.py
│   │   └── cuda.py
│   │
│   ├── cache/
│   │   ├── lru.py
│   │   ├── prefix.py
│   │   └── ttl.py
│   │
│   ├── infra/
│   │   ├── rate_limiter.py
│   │   └── metrics.py
│   │
│   ├── profiler/
│   │   ├── profiler.py
│   │   ├── events.py
│   │   └── diagnose.py
│   │
│   └── config/
│       └── settings.py
│
├── benchmarks/
│   ├── latency.py
│   ├── throughput.py
│   ├── batching.py
│   ├── kv_cache.py
│   ├── cpu_vs_gpu.py
│   └── prefix_cache.py
│
├── tests/
│   ├── runtime/
│   ├── scheduler/
│   ├── cache/
│   ├── backends/
│   └── api/
│
└── examples/
    ├── basic.py
    ├── streaming.py
    └── benchmark.py
```

---

# 33. CLI

The final project should feel like an actual tool.

### Start server

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cpu
```

or:

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cuda
```

### Benchmark

```bash
microgen benchmark
```

### Diagnose

```bash
microgen diagnose
```

### Model information

```bash
microgen model-info
```

---

# 34. Testing Strategy

Testing shouldn't only test whether the HTTP endpoint returns something.

There should be multiple layers.

### Unit tests

```text
KV cache
LRU cache
TTL store
rate limiter
scheduler
sampler
request lifecycle
```

### Integration tests

```text
model → engine
engine → scheduler
scheduler → backend
API → engine
```

### Hardware tests

```text
CPU execution
CUDA execution
```

GPU tests should be skipped gracefully on machines without CUDA.

---

# 35. The Most Important Engineering Rule

MicroGen should follow:

> **Measure before optimizing.**

For every performance claim:

```text
"I made it faster"
```

must be backed by:

```text
before
after
workload
hardware
metric
```

For example:

```text
Baseline:
42 tok/s

KV cache:
91 tok/s

Improvement:
+116%
```

This makes MicroGen an **inference systems research project**, not just a toy server.

---

# 36. What You Will Learn

By completing MicroGen, you'll have touched essentially the entire inference stack:

```text
                LLM Inference
                     │
       ┌─────────────┼──────────────┐
       │             │              │
   Model           Runtime       Hardware
       │             │              │
Transformers     PyTorch          CPU
Tokenizer        KV Cache         GPU
Architecture     Sampling        CUDA
                 Batching
                     │
              ┌──────┴──────┐
              │             │
          Scheduling      Memory
              │             │
          Queueing       KV Cache
          Admission      LRU
          Batching       Prefix
          Streaming      TTL
              │
              ▼
          Performance
              │
          Profiling
          Benchmarking
          Diagnostics
          Optimization
```

---

# 37. What MicroGen Is NOT

We deliberately **do not** try to recreate all of vLLM.

Do **not** initially build:

* distributed inference
* tensor parallelism
* pipeline parallelism
* multi-node serving
* custom CUDA kernels from scratch
* support for dozens of architectures
* Kubernetes deployment
* authentication systems
* enterprise monitoring
* sophisticated distributed KV caches

Those are future directions.

The goal is to build a **small system where every component is understandable**.

---

# 38. Final Goal

At the end, you should be able to run:

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cuda
```

and have:

```text
                    MicroGen
                       │
          ┌────────────┴────────────┐
          │                         │
       Requests                 Hardware
          │                         │
       Queue                   CPU / GPU
          │                         │
      Scheduler                PyTorch
          │                         │
      Batching                 KV Cache
          │                         │
     Continuous                Generation
      Batching                     │
          │                         │
       Streaming ◄─────────────────┘
          │
       Metrics
          │
       Profiler
          │
      Diagnosis
```

Then switch to:

```bash
microgen serve \
    --model sshleifer/tiny-gpt2 \
    --device cpu
```

and **the same system works**.

That CPU/GPU duality is fundamental to MicroGen.

---

# 39. The Bigger Picture

The really interesting part is how MicroGen connects to **Nerva**.

Think of them as two layers:

```text
                    Nerva
          ┌────────────────────────┐
          │ Hardware profiling     │
          │ Cost modeling          │
          │ Execution planning     │
          │ Memory decisions       │
          │ CPU/GPU decisions      │
          │ Offloading decisions   │
          └───────────┬────────────┘
                      │
                Execution Plan
                      │
                      ▼
                   MicroGen
          ┌────────────────────────┐
          │ Scheduler              │
          │ Runtime                │
          │ KV Cache               │
          │ Batching               │
          │ CPU/GPU Backend        │
          │ Streaming              │
          └────────────────────────┘
```

**MicroGen teaches you how the inference runtime works.**

**Nerva eventually learns how to decide the optimal way to execute workloads on available hardware.**

That makes MicroGen a very good practical foundation for the inference-planning work you're already doing with Nerva.

### The final one-line definition

> **MicroGen is a modular, hardware-aware LLM inference engine built from scratch to understand and optimize modern model serving, with first-class CPU and GPU execution, scheduling, batching, KV caching, streaming, profiling, and performance diagnostics.**
