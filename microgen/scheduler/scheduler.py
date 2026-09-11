"""Continuous batching scheduler managing dynamic request admission, prefill, and decode loops."""

import time
from typing import Dict, List, Optional, Set
import torch

from microgen.backends.base import InferenceBackend
from microgen.runtime.kv_cache import KVCacheManager
from microgen.scheduler.batch import (
    create_decode_batch,
    create_prefill_batch,
    update_requests_with_sampled_tokens,
)
from microgen.scheduler.queue import Request, RequestQueue, RequestStatus


class ContinuousBatchingScheduler:
    """Continuous batching scheduler dynamically managing prefill and decode iterations."""

    def __init__(
        self,
        backend: InferenceBackend,
        kv_cache_manager: KVCacheManager,
        max_batch_size: int = 8,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
    ) -> None:
        self.backend = backend
        self.kv_cache_manager = kv_cache_manager
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.request_queue = RequestQueue()
        self.running_requests: List[Request] = []

        self.profiling_stats: Dict[str, float] = {
            "queue_pop_ms": 0.0,
            "kv_manage_ms": 0.0,
            "backend_cuda_ms": 0.0,
            "python_loop_ms": 0.0,
            "total_steps": 0.0,
        }

    def reset_profiling(self) -> None:
        """Reset accumulated micro-profiling latency metrics."""
        self.profiling_stats = {
            "queue_pop_ms": 0.0,
            "kv_manage_ms": 0.0,
            "backend_cuda_ms": 0.0,
            "python_loop_ms": 0.0,
            "total_steps": 0.0,
        }

    def get_profile_breakdown(self) -> Dict[str, float]:
        """Return granular micro-profiling latency breakdown and overhead ratios."""
        q_pop = self.profiling_stats["queue_pop_ms"]
        kv_mg = self.profiling_stats["kv_manage_ms"]
        b_cuda = self.profiling_stats["backend_cuda_ms"]
        py_loop = self.profiling_stats["python_loop_ms"]
        tot_ms = q_pop + kv_mg + b_cuda + py_loop
        overhead_ms = q_pop + kv_mg + py_loop
        overhead_ratio = (overhead_ms / tot_ms) if tot_ms > 0 else 0.0

        return {
            "queue_pop_ms": q_pop,
            "kv_manage_ms": kv_mg,
            "backend_cuda_ms": b_cuda,
            "python_loop_ms": py_loop,
            "total_scheduler_ms": tot_ms,
            "scheduler_overhead_ms": overhead_ms,
            "scheduler_overhead_ratio": overhead_ratio,
            "total_steps": self.profiling_stats["total_steps"],
        }

    def add_request(self, request: Request) -> None:
        """Submit a new generation request to the scheduler queue."""
        self.request_queue.enqueue(request)

    def step(self) -> List[Request]:
        """Perform one iteration of continuous batching execution.

        Returns list of requests that finished in this step.
        """
        self.profiling_stats["total_steps"] += 1.0
        step_completed_requests: List[Request] = []

        # 1. Admit pending requests if batch capacity permits
        available_slots = self.max_batch_size - len(self.running_requests)
        if available_slots > 0 and not self.request_queue.is_empty():
            t0_pop = time.perf_counter()
            new_requests = self.request_queue.pop_batch(available_slots)
            t1_pop = time.perf_counter()
            self.profiling_stats["queue_pop_ms"] += (t1_pop - t0_pop) * 1000.0

            if new_requests:
                prefill_completed = self._execute_prefill(new_requests)
                step_completed_requests.extend(prefill_completed)

        # 2. Execute decode step for existing running requests
        t0_py = time.perf_counter()
        active_running = [req for req in self.running_requests if req.status == RequestStatus.RUNNING]
        t1_py = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_py - t0_py) * 1000.0

        if active_running:
            decode_completed = self._execute_decode(active_running)
            step_completed_requests.extend(decode_completed)

        # 3. Clean up running_requests list
        t0_clean = time.perf_counter()
        self.running_requests = [req for req in self.running_requests if req.status == RequestStatus.RUNNING]
        t1_clean = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_clean - t0_clean) * 1000.0

        return step_completed_requests

    def _execute_prefill(self, new_requests: List[Request]) -> List[Request]:
        """Run prefill forward pass for newly admitted requests."""
        t0_batch = time.perf_counter()
        batch_id = f"prefill-{len(new_requests)}"
        batch = create_prefill_batch(
            batch_id=batch_id,
            requests=new_requests,
            pad_token_id=self.pad_token_id,
            device=getattr(self.backend.device, "device", None),
        )
        t1_batch = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_batch - t0_batch) * 1000.0

        sampled_tokens: List[int] = []
        for idx, req in enumerate(new_requests):
            t0_kv = time.perf_counter()
            cache = self.kv_cache_manager.allocate(
                req.request_id, max_seq_len=req.num_prompt_tokens + req.max_new_tokens
            )
            t1_kv = time.perf_counter()
            self.profiling_stats["kv_manage_ms"] += (t1_kv - t0_kv) * 1000.0

            input_ids = batch.input_ids[idx : idx + 1]
            attention_mask = batch.attention_mask[idx : idx + 1]

            t0_cuda = time.perf_counter()
            logits, _ = self.backend.prefill(
                input_ids=input_ids, attention_mask=attention_mask, cache=cache
            )

            # Sample next token for request
            sampled_id = self.backend.sample(
                logits,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
            t1_cuda = time.perf_counter()
            self.profiling_stats["backend_cuda_ms"] += (t1_cuda - t0_cuda) * 1000.0

            token_val = int(sampled_id.item())
            sampled_tokens.append(token_val)

        t0_post = time.perf_counter()
        completed_ids = update_requests_with_sampled_tokens(
            new_requests, sampled_tokens, eos_token_id=self.eos_token_id
        )

        completed_set: Set[str] = set(completed_ids)
        completed_requests: List[Request] = []

        for req in new_requests:
            if req.request_id in completed_set:
                t0_free = time.perf_counter()
                self.kv_cache_manager.free(req.request_id)
                t1_free = time.perf_counter()
                self.profiling_stats["kv_manage_ms"] += (t1_free - t0_free) * 1000.0
                completed_requests.append(req)
            else:
                self.running_requests.append(req)
        t1_post = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_post - t0_post) * 1000.0

        return completed_requests

    def _execute_decode(self, running_requests: List[Request]) -> List[Request]:
        """Run decode step for active running requests."""
        t0_batch = time.perf_counter()
        next_tokens = [req.generated_token_ids[-1] for req in running_requests]
        batch_id = f"decode-{len(running_requests)}"

        batch = create_decode_batch(
            batch_id=batch_id,
            requests=running_requests,
            next_token_ids=next_tokens,
            device=getattr(self.backend.device, "device", None),
        )
        t1_batch = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_batch - t0_batch) * 1000.0

        sampled_tokens: List[int] = []
        for idx, req in enumerate(running_requests):
            t0_kv = time.perf_counter()
            cache = self.kv_cache_manager.get(req.request_id)
            t1_kv = time.perf_counter()
            self.profiling_stats["kv_manage_ms"] += (t1_kv - t0_kv) * 1000.0

            if cache is None:
                raise RuntimeError(f"Missing KV cache state for active request {req.request_id}")

            input_ids = batch.input_ids[idx : idx + 1]
            attention_mask = batch.attention_mask[idx : idx + 1]

            t0_cuda = time.perf_counter()
            logits, _ = self.backend.decode(
                token_ids=input_ids, attention_mask=attention_mask, cache=cache
            )

            sampled_id = self.backend.sample(
                logits,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
            t1_cuda = time.perf_counter()
            self.profiling_stats["backend_cuda_ms"] += (t1_cuda - t0_cuda) * 1000.0

            token_val = int(sampled_id.item())
            sampled_tokens.append(token_val)

        t0_post = time.perf_counter()
        completed_ids = update_requests_with_sampled_tokens(
            running_requests, sampled_tokens, eos_token_id=self.eos_token_id
        )

        completed_set: Set[str] = set(completed_ids)
        completed_requests: List[Request] = []

        for req in running_requests:
            if req.request_id in completed_set:
                t0_free = time.perf_counter()
                self.kv_cache_manager.free(req.request_id)
                t1_free = time.perf_counter()
                self.profiling_stats["kv_manage_ms"] += (t1_free - t0_free) * 1000.0
                completed_requests.append(req)
        t1_post = time.perf_counter()
        self.profiling_stats["python_loop_ms"] += (t1_post - t0_post) * 1000.0

        return completed_requests

    def run_until_complete(self) -> List[Request]:
        """Execute step() continuously until all requests in queue and running pool finish."""
        all_completed: List[Request] = []
        while not self.request_queue.is_empty() or len(self.running_requests) > 0:
            finished = self.step()
            all_completed.extend(finished)
        return all_completed

