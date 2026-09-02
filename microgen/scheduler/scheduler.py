"""Continuous batching scheduler managing dynamic request admission, prefill, and decode loops."""

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

    def add_request(self, request: Request) -> None:
        """Submit a new generation request to the scheduler queue."""
        self.request_queue.enqueue(request)

    def step(self) -> List[Request]:
        """Perform one iteration of continuous batching execution.

        Returns list of requests that finished in this step.
        """
        step_completed_requests: List[Request] = []

        # 1. Admit pending requests if batch capacity permits
        available_slots = self.max_batch_size - len(self.running_requests)
        if available_slots > 0 and not self.request_queue.is_empty():
            new_requests = self.request_queue.pop_batch(available_slots)
            if new_requests:
                prefill_completed = self._execute_prefill(new_requests)
                step_completed_requests.extend(prefill_completed)

        # 2. Execute decode step for existing running requests
        active_running = [req for req in self.running_requests if req.status == RequestStatus.RUNNING]
        if active_running:
            decode_completed = self._execute_decode(active_running)
            step_completed_requests.extend(decode_completed)

        # 3. Clean up running_requests list
        self.running_requests = [req for req in self.running_requests if req.status == RequestStatus.RUNNING]

        return step_completed_requests

    def _execute_prefill(self, new_requests: List[Request]) -> List[Request]:
        """Run prefill forward pass for newly admitted requests."""
        batch_id = f"prefill-{len(new_requests)}"
        batch = create_prefill_batch(
            batch_id=batch_id,
            requests=new_requests,
            pad_token_id=self.pad_token_id,
            device=getattr(self.backend.device, "device", None),
        )

        sampled_tokens: List[int] = []
        for idx, req in enumerate(new_requests):
            cache = self.kv_cache_manager.allocate(
                req.request_id, max_seq_len=req.num_prompt_tokens + req.max_new_tokens
            )
            input_ids = batch.input_ids[idx : idx + 1]
            attention_mask = batch.attention_mask[idx : idx + 1]

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
            token_val = int(sampled_id.item())
            sampled_tokens.append(token_val)

        completed_ids = update_requests_with_sampled_tokens(
            new_requests, sampled_tokens, eos_token_id=self.eos_token_id
        )

        completed_set: Set[str] = set(completed_ids)
        completed_requests: List[Request] = []

        for req in new_requests:
            if req.request_id in completed_set:
                self.kv_cache_manager.free(req.request_id)
                completed_requests.append(req)
            else:
                self.running_requests.append(req)

        return completed_requests

    def _execute_decode(self, running_requests: List[Request]) -> List[Request]:
        """Run decode step for active running requests."""
        next_tokens = [req.generated_token_ids[-1] for req in running_requests]
        batch_id = f"decode-{len(running_requests)}"

        batch = create_decode_batch(
            batch_id=batch_id,
            requests=running_requests,
            next_token_ids=next_tokens,
            device=getattr(self.backend.device, "device", None),
        )

        sampled_tokens: List[int] = []
        for idx, req in enumerate(running_requests):
            cache = self.kv_cache_manager.get(req.request_id)
            if cache is None:
                raise RuntimeError(f"Missing KV cache state for active request {req.request_id}")

            input_ids = batch.input_ids[idx : idx + 1]
            attention_mask = batch.attention_mask[idx : idx + 1]

            logits, _ = self.backend.decode(
                token_ids=input_ids, attention_mask=attention_mask, cache=cache
            )

            sampled_id = self.backend.sample(
                logits,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
            token_val = int(sampled_id.item())
            sampled_tokens.append(token_val)

        completed_ids = update_requests_with_sampled_tokens(
            running_requests, sampled_tokens, eos_token_id=self.eos_token_id
        )

        completed_set: Set[str] = set(completed_ids)
        completed_requests: List[Request] = []

        for req in running_requests:
            if req.request_id in completed_set:
                self.kv_cache_manager.free(req.request_id)
                completed_requests.append(req)

        return completed_requests

    def run_until_complete(self) -> List[Request]:
        """Execute step() continuously until all requests in queue and running pool finish."""
        all_completed: List[Request] = []
        while not self.request_queue.is_empty() or len(self.running_requests) > 0:
            finished = self.step()
            all_completed.extend(finished)
        return all_completed
