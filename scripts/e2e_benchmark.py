"""End-to-End Performance Benchmarking Suite for MicroGen LLM Inference Engine.

Measures Time To First Token (TTFT), Inter-Token Latency (ITL), aggregate throughput (tokens/sec),
and hardware memory utilization across concurrency levels.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.backends.pytorch import PyTorchBackend
from microgen.devices import get_device
from microgen.runtime import KVCacheManager
from microgen.scheduler import ContinuousBatchingScheduler
from microgen.scheduler.queue import Request


def run_e2e_benchmark(
    model_name: str = "sshleifer/tiny-gpt2",
    device_name: str = "cpu",
    prompt: str = "In a warm summer evening, the ancient forest whispered stories of",
    max_new_tokens: int = 20,
    concurrency_levels: List[int] = [1, 2, 4],
    output_json: str = "benchmark_results.json",
) -> Dict[str, Any]:
    """Execute end-to-end multi-concurrency benchmark suite."""
    print("==========================================================")
    print(" MicroGen End-to-End Inference Engine Benchmarking Suite ")
    print("==========================================================")
    print(f"Model: {model_name}")
    print(f"Device: {device_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max New Tokens per Request: {max_new_tokens}")
    print(f"Concurrency Levels: {concurrency_levels}")
    print("----------------------------------------------------------")

    device = get_device(device_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    backend = PyTorchBackend(device=device)
    backend.load_model(model_name, model_instance=model)

    prompt_ids = tokenizer.encode(prompt)

    benchmark_summary = {
        "model": model_name,
        "device": device_name,
        "max_new_tokens": max_new_tokens,
        "concurrency_results": [],
    }

    for concurrency in concurrency_levels:
        print(f"\n[Benchmarking Concurrency = {concurrency}]")
        kv_manager = KVCacheManager()
        scheduler = ContinuousBatchingScheduler(
            backend=backend,
            kv_cache_manager=kv_manager,
            max_batch_size=concurrency,
            eos_token_id=tokenizer.eos_token_id or 50256,
        )

        requests = []
        for i in range(concurrency):
            req = Request(
                request_id=f"bench-req-{concurrency}-{i}",
                prompt=prompt,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
            )
            requests.append(req)
            scheduler.add_request(req)

        start_wall_time = time.perf_counter()

        # Step-by-step loop to record TTFT and token arrival timestamps
        token_timestamps: Dict[str, List[float]] = {r.request_id: [] for r in requests}

        while not scheduler.request_queue.is_empty() or len(scheduler.running_requests) > 0:
            step_start = time.perf_counter()
            finished_reqs = scheduler.step()
            step_end = time.perf_counter()

            for req in scheduler.running_requests + finished_reqs:
                if req.num_generated_tokens > len(token_timestamps[req.request_id]):
                    token_timestamps[req.request_id].append(step_end)

        end_wall_time = time.perf_counter()
        total_time_sec = end_wall_time - start_wall_time

        # Calculate metrics
        ttft_list: List[float] = []
        itl_list: List[float] = []
        total_generated_tokens = sum(len(r.generated_token_ids) for r in requests)

        for req in requests:
            ts = token_timestamps[req.request_id]
            if ts:
                # TTFT is first token time minus benchmark start wall time
                ttft = (ts[0] - start_wall_time) * 1000.0  # ms
                ttft_list.append(ttft)

                # ITL is average time between consecutive tokens
                if len(ts) > 1:
                    itls = [(ts[j] - ts[j - 1]) * 1000.0 for j in range(1, len(ts))]
                    itl_list.append(sum(itls) / len(itls))

        avg_ttft_ms = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
        avg_itl_ms = sum(itl_list) / len(itl_list) if itl_list else 0.0
        throughput_tps = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0.0

        mem_info = backend.get_memory_usage()

        res_entry = {
            "concurrency": concurrency,
            "total_requests": concurrency,
            "total_generated_tokens": total_generated_tokens,
            "total_time_sec": round(total_time_sec, 4),
            "throughput_tokens_per_sec": round(throughput_tps, 2),
            "avg_ttft_ms": round(avg_ttft_ms, 2),
            "avg_itl_ms": round(avg_itl_ms, 2),
            "memory_usage": mem_info,
        }
        benchmark_summary["concurrency_results"].append(res_entry)

        print(f" Total Time: {total_time_sec:.4f} s")
        print(f" Total Tokens Generated: {total_generated_tokens}")
        print(f" Throughput: {throughput_tps:.2f} tokens/sec")
        print(f" Avg Time To First Token (TTFT): {avg_ttft_ms:.2f} ms")
        print(f" Avg Inter-Token Latency (ITL): {avg_itl_ms:.2f} ms")

    # Save structured report
    with open(output_json, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n==========================================================")
    print(f" Benchmark complete. Results written to {output_json}")
    print("==========================================================")
    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroGen E2E Performance Benchmark")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Model name or path")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--prompt", default="In a warm summer evening, the ancient forest whispered", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max new tokens to generate")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON output file")

    args = parser.parse_args()
    run_e2e_benchmark(
        model_name=args.model,
        device_name=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        output_json=args.output,
    )
