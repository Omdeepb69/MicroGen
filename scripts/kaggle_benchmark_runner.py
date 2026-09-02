"""Kaggle automated benchmark runner and standalone HTML performance report generator."""

import os
import sys
import time
import json
import torch
from typing import Dict, Any, List

from microgen.devices import get_device, CUDADevice
from microgen.backends import PyTorchBackend, QuantizedPyTorchBackend, TensorParallelPyTorchBackend
from microgen.runtime import KVCacheState

MODEL_NAME = "sshleifer/tiny-gpt2"


def generate_html_report(results: Dict[str, Any], output_path: str = "microgen_benchmark_report.html") -> None:
    """Generate a modern, self-contained HTML benchmark report with CSS charts."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MicroGen Benchmark Report - Kaggle GPU/CPU Suite</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --accent-blue: #38bdf8;
            --accent-purple: #a855f7;
            --accent-green: #34d399;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        h1 {{
            font-size: 2.25rem;
            margin: 0 0 10px 0;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .meta-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 15px;
        }}
        .meta-label {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 5px;
        }}
        .meta-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--accent-blue);
        }}
        .table-container {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            margin-bottom: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th, td {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: #0f172a;
            color: #94a3b8;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-fp32 {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); }}
        .badge-int8 {{ background: rgba(52, 211, 153, 0.2); color: var(--accent-green); }}
        .badge-tp {{ background: rgba(168, 85, 247, 0.2); color: var(--accent-purple); }}
        .metric-bar {{
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            margin-top: 6px;
        }}
        footer {{
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ MicroGen Inference Performance Report</h1>
            <p style="color: #94a3b8; margin: 0;">Automated Benchmark Suite executed on Kaggle / Hardware Runtime</p>
        </header>

        <div class="meta-grid">
            <div class="meta-card">
                <div class="meta-label">Model Target</div>
                <div class="meta-value">{results.get('model_name', MODEL_NAME)}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">CUDA Available</div>
                <div class="meta-value">{results.get('cuda_available', False)}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">GPU Devices Count</div>
                <div class="meta-value">{results.get('device_count', 0)}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Timestamp</div>
                <div class="meta-value">{results.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))}</div>
            </div>
        </div>

        <h2 style="font-size: 1.5rem; margin-bottom: 20px;">Execution Metrics Comparison</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Backend Strategy</th>
                        <th>TTFT (ms)</th>
                        <th>ITL (ms/tok)</th>
                        <th>Throughput (tok/sec)</th>
                        <th>Memory Usage (MB)</th>
                    </tr>
                </thead>
                <tbody>
"""
    for backend_name, metrics in results.get("backends", {}).items():
        badge_cls = "badge-fp32"
        if "quantized" in backend_name.lower():
            badge_cls = "badge-int8"
        elif "tensor_parallel" in backend_name.lower():
            badge_cls = "badge-tp"

        html_content += f"""
                    <tr>
                        <td>
                            <span class="badge {badge_cls}">{backend_name}</span>
                        </td>
                        <td><strong>{metrics.get('ttft_ms', 0):.2f}</strong> ms</td>
                        <td><strong>{metrics.get('itl_ms', 0):.2f}</strong> ms</td>
                        <td>
                            <strong>{metrics.get('throughput_tps', 0):.1f}</strong> tok/s
                            <div class="metric-bar" style="width: {min(100, int(metrics.get('throughput_tps', 0) * 2))}%;"></div>
                        </td>
                        <td>{metrics.get('memory_mb', 0):.2f} MB</td>
                    </tr>
"""
    html_content += """
                </tbody>
            </table>
        </div>

        <footer>
            Generated automatically by MicroGen Inference Engine — Kaggle & Multi-GPU Benchmark Suite
        </footer>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML benchmark report successfully written to: {output_path}")


def run_benchmarks(prompt: str = "MicroGen LLM inference engine delivers high performance", gen_tokens: int = 16) -> Dict[str, Any]:
    """Execute end-to-end performance benchmarks across available backends."""
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "model_name": MODEL_NAME,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backends": {},
    }

    # Define backend configurations to test
    configurations = [
        ("PyTorch (FP32)", PyTorchBackend(device=device)),
        ("QuantizedPyTorch (INT8)", QuantizedPyTorchBackend(device=device)),
        ("TensorParallelPyTorch (TP=2)", TensorParallelPyTorchBackend(world_size=2, devices=[device, device])),
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for name, backend in configurations:
        print(f"Benchmarking backend: {name}...")
        backend.load_model(MODEL_NAME)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Measure TTFT (Prefill step)
        start_ttft = time.perf_counter()
        logits, cache = backend.prefill(input_ids)
        ttft_ms = (time.perf_counter() - start_ttft) * 1000.0

        # Measure Decode steps for ITL
        decode_times = []
        sampled_tokens = []
        current_token = backend.sample(logits)
        sampled_tokens.append(current_token)

        for _ in range(gen_tokens - 1):
            start_decode = time.perf_counter()
            dec_logits, cache = backend.decode(current_token, cache=cache)
            decode_times.append((time.perf_counter() - start_decode) * 1000.0)
            current_token = backend.sample(dec_logits)
            sampled_tokens.append(current_token)

        avg_itl_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
        total_time_sec = (ttft_ms + sum(decode_times)) / 1000.0
        throughput_tps = gen_tokens / total_time_sec if total_time_sec > 0 else 0.0

        mem_info = backend.get_memory_usage()
        memory_mb = mem_info.get("vram_used_bytes", mem_info.get("ram_used_bytes", 0)) / (1024 * 1024)

        results["backends"][name] = {
            "ttft_ms": round(ttft_ms, 2),
            "itl_ms": round(avg_itl_ms, 2),
            "throughput_tps": round(throughput_tps, 2),
            "memory_mb": round(memory_mb, 2),
        }

    return results


def main() -> None:
    print("Starting MicroGen Kaggle & Multi-GPU Benchmark Suite...")
    results = run_benchmarks()

    # Save JSON metrics
    json_path = "kaggle_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to {json_path}")

    # Render HTML Report
    generate_html_report(results, output_path="microgen_benchmark_report.html")


if __name__ == "__main__":
    main()
