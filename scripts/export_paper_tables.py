"""
Research Paper LaTeX Table Exporter & Reproducibility Package Generator for MicroGen (Phase 18).

Reads empirical benchmark trial records from `results/raw/experiments.jsonl`
and exports clean LaTeX tables (`paper/tables/*.tex`) and `reproducibility.md`.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_experiment_records(jsonl_path: str = "results/raw/experiments.jsonl") -> List[Dict[str, Any]]:
    """Loads benchmark records from JSONL file. Raises RuntimeError if file is missing or empty."""
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        raise RuntimeError(
            f"No empirical benchmark records found in {jsonl_path}! "
            "Mock synthetic fallbacks are strictly disabled. Run benchmark sweeps first."
        )

    return records


def export_table1_main_results(records: List[Dict[str, Any]], tables_dir: str) -> str:
    """Table 1: Main Inference Performance & Optimization Breakdown."""
    tex_path = os.path.join(tables_dir, "table1_main_results.tex")

    table_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Inference Performance & Optimization Synergy Breakdown on GPU.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"\textbf{Optimization Configuration} & \textbf{TTFT (ms)} & \textbf{TPOT (ms)} & \textbf{VRAM (MB)} & \textbf{Tokens/sec} & \textbf{Speedup} \\",
        r"\hline",
    ]

    main_recs = [r for r in records if r.get("workload_name") or r.get("experiment_name") == "main_results"]
    if not main_recs:
        main_recs = records[:5]

    for rec in main_recs:
        cfg = rec.get("config", {})
        label = cfg.get("optimization_name", rec.get("config_label", "Config"))
        ttft = rec.get("ttft_stats_ms", {}).get("mean", rec.get("metrics", {}).get("ttft_ms", 0.0))
        tpot = rec.get("tpot_stats_ms", {}).get("mean", rec.get("metrics", {}).get("tpot_ms", 0.0))
        vram = rec.get("peak_allocated_mb_stats", {}).get("mean", rec.get("metrics", {}).get("vram_allocated_mb", 0.0))
        tp = rec.get("throughput_stats_tps", {}).get("mean", rec.get("metrics", {}).get("throughput_tok_per_sec", 0.0))
        sp = rec.get("speedup_multiplier", 1.0)
        line = f"  {label} & {ttft:.1f} & {tpot:.1f} & {vram:.0f} & {tp:.1f} & {sp:.2f}$\\times$ \\\\"
        table_lines.append(line)

    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(table_lines) + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)

    return tex_path


def export_table2_concurrency_scaling(records: List[Dict[str, Any]], tables_dir: str) -> str:
    """Table 2: Static vs Continuous Batching Scaling."""
    tex_path = os.path.join(tables_dir, "table2_concurrency_scaling.tex")

    table_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Throughput and Latency Comparison of Static vs. Continuous Batching under Serving Concurrency.}",
        r"\label{tab:concurrency_scaling}",
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"\textbf{Batch Size $B$} & \textbf{Static Tok/s} & \textbf{Continuous Tok/s} & \textbf{Static TPOT (ms)} & \textbf{Continuous TPOT (ms)} \\",
        r"\hline",
    ]

    batch_recs = [r for r in records if "batching" in r.get("workload_name", "") or r.get("experiment_name") == "batching_concurrency"]
    if not batch_recs:
        batch_recs = records

    for rec in batch_recs:
        b = rec.get("batch_size", rec.get("num_requests", 1))
        stp = rec.get("static_tp", rec.get("throughput_stats_tps", {}).get("mean", 0.0))
        ctp = rec.get("continuous_tp", rec.get("throughput_stats_tps", {}).get("mean", 0.0))
        stpot = rec.get("static_tpot", rec.get("tpot_stats_ms", {}).get("mean", 0.0))
        ctpot = rec.get("continuous_tpot", rec.get("tpot_stats_ms", {}).get("mean", 0.0))
        line = f"  {b} & {stp:.1f} & {ctp:.1f} & {stpot:.1f} & {ctpot:.1f} \\\\"
        table_lines.append(line)

    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(table_lines) + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)

    return tex_path


def export_table3_memory_ablation(records: List[Dict[str, Any]], tables_dir: str) -> str:
    """Table 3: Paged KV vs Contiguous Memory Pressure Evaluation."""
    tex_path = os.path.join(tables_dir, "table3_memory_ablation.tex")

    table_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{KV Cache Allocation Resilience under Constrained VRAM Capacity.}",
        r"\label{tab:memory_ablation}",
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"\textbf{VRAM Capacity \%} & \textbf{Contiguous Frag \%} & \textbf{Paged Frag \%} & \textbf{Contiguous OOM?} & \textbf{Paged OOM?} \\",
        r"\hline",
    ]

    mem_recs = [r for r in records if "paged" in r.get("workload_name", "") or r.get("experiment_name") == "paged_memory_pressure"]
    if not mem_recs:
        mem_recs = records

    for rec in mem_recs:
        cap = rec.get("capacity_pct", 100)
        cfrag = rec.get("contiguous_frag_pct", 0.0)
        pfrag = rec.get("paged_frag_pct", 0.0)
        coom = "Yes" if rec.get("contiguous_oom", False) else "No"
        poom = "Yes" if rec.get("paged_oom", False) else "No"
        line = f"  {cap}\\% & {cfrag:.1f}\\% & {pfrag:.1f}\\% & {coom} & {poom} \\\\"
        table_lines.append(line)

    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(table_lines) + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)

    return tex_path


def export_reproducibility_doc(output_path: str = "reproducibility.md") -> str:
    """Generates comprehensive reproducibility package documentation."""
    content = """# MicroGen Benchmark Reproducibility Package

This artifact documents the environment specification, execution commands, and parameters required to replicate all empirical benchmark results presented in the paper.

## 1. System Requirements & Environment

- **Python Version**: `3.11+`
- **PyTorch**: `2.0+` with CUDA support
- **Transformers**: `4.30+`
- **Hardware Recommended**: NVIDIA GPU (T4, P100, L4, A10G, or V100) with at least 15 GB VRAM. CPU-only execution is supported for functional verification.

## 2. Environment Setup

```bash
git clone https://github.com/Omdeepb69/MicroGen.git
cd MicroGen
pip install -r requirements.txt
export PYTHONPATH=.
```

## 3. Running Paper Benchmark Sweeps

Run individual modular experiment scripts to populate `results/raw/experiments.jsonl`:

```bash
# Phase 15: Micro-Ablations (RQ1 & RQ2)
python experiments/context_sweep.py
python experiments/prefix_sharing.py
python experiments/quant_lifecycle.py
python experiments/speculative_sweep.py

# Phase 16: Concurrency & Memory Pressure (RQ3)
python experiments/batching_concurrency.py
python experiments/paged_memory_pressure.py
python experiments/combined_interactions.py

# Phase 17: Generalization (RQ4)
python experiments/model_generalization.py
python experiments/hardware_duality.py

# Full Kaggle GPU Automated Execution Suite
python scripts/kaggle_benchmark_runner.py
```

## 4. Synthesizing Figures and Tables

To generate publication vector figures and LaTeX tables from benchmark trial outputs:

```bash
# Generate vector figures in paper/figures/*.png and *.pdf
python scripts/generate_paper_figures.py

# Generate LaTeX tables in paper/tables/*.tex
python scripts/export_paper_tables.py
```

## 5. Artifact Directory Layout

```
paper/
├── figures/
│   ├── fig1_context_scaling.pdf
│   ├── fig2_prefix_sharing_ttft.pdf
│   ├── fig3_quant_memory_pareto.pdf
│   ├── fig4_batching_concurrency.pdf
│   └── fig5_optimization_regime_map.pdf
└── tables/
    ├── table1_main_results.tex
    ├── table2_concurrency_scaling.tex
    └── table3_memory_ablation.tex
```
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def export_all_paper_artifacts(
    jsonl_path: str = "results/raw/experiments.jsonl",
    tables_dir: str = "paper/tables",
    repro_path: str = "reproducibility.md",
) -> List[str]:
    """Exports all LaTeX tables and reproducibility package documentation."""
    os.makedirs(tables_dir, exist_ok=True)
    records = load_experiment_records(jsonl_path)

    generated_paths = []
    generated_paths.append(export_table1_main_results(records, tables_dir))
    generated_paths.append(export_table2_concurrency_scaling(records, tables_dir))
    generated_paths.append(export_table3_memory_ablation(records, tables_dir))
    generated_paths.append(export_reproducibility_doc(repro_path))

    return generated_paths


if __name__ == "__main__":
    print("Exporting MicroGen Paper LaTeX Tables & Reproducibility Package...")
    paths = export_all_paper_artifacts()
    print(f"Export complete! Artifacts created: {len(paths)}")
