# MicroGen Benchmark Reproducibility Package

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
