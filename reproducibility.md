# MicroGen Empirical Research & Reproducibility Package (v2)

**Paper Title**: *MicroGen: An Empirical Study of When LLM Inference Optimizations Help—and When They Hurt*

This package documents the environment specification, execution manifests, correctness validation gates, and 100% data-driven pipeline required to replicate all empirical benchmark findings presented in the paper.

---

## 1. Core Principles & Provenance

1. **Zero Hardcoded Figures/Tables**: `scripts/export_paper_tables.py` and `scripts/generate_paper_figures.py` read 100% dynamically from raw JSON/JSONL artifacts (`results/raw/experiments.jsonl`). Synthetic/mock fallbacks are strictly disabled.
2. **Correctness First**: Before any benchmark pass, all backends must pass token-by-token greedy output identity checks and logit cosine similarity gates (`benchmarks/correctness.py`).
3. **Statistical Protocol**: $W=5$ warmup passes, $N=30$ independent repetitions per configuration, $M \ge 100$ request observations for tail latency estimation.
4. **Full Provenance Chain**: Every raw trial record logs:
   - `experiment_id` & `git_commit` hash
   - `wall_clock_timestamp` (ISO 8601)
   - `hardware_name`, `cuda_version`, `pytorch_version`, `python_version`
   - `seed` & `trial_number`

---

## 2. Environment Setup

```bash
git clone https://github.com/Omdeepb69/MicroGen.git
cd MicroGen
pip install -r requirements.txt
export PYTHONPATH=.
```

---

## 3. Correctness Validation

Before executing benchmark sweeps, run the correctness gate suite:

```bash
python -m benchmarks.correctness
```

---

## 4. Manifest-Driven Benchmark Execution

All experiment parameters are governed by manifests in `experiments/manifests/`:

```bash
# Correctness verification
python -m benchmarks.correctness

# Modular experiment sweeps
python experiments/context_sweep.py
python experiments/prefix_sharing.py
python experiments/quant_lifecycle.py
python experiments/speculative_sweep.py
python experiments/batching_concurrency.py
python experiments/paged_memory_pressure.py

# Full Kaggle GPU Suite
python scripts/kaggle_benchmark_runner.py
```

---

## 5. Data-Driven Synthesis

To synthesize publication figures and LaTeX tables directly from empirical logs:

```bash
# Export LaTeX tables (paper/tables/*.tex)
python scripts/export_paper_tables.py

# Generate vector figures (paper/figures/*.pdf, *.png)
python scripts/generate_paper_figures.py
```
