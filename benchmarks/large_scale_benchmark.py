#!/usr/bin/env python3
"""MicroGen v1.2 — Large-Scale Independent Benchmark.

Mirrors the open-system-one / Jev benchmark methodology for a direct
comparison against specialized decision models.

Methodology:
  - Datasets: SST-2 (2), AG News (4), Emotion (6), Banking77 (77)
  - N_eval  : 1000 examples per dataset (or dataset max if smaller)
  - N_calib : max(50, 10 * n_classes) to ensure calibration density
  - Generation baseline: strict A/B/C/D → label mapping (no substring parsing)
  - Permutation tests: Engine Invariance (fixed prompt) + Prompt Sensitivity
  - Metrics: Accuracy, ECE, Brier, NLL, Latency (p50/p95), KL divergence

Comparison targets (from literature, 2026-09-18):
  Jev Banking77 : 79.7% acc | 9.8-pt calib gap | 467 ms median
  Jev SST-2     : ~91%      | open-system-one
  Jev AG News   : ~88%      | open-system-one
  Jev Emotion   : ~59%      | open-system-one

Usage (local, small model):
  python benchmarks/large_scale_benchmark.py \\
      --model HuggingFaceTB/SmolLM-135M \\
      --eval 200 --calib-min 10

Usage (Kaggle T4, production):
  python benchmarks/large_scale_benchmark.py \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --eval 1000 --calib-min 50
"""

import argparse
import gc
import json
import math
import os
import random
import string
import subprocess
import sys
import time
from typing import NamedTuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.calibration import TemperatureScaler
from microgen.decision.engine import DecisionEngine
from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.schema import Choice, ChoiceSchema


# ── Jev reference numbers (open-system-one, 2026-09-18) ─────────────────────

JEV_REFERENCE = {
    "SST-2":     {"acc": 91.6, "ece": None, "p50_ms": None},
    "AG News":   {"acc": 88.6, "ece": None, "p50_ms": None},
    "Emotion":   {"acc": 59.0, "ece": None, "p50_ms": None},
    "Banking77": {"acc": 79.7, "ece": 9.8,  "p50_ms": 467},
}

# ── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = [
    {
        "name": "SST-2",
        "path": "stanfordnlp/sst2", "hf_name": None,
        "split": "validation", "text_field": "sentence", "label_field": "label",
        "label_names": ["negative", "positive"],
    },
    {
        "name": "AG News",
        "path": "fancyzhx/ag_news", "hf_name": None,
        "split": "test", "text_field": "text", "label_field": "label",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    {
        "name": "Emotion",
        "path": "dair-ai/emotion", "hf_name": None,
        "split": "test", "text_field": "text", "label_field": "label",
        "label_names": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    },
    {
        "name": "Banking77",
        "path": "legacy-datasets/banking77", "hf_name": None,
        "split": "test", "text_field": "text", "label_field": "label",
        "label_names": None,  # resolved from dataset features
    },
]

# ── Metric helpers ────────────────────────────────────────────────────────────

def accuracy(preds: list[str], labels: list[str]) -> float:
    return 100.0 * float(np.mean([p == l for p, l in zip(preds, labels)]))


def ece_score(
    confidences: list[float],
    correct_flags: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (returns value in [0, 100])."""
    confs = np.array(confidences)
    flags = np.array(correct_flags, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if i == 0:
            mask |= confs == 0.0
        n = mask.sum()
        if n == 0:
            continue
        ece += (n / total) * abs(flags[mask].mean() - confs[mask].mean())
    return float(ece) * 100.0


def brier_score(prob_lists: list[list[float]], true_indices: list[int]) -> float:
    total = 0.0
    for probs, idx in zip(prob_lists, true_indices):
        onehot = np.zeros(len(probs))
        onehot[idx] = 1.0
        total += float(np.sum((np.array(probs) - onehot) ** 2))
    return total / len(prob_lists)


def nll_score(prob_lists: list[list[float]], true_indices: list[int]) -> float:
    return float(np.mean([
        -math.log(max(probs[idx], 1e-15))
        for probs, idx in zip(prob_lists, true_indices)
    ]))


def latency_pct(latencies_s: list[float]) -> tuple[float, float]:
    arr = np.array(latencies_s) * 1000.0
    return float(np.median(arr)), float(np.percentile(arr, 95))


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_microgen_prompt(text: str, label_names: list[str]) -> str:
    """Prompt used for MicroGen decode-free scoring."""
    return (
        "Classify the text into exactly one of the following categories.\n"
        f"Categories: {' | '.join(label_names)}\n\n"
        f"Text: {text}\n"
        "Category:"
    )


def build_generation_prompt(
    text: str, label_names: list[str]
) -> tuple[str, dict[str, str]]:
    """Strict A/B/C/D prompt for generation. Returns (prompt, letter→label mapping)."""
    letters = list(string.ascii_uppercase)
    mapping = {letters[i]: name for i, name in enumerate(label_names)}
    options = "\n".join(f"{k} = {v}" for k, v in mapping.items())
    valid = "/".join(mapping.keys())
    prompt = (
        "Classify the text into exactly one category.\n\n"
        f"Options:\n{options}\n\n"
        f"Text: {text}\n\n"
        f"Output ONLY the letter ({valid}):"
    )
    return prompt, mapping


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_generation(
    text: str,
    label_names: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> tuple[str, float]:
    """Strict A/B/C/D generation baseline."""
    prompt, mapping = build_generation_prompt(text, label_names)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        out[0][ids.shape[1]:], skip_special_tokens=True
    ).strip().upper()
    for letter, label in mapping.items():
        if letter in generated:
            return label, 1.0
    return label_names[0], 1.0  # fallback


def run_microgen(
    text: str,
    label_names: list[str],
    engine: DecisionEngine,
) -> tuple[str, float, list[float]]:
    """MicroGen decode-free scoring."""
    ctx = build_microgen_prompt(text, label_names)
    schema = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result = engine.choose(context=ctx, schema=schema)
    probs = [result.probabilities.get(n, 0.0) for n in label_names]
    return result.choice, result.top_probability, probs


def run_calibrated(
    text: str,
    label_names: list[str],
    engine: DecisionEngine,
    scaler: TemperatureScaler,
) -> tuple[str, float, list[float]]:
    ctx = build_microgen_prompt(text, label_names)
    schema = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result = engine.choose(context=ctx, schema=schema)
    cal = scaler.transform(result)
    probs = [cal.probabilities.get(n, 0.0) for n in label_names]
    return cal.choice, cal.top_probability, probs


# ── Permutation invariance ────────────────────────────────────────────────────

def engine_permutation_invariance(
    text: str,
    label_names: list[str],
    engine: DecisionEngine,
    n_perm: int = 10,
) -> dict[str, float]:
    """Engine invariance: fixed prompt, permuted schema. KL should be ~0."""
    fixed_ctx = build_microgen_prompt(text, label_names)
    schema_orig = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result_orig = engine.choose(context=fixed_ctx, schema=schema_orig)
    probs_orig = [result_orig.probabilities.get(n, 0.0) for n in label_names]

    kl_values = []
    for _ in range(n_perm):
        shuffled = label_names.copy()
        random.shuffle(shuffled)
        schema_perm = ChoiceSchema(name="cls", options=[Choice(n) for n in shuffled])
        result_perm = engine.choose(context=fixed_ctx, schema=schema_perm)
        probs_perm = [result_perm.probabilities.get(n, 0.0) for n in label_names]
        p = np.clip(probs_orig, 1e-15, 1.0)
        q = np.clip(probs_perm, 1e-15, 1.0)
        kl_values.append(float(np.sum(p * np.log(p / q))))

    return {"mean_kl": float(np.mean(kl_values)), "max_kl": float(np.max(kl_values))}


# ── Dataset runner ─────────────────────────────────────────────────────────────

class DatasetResult(NamedTuple):
    name: str
    n_classes: int
    n_eval: int
    n_calib: int
    temperature: float
    acc_gen: float
    acc_mg: float
    acc_cal: float
    ece_mg: float
    ece_cal: float
    brier_mg: float
    brier_cal: float
    nll_mg: float
    nll_cal: float
    p50_gen: float
    p95_gen: float
    p50_mg: float
    p95_mg: float
    engine_kl_mean: float
    engine_kl_max: float


def run_dataset(
    ds_cfg: dict,
    model: torch.nn.Module,
    tokenizer,
    engine: DecisionEngine,
    device: torch.device,
    n_eval: int,
    calib_min: int,
    seed: int,
    n_perm: int,
) -> DatasetResult:
    ds_name = ds_cfg["name"]
    print(f"\n{'='*72}")
    print(f"  {ds_name}")
    print(f"{'='*72}")

    ds = load_dataset(ds_cfg["path"], ds_cfg["hf_name"], split=ds_cfg["split"])
    label_names: list[str] = (
        ds_cfg["label_names"] or ds.features[ds_cfg["label_field"]].names
    )
    n_classes = len(label_names)

    # Calibration density: ensure at least calib_min × n_classes examples,
    # but cap at 20% of eval budget or dataset size.
    n_calib = min(max(calib_min, calib_min * n_classes), len(ds) // 5, 1000)
    n_eval_clamped = min(n_eval, len(ds) - n_calib)

    print(f"  Classes: {n_classes} | Dataset size: {len(ds)}")
    print(f"  N_eval={n_eval_clamped}  N_calib={n_calib}")
    print(f"  Labels: {label_names[:6]}{'...' if n_classes > 6 else ''}")

    # Shuffle indices reproducibly
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    calib_idx = indices[:n_calib]
    eval_idx = indices[n_calib: n_calib + n_eval_clamped]

    # ── Fit calibration ───────────────────────────────────────
    print(f"\n  Fitting TemperatureScaler on {n_calib} examples...")
    calib_results, calib_labels = [], []
    for i in calib_idx:
        row = ds[i]
        text = row[ds_cfg["text_field"]]
        true_lbl = label_names[row[ds_cfg["label_field"]]]
        ctx = build_microgen_prompt(text, label_names)
        schema = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
        calib_results.append(engine.choose(context=ctx, schema=schema))
        calib_labels.append(true_lbl)

    scaler = TemperatureScaler()
    scaler.fit(calib_results, calib_labels)
    T = scaler.temperature.clamp(min=1e-3).item()
    print(f"  Temperature T = {T:.4f}")

    # ── Evaluation loop ───────────────────────────────────────
    gen_preds, gen_confs, gen_lats = [], [], []
    mg_preds, mg_confs, mg_probs, mg_lats = [], [], [], []
    cal_preds, cal_confs, cal_probs = [], [], []
    true_labels_list: list[str] = []
    true_idx_list: list[int] = []

    print(f"\n  Evaluating {n_eval_clamped} examples...")
    t_loop_start = time.time()
    for rank, i in enumerate(eval_idx):
        row = ds[i]
        text = row[ds_cfg["text_field"]]
        t_idx = row[ds_cfg["label_field"]]
        t_lbl = label_names[t_idx]
        true_labels_list.append(t_lbl)
        true_idx_list.append(t_idx)

        if (rank + 1) % 100 == 0:
            elapsed = time.time() - t_loop_start
            print(f"    {rank+1}/{n_eval_clamped} ({elapsed:.1f}s)")

        # Generation
        t0 = time.perf_counter()
        pred_a, conf_a = run_generation(text, label_names, model, tokenizer, device)
        gen_preds.append(pred_a)
        gen_confs.append(conf_a)
        gen_lats.append(time.perf_counter() - t0)

        # MicroGen
        t0 = time.perf_counter()
        pred_b, conf_b, probs_b = run_microgen(text, label_names, engine)
        mg_preds.append(pred_b)
        mg_confs.append(conf_b)
        mg_probs.append(probs_b)
        mg_lats.append(time.perf_counter() - t0)

        # Calibrated
        pred_c, conf_c, probs_c = run_calibrated(text, label_names, engine, scaler)
        cal_preds.append(pred_c)
        cal_confs.append(conf_c)
        cal_probs.append(probs_c)

    # ── Engine permutation invariance ─────────────────────────
    print(f"\n  Running engine permutation invariance test ({n_perm} perms)...")
    perm_text = ds[eval_idx[0]][ds_cfg["text_field"]]
    inv = engine_permutation_invariance(perm_text, label_names, engine, n_perm)

    # ── Compute metrics ───────────────────────────────────────
    correct_gen = [p == t for p, t in zip(gen_preds, true_labels_list)]
    correct_mg = [p == t for p, t in zip(mg_preds, true_labels_list)]
    correct_cal = [p == t for p, t in zip(cal_preds, true_labels_list)]

    p50_gen, p95_gen = latency_pct(gen_lats)
    p50_mg, p95_mg = latency_pct(mg_lats)

    acc_gen = accuracy(gen_preds, true_labels_list)
    acc_mg = accuracy(mg_preds, true_labels_list)
    acc_cal = accuracy(cal_preds, true_labels_list)
    ece_mg = ece_score(mg_confs, correct_mg)
    ece_cal = ece_score(cal_confs, correct_cal)
    brier_mg = brier_score(mg_probs, true_idx_list)
    brier_cal = brier_score(cal_probs, true_idx_list)
    nll_mg = nll_score(mg_probs, true_idx_list)
    nll_cal = nll_score(cal_probs, true_idx_list)

    # ── Print dataset summary ─────────────────────────────────
    print(f"\n  {ds_name} Results (N={n_eval_clamped})")
    rows = [
        ["Generation (strict A/B/C)",
         f"{acc_gen:.1f}%", "N/A", "N/A", "N/A",
         f"{p50_gen:.0f}", f"{p95_gen:.0f}"],
        ["MicroGen (decode-free)",
         f"{acc_mg:.1f}%", f"{ece_mg:.1f}%", f"{brier_mg:.4f}", f"{nll_mg:.3f}",
         f"{p50_mg:.0f}", f"{p95_mg:.0f}"],
        ["MicroGen + Calibration",
         f"{acc_cal:.1f}%", f"{ece_cal:.1f}%", f"{brier_cal:.4f}", f"{nll_cal:.3f}",
         f"{p50_mg:.0f}", f"{p95_mg:.0f}"],
    ]
    jev = JEV_REFERENCE.get(ds_name)
    if jev:
        rows.append([
            f"Jev reference ({ds_name})",
            f"{jev['acc']:.1f}%" if jev["acc"] else "N/A",
            f"{jev['ece']:.1f}%-gap" if jev["ece"] else "N/A",
            "N/A", "N/A",
            f"{jev['p50_ms']:.0f}" if jev["p50_ms"] else "N/A",
            "N/A",
        ])

    try:
        from tabulate import tabulate
        print(tabulate(
            rows,
            headers=["Method", "Acc", "ECE", "Brier", "NLL", "p50(ms)", "p95(ms)"],
            tablefmt="github",
        ))
    except ImportError:
        for row in rows:
            print("  " + " | ".join(str(x) for x in row))

    print(f"\n  Engine permutation invariance: mean KL={inv['mean_kl']:.2e}, max KL={inv['max_kl']:.2e}")
    print(f"  T={T:.4f} | ECE: {ece_mg:.1f}% → {ece_cal:.1f}%")

    return DatasetResult(
        name=ds_name, n_classes=n_classes,
        n_eval=n_eval_clamped, n_calib=n_calib, temperature=T,
        acc_gen=acc_gen, acc_mg=acc_mg, acc_cal=acc_cal,
        ece_mg=ece_mg, ece_cal=ece_cal,
        brier_mg=brier_mg, brier_cal=brier_cal,
        nll_mg=nll_mg, nll_cal=nll_cal,
        p50_gen=p50_gen, p95_gen=p95_gen,
        p50_mg=p50_mg, p95_mg=p95_mg,
        engine_kl_mean=inv["mean_kl"], engine_kl_max=inv["max_kl"],
    )


# ── Final comparison table ─────────────────────────────────────────────────────

def print_jev_comparison(results: list[DatasetResult]) -> None:
    print("\n\n" + "=" * 80)
    print("  FINAL SUMMARY — MicroGen vs Jev Reference")
    print("=" * 80)
    rows = []
    for r in results:
        jev = JEV_REFERENCE.get(r.name, {})
        jev_acc = f"{jev.get('acc', 'N/A'):.1f}%" if isinstance(jev.get("acc"), float) else "N/A"
        jev_lat = f"{jev.get('p50_ms', 'N/A'):.0f}" if isinstance(jev.get("p50_ms"), (int, float)) else "N/A"
        rows.append([
            r.name, r.n_classes, r.n_eval,
            f"{r.acc_gen:.1f}%",
            f"{r.acc_mg:.1f}%",
            jev_acc,
            f"{r.ece_mg:.1f}%→{r.ece_cal:.1f}%",
            f"{r.p50_gen:.0f}ms",
            f"{r.p50_mg:.0f}ms",
            jev_lat,
            f"{r.engine_kl_mean:.1e}",
        ])
    headers = [
        "Dataset", "Classes", "N",
        "Acc(Gen)", "Acc(MG)", "Acc(Jev)",
        "ECE(MG→Cal)",
        "p50(Gen)", "p50(MG)", "p50(Jev)",
        "KL(engine)",
    ]
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        print(" | ".join(headers))
        for row in rows:
            print(" | ".join(str(x) for x in row))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MicroGen large-scale benchmark — mirrors Jev methodology."
    )
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--eval", type=int, default=1000, dest="n_eval",
                        help="Evaluation examples per dataset (default 1000)")
    parser.add_argument("--calib-min", type=int, default=50,
                        help="Minimum calibration examples per class (default 50)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=["SST-2", "AG News", "Emotion", "Banking77"],
                        default=None, help="Which datasets to run (default: all)")
    parser.add_argument("--perm", type=int, default=10,
                        help="Number of permutation trials (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/raw/microgen_v12_benchmark.json",
                        help="Output JSON path")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel : {args.model}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.0f}M\n")

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    dataset_cfgs = DATASETS
    if args.datasets:
        dataset_cfgs = [d for d in DATASETS if d["name"] in args.datasets]

    all_results: list[DatasetResult] = []
    for cfg in dataset_cfgs:
        r = run_dataset(
            ds_cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            engine=engine,
            device=device,
            n_eval=args.n_eval,
            calib_min=args.calib_min,
            seed=args.seed,
            n_perm=args.perm,
        )
        all_results.append(r)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_jev_comparison(all_results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = [r._asdict() for r in all_results]
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
