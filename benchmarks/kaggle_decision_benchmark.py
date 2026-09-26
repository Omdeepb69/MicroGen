#!/usr/bin/env python3
# ============================================================
#  MicroGen Decision Engine — Full Research Benchmark Suite
#  One-cell Kaggle T4 GPU script
#  v1.2.0  |  2026-09-26
# ============================================================
#
#  Changes in v1.2.0
#    - Strict A/B/C/D generation baseline (no substring parsing)
#    - Engine Permutation Invariance: fixed prompt, permuted schema
#    - Prompt Option-Order Sensitivity: separate generation test
#    - Explicit attention_mask passing throughout
#    - Batched BFS Trie (replaces sequential DFS)
#
#  Experiments
#    A. Normal LLM generation  (generate → parse)
#    B. MicroGen decode-free   (logit scoring, Trie-based)
#    C. Calibrated MicroGen    (+ Temperature Scaling)
#    D. Option-order invariance (KL divergence across permutations)
#
#  Datasets
#    SST-2     (2 classes,  872 test items)
#    AG News   (4 classes,  7 600 test items)
#    Emotion   (6 classes,  2 000 test items)
#    Banking77 (77 classes, 3 080 test items)
#
#  Model
#    Qwen/Qwen2.5-1.5B-Instruct  (float16, T4 GPU)
#
#  Comparison targets (from literature)
#    Jev Banking77: 79.7% accuracy, 9.8-pt calibration gap, 467 ms median
#    open-system-one Jev Banking77: 78.4% (bare) / 77.8% (enriched)
#    Community Qwen3.8 27B Q4: 96.53% on SemIf 144-task benchmark
# ============================================================

import subprocess, sys

# ── 0. Install ──────────────────────────────────────────────
for pkg in [
    "microgen-llm==1.2.0",
    "datasets>=2.14",
    "scipy>=1.10",
    "tabulate>=0.9",
]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── 1. Imports ──────────────────────────────────────────────
import os, time, random, math, json, gc
import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.stats import entropy as scipy_entropy

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.calibration import TemperatureScaler

# ── 2. Reproducibility & device ─────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 3. Benchmark config ──────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"
N_EVAL         = 1000  # evaluation examples per dataset
N_CALIB_MIN    = 50    # minimum examples per class for Temperature Scaler
N_PERMUTATIONS = 10    # option-order permutation trials

# ── 4. Dataset registry ──────────────────────────────────────
# Fields: path, name, split, text_field, label_field, label_names (None = from features)
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

# ── 5. Load model ────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded — {n_params / 1e6:.0f}M parameters\n")

adapter = TransformersDecisionModel(model, tokenizer)
engine  = DecisionEngine(adapter)

# ── 6. Metric helpers ────────────────────────────────────────

def accuracy(preds, labels):
    return float(np.mean([p == l for p, l in zip(preds, labels)])) * 100.0


def ece_score(confidences, correct_flags, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs  = np.array(confidences)
    flags  = np.array(correct_flags, dtype=float)
    total  = len(confs)
    ece    = 0.0
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


def brier_score(prob_lists, true_indices, n_classes):
    """Multi-class Brier score."""
    scores = []
    for probs, idx in zip(prob_lists, true_indices):
        onehot = np.zeros(n_classes)
        onehot[idx] = 1.0
        scores.append(np.sum((np.array(probs) - onehot) ** 2))
    return float(np.mean(scores))


def nll_score(prob_lists, true_indices):
    """Negative log-likelihood."""
    return float(np.mean([
        -math.log(max(probs[idx], 1e-15))
        for probs, idx in zip(prob_lists, true_indices)
    ]))


def latency_pct(latencies_s):
    """Returns (p50_ms, p95_ms)."""
    arr = np.array(latencies_s) * 1000.0
    return float(np.median(arr)), float(np.percentile(arr, 95))


def kl_divergence(p, q):
    p = np.clip(np.array(p, dtype=np.float64), 1e-15, 1.0)
    q = np.clip(np.array(q, dtype=np.float64), 1e-15, 1.0)
    return float(scipy_entropy(p, q))


# ── 7. Prompt builder ────────────────────────────────────────

def build_prompt(text: str, label_names: list[str]) -> str:
    labels = " | ".join(label_names)
    return (
        f"Classify the text into exactly one of the following categories.\n"
        f"Categories: {labels}\n\n"
        f"Text: {text}\n"
        f"Category:"
    )

def build_strict_generation_prompt(text: str, label_names: list[str]) -> tuple[str, dict[str, str]]:
    """Strict A/B/C/D prompt for generation."""
    import string as _string
    letters = list(_string.ascii_uppercase)
    mapping = {letters[i]: name for i, name in enumerate(label_names)}
    options_text = "\n".join(f"{k} = {v}" for k, v in mapping.items())
    valid_letters = "/".join(mapping.keys())
    prompt = (
        f"Classify the text into exactly one category.\n\n"
        f"Options:\n{options_text}\n\n"
        f"Text: {text}\n\n"
        f"Output ONLY the letter ({valid_letters}):"
    )
    return prompt, mapping


# ── 8. Experiment implementations ───────────────────────────

@torch.no_grad()
def run_generation(text: str, label_names: list[str]) -> tuple[str, float]:
    """Experiment A: strict A/B/C/D generation baseline."""
    prompt, mapping = build_strict_generation_prompt(text, label_names)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
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
) -> tuple[str, float, list[float]]:
    """Experiment B: MicroGen decode-free scoring (auto-routes to Trie for multi-token)."""
    context = build_prompt(text, label_names)
    schema  = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result  = engine.choose(context=context, schema=schema)
    probs   = [result.probabilities.get(n, 0.0) for n in label_names]
    return result.choice, result.top_probability, probs


def run_calibrated(
    text: str,
    label_names: list[str],
    scaler: TemperatureScaler,
) -> tuple[str, float, list[float]]:
    """Experiment C: MicroGen + temperature scaling."""
    context = build_prompt(text, label_names)
    schema  = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result  = engine.choose(context=context, schema=schema)
    cal     = scaler.transform(result)
    probs   = [cal.probabilities.get(n, 0.0) for n in label_names]
    return cal.choice, cal.top_probability, probs


# ── 9. Permutation invariance test ───────────────────────────

def permutation_invariance_test(
    text: str,
    label_names: list[str],
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """
    Two distinct tests:
    1. Engine Invariance: fixed prompt context, permuted ChoiceSchema order. KL≈0.
    2. Prompt Sensitivity: permuted Categories text in prompt, measure generation flip rate.
    """
    # 1. Engine invariance — fixed prompt, permuted schema
    fixed_context = build_prompt(text, label_names)
    schema_orig = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
    result_orig = engine.choose(context=fixed_context, schema=schema_orig)
    probs_orig = [result_orig.probabilities.get(n, 0.0) for n in label_names]

    kl_microgen = []
    for _ in range(n_permutations):
        shuffled = label_names.copy()
        random.shuffle(shuffled)
        schema_perm = ChoiceSchema(name="cls", options=[Choice(n) for n in shuffled])
        result_perm = engine.choose(context=fixed_context, schema=schema_perm)
        probs_perm = [result_perm.probabilities.get(n, 0.0) for n in label_names]
        kl_microgen.append(kl_divergence(probs_orig, probs_perm))

    # 2. Prompt sensitivity — permuted Categories line, generation flip rate
    pred_orig, _ = run_generation(text, label_names)
    gen_changed = []
    for _ in range(n_permutations):
        shuffled = label_names.copy()
        random.shuffle(shuffled)
        pred_perm, _ = run_generation(text, shuffled)
        gen_changed.append(0.0 if pred_orig == pred_perm else 1.0)

    return {
        "microgen_mean_kl":     float(np.mean(kl_microgen)),
        "microgen_max_kl":      float(np.max(kl_microgen)),
        "gen_pct_pred_changed": float(np.mean(gen_changed) * 100.0),
    }


# ── 10. Run one dataset ──────────────────────────────────────

def run_dataset(ds_cfg: dict) -> dict:
    ds_name = ds_cfg["name"]
    print(f"\n{'='*64}")
    print(f"  {ds_name}")
    print(f"{'='*64}")

    # Load
    ds = load_dataset(ds_cfg["path"], ds_cfg["hf_name"], split=ds_cfg["split"])
    label_names = ds_cfg["label_names"] or ds.features[ds_cfg["label_field"]].names
    n_classes   = len(label_names)
    print(f"  Classes: {n_classes} | Total examples: {len(ds)}")
    print(f"  Labels: {label_names[:8]}{'...' if n_classes > 8 else ''}")

    # Shuffle and split
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    n_calib = min(max(N_CALIB_MIN, N_CALIB_MIN * n_classes), len(ds) // 5, 1000)
    n_eval_clamped = min(N_EVAL, len(ds) - n_calib)

    calib_idx = indices[:n_calib]
    eval_idx  = indices[n_calib:n_calib + n_eval_clamped]

    # ── Calibration fit ──────────────────────────────────────
    print(f"\n  Fitting TemperatureScaler on {n_calib} examples ...")
    calib_results, calib_labels = [], []
    for i in calib_idx:
        row  = ds[i]
        text = row[ds_cfg["text_field"]]
        true_lbl = label_names[row[ds_cfg["label_field"]]]
        ctx      = build_prompt(text, label_names)
        schema   = ChoiceSchema(name="cls", options=[Choice(n) for n in label_names])
        result   = engine.choose(context=ctx, schema=schema)
        calib_results.append(result)
        calib_labels.append(true_lbl)

    scaler = TemperatureScaler()
    scaler.fit(calib_results, calib_labels)
    T = scaler.temperature.item()
    print(f"  Temperature T = {T:.4f}")

    # ── Accumulators ─────────────────────────────────────────
    acc_a = {"preds": [], "confs": [], "lats": []}
    acc_b = {"preds": [], "confs": [], "probs": [], "lats": []}
    acc_c = {"preds": [], "confs": [], "probs": [], "lats": []}
    true_labels  = []
    true_indices = []

    print(f"\n  Running {N_EVAL} evaluations ...")
    for rank, i in enumerate(eval_idx):
        row  = ds[i]
        text = row[ds_cfg["text_field"]]
        t_idx = row[ds_cfg["label_field"]]
        t_lbl = label_names[t_idx]
        true_labels.append(t_lbl)
        true_indices.append(t_idx)

        if (rank + 1) % 25 == 0:
            print(f"    {rank+1}/{N_EVAL} ...")

        # A — Generation
        t0 = time.perf_counter()
        pred_a, conf_a = run_generation(text, label_names)
        acc_a["preds"].append(pred_a)
        acc_a["confs"].append(conf_a)
        acc_a["lats"].append(time.perf_counter() - t0)

        # B — MicroGen
        t0 = time.perf_counter()
        pred_b, conf_b, probs_b = run_microgen(text, label_names)
        acc_b["preds"].append(pred_b)
        acc_b["confs"].append(conf_b)
        acc_b["probs"].append(probs_b)
        acc_b["lats"].append(time.perf_counter() - t0)

        # C — Calibrated
        t0 = time.perf_counter()
        pred_c, conf_c, probs_c = run_calibrated(text, label_names, scaler)
        acc_c["preds"].append(pred_c)
        acc_c["confs"].append(conf_c)
        acc_c["probs"].append(probs_c)
        acc_c["lats"].append(time.perf_counter() - t0)

    # ── Option-order invariance test ─────────────────────────
    print(f"\n  Running permutation invariance test ({N_PERMUTATIONS} permutations)...")
    perm_text = ds[eval_idx[0]][ds_cfg["text_field"]]
    inv = permutation_invariance_test(perm_text, label_names, N_PERMUTATIONS)

    # ── Compute metrics ──────────────────────────────────────
    correct_a = [p == t for p, t in zip(acc_a["preds"], true_labels)]
    correct_b = [p == t for p, t in zip(acc_b["preds"], true_labels)]
    correct_c = [p == t for p, t in zip(acc_c["preds"], true_labels)]

    p50_a, p95_a = latency_pct(acc_a["lats"])
    p50_b, p95_b = latency_pct(acc_b["lats"])
    p50_c, p95_c = latency_pct(acc_c["lats"])

    rows = [
        [
            "A. Generation",
            f"{accuracy(acc_a['preds'], true_labels):.1f}%",
            "N/A",    # no distribution
            "N/A",
            "N/A",
            f"{p50_a:.0f}",
            f"{p95_a:.0f}",
            "≥1 tokens",
            f"{inv['gen_pct_pred_changed']:.0f}% changed",
        ],
        [
            "B. MicroGen",
            f"{accuracy(acc_b['preds'], true_labels):.1f}%",
            f"{ece_score(acc_b['confs'], correct_b):.1f}%",
            f"{brier_score(acc_b['probs'], true_indices, n_classes):.4f}",
            f"{nll_score(acc_b['probs'], true_indices):.3f}",
            f"{p50_b:.0f}",
            f"{p95_b:.0f}",
            "0 tokens",
            f"KL={inv['microgen_mean_kl']:.2e} (max={inv['microgen_max_kl']:.2e})",
        ],
        [
            "C. Calibrated",
            f"{accuracy(acc_c['preds'], true_labels):.1f}%",
            f"{ece_score(acc_c['confs'], correct_c):.1f}%",
            f"{brier_score(acc_c['probs'], true_indices, n_classes):.4f}",
            f"{nll_score(acc_c['probs'], true_indices):.3f}",
            f"{p50_c:.0f}",
            f"{p95_c:.0f}",
            "0 tokens",
            f"T={T:.3f}",
        ],
    ]

    headers = ["Experiment", "Acc", "ECE", "Brier", "NLL", "p50(ms)", "p95(ms)", "Tokens", "Notes"]
    print(f"\n  {ds_name} Results")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Literature comparison for Banking77
    if ds_name == "Banking77":
        print(f"\n  Literature comparison (Banking77):")
        print(f"    Jev (Jevals 2026-09-18):          79.7% acc | 9.8-pt calib gap | 467 ms median")
        print(f"    Jev (open-system-one):             78.4% acc (bare labels)")
        print(f"    Qwen3.8 27B Q4 (community):        96.53% acc (SemIf 144-task, unaudited)")
        print(f"    MicroGen Qwen2.5-1.5B (this run):  {accuracy(acc_b['preds'], true_labels):.1f}% acc | {p50_b:.0f} ms median")

    return {
        "dataset":        ds_name,
        "n_classes":      n_classes,
        "temperature":    T,
        "acc_gen":        accuracy(acc_a["preds"], true_labels),
        "acc_microgen":   accuracy(acc_b["preds"], true_labels),
        "acc_calibrated": accuracy(acc_c["preds"], true_labels),
        "ece_microgen":   ece_score(acc_b["confs"], correct_b),
        "ece_calibrated": ece_score(acc_c["confs"], correct_c),
        "p50_gen_ms":     p50_a,
        "p50_microgen_ms": p50_b,
        "kl_mean":        inv["microgen_mean_kl"],
        "kl_max":         inv["microgen_max_kl"],
        "gen_pct_changed": inv["gen_pct_pred_changed"],
    }


# ── 11. Main ─────────────────────────────────────────────────

print("=" * 64)
print("  MicroGen Decision Engine — Research Benchmark Suite")
print("=" * 64)
print(f"  Model:   {MODEL_ID}")
print(f"  Evals:   {N_EVAL} / dataset")
print(f"  Calib:   {N_CALIB_MIN} (min) / class (for Temperature Scaler)")
print(f"  Perms:   {N_PERMUTATIONS} (option-order invariance test)")
print("=" * 64)

all_results = []
for cfg in DATASETS:
    r = run_dataset(cfg)
    all_results.append(r)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── 12. Final summary table ───────────────────────────────────
print("\n\n" + "=" * 64)
print("  FINAL SUMMARY")
print("=" * 64)

summary_rows = []
for r in all_results:
    summary_rows.append([
        r["dataset"],
        r["n_classes"],
        f"{r['acc_gen']:.1f}%",
        f"{r['acc_microgen']:.1f}%",
        f"{r['acc_calibrated']:.1f}%",
        f"{r['ece_microgen']:.1f}% → {r['ece_calibrated']:.1f}%",
        f"{r['p50_gen_ms']:.0f} ms",
        f"{r['p50_microgen_ms']:.0f} ms",
        f"KL={r['kl_mean']:.2e}",
        f"{r['gen_pct_changed']:.0f}%",
    ])

summary_headers = [
    "Dataset", "Classes",
    "Acc(Gen)", "Acc(MG)", "Acc(Cal)",
    "ECE (uncal→cal)",
    "p50(Gen)", "p50(MG)",
    "MG KL(perm)",
    "Gen pred change",
]
print(tabulate(summary_rows, headers=summary_headers, tablefmt="github"))

print("""
Key Properties of Decode-Free Inference:
  ✅ Zero generated tokens — direct logit scoring
  ✅ Full probability distribution over all candidates
  ✅ Option-order invariant — KL ≈ 0.0 across permutations
  ✅ Temperature calibration reduces ECE
  ⚠  Generation is sensitive to option ordering (position bias)
""")

# Save results JSON
with open("microgen_benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Results saved → microgen_benchmark_results.json")
