"""Calibration Scaling Benchmark.

Evaluates how the number of calibration examples affects Temperature Scaling
quality across ECE, NLL, and Brier score. Runs BoolQ at a single large
evaluation budget, then calibrates using progressively larger subsets to show
how calibration quality scales with sample count.

Task 6.3 scope: N_calib ∈ {50, 100, 250, 500, 1000}.
"""

import argparse
import math
import time
from typing import NamedTuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.calibration import TemperatureScaler
from microgen.decision.engine import DecisionEngine
from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.schema import DecisionResult


# ── Calibration metrics ──────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: list[float],
    accuracies: list[float],
    num_bins: int = 10,
) -> float:
    """ECE using equal-width bins."""
    bins = np.linspace(0, 1, num_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [i for i, c in enumerate(confidences) if lo < c <= hi]
        if not mask:
            continue
        bin_conf = np.mean([confidences[i] for i in mask])
        bin_acc = np.mean([accuracies[i] for i in mask])
        ece += (len(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(
    prob_lists: list[list[float]],
    true_indices: list[int],
) -> float:
    """Multiclass Brier score: mean squared error between prob vector and one-hot."""
    total = 0.0
    for probs, idx in zip(prob_lists, true_indices):
        onehot = np.zeros(len(probs))
        onehot[idx] = 1.0
        total += float(np.sum((np.array(probs) - onehot) ** 2))
    return total / len(prob_lists)


def nll_score(
    prob_lists: list[list[float]],
    true_indices: list[int],
) -> float:
    """Negative log-likelihood."""
    return float(np.mean([
        -math.log(max(probs[idx], 1e-15))
        for probs, idx in zip(prob_lists, true_indices)
    ]))


# ── Result container ─────────────────────────────────────────────────────────

class CalibMetrics(NamedTuple):
    n_calib: int
    temperature: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float


# ── Main benchmark ────────────────────────────────────────────────────────────

def run(
    n_eval: int = 1000,
    calib_sizes: list[int] | None = None,
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    seed: int = 42,
) -> None:
    if calib_sizes is None:
        calib_sizes = [50, 100, 250, 500, 1000]

    # Clamp calib sizes to n_eval
    calib_sizes = [n for n in calib_sizes if n <= n_eval]

    print(f"Model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    print("Downloading BoolQ validation split...")
    dataset = load_dataset("google/boolq", split="validation")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_eval, len(dataset))))

    print(f"\nEvaluating {len(dataset)} items...")
    results_raw: list[DecisionResult] = []
    true_labels: list[str] = []
    t0 = time.time()

    for i, item in enumerate(dataset):
        label = "YES" if item["answer"] else "NO"
        true_labels.append(label)
        result = engine.yes_no(
            context=f"Passage: {item['passage']}",
            question=item["question"],
        )
        results_raw.append(result)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dataset)} ({time.time()-t0:.1f}s)")

    print(f"Inference complete in {time.time()-t0:.1f}s\n")

    # Build raw metric arrays once — calibration only changes the temperature
    label_list = ["YES", "NO"]
    uncal_conf = [r.top_probability for r in results_raw]
    uncal_acc = [1.0 if r.choice == t else 0.0 for r, t in zip(results_raw, true_labels)]
    uncal_probs = [
        [r.probabilities.get("YES", 0.0), r.probabilities.get("NO", 0.0)]
        for r in results_raw
    ]
    true_indices = [label_list.index(t) for t in true_labels]

    uncal_ece = expected_calibration_error(uncal_conf, uncal_acc)
    uncal_brier = brier_score(uncal_probs, true_indices)
    uncal_nll = nll_score(uncal_probs, true_indices)

    print(f"Uncalibrated (N={len(results_raw)} eval):")
    print(f"  ECE:   {uncal_ece * 100:.2f}%")
    print(f"  Brier: {uncal_brier:.4f}")
    print(f"  NLL:   {uncal_nll:.4f}")
    print(f"  Acc:   {np.mean(uncal_acc)*100:.2f}%")

    # ── Sweep calibration sizes ───────────────────────────────────────────────
    all_metrics: list[CalibMetrics] = []

    for n_calib in calib_sizes:
        calib_results = results_raw[:n_calib]
        calib_labels = true_labels[:n_calib]

        scaler = TemperatureScaler()
        scaler.fit(calib_results, calib_labels)
        t_fitted = scaler.temperature.clamp(min=1e-3).item()

        # Apply calibration to the full eval set
        cal_conf = []
        cal_acc = []
        cal_probs = []

        for res, label in zip(results_raw, true_labels):
            cal_res = scaler.transform(res)
            cal_conf.append(cal_res.top_probability)
            cal_acc.append(1.0 if cal_res.choice == label else 0.0)
            cal_probs.append([
                cal_res.probabilities.get("YES", 0.0),
                cal_res.probabilities.get("NO", 0.0),
            ])

        cal_ece = expected_calibration_error(cal_conf, cal_acc)
        cal_brier = brier_score(cal_probs, true_indices)
        cal_nll = nll_score(cal_probs, true_indices)

        m = CalibMetrics(
            n_calib=n_calib,
            temperature=t_fitted,
            ece_before=uncal_ece,
            ece_after=cal_ece,
            brier_before=uncal_brier,
            brier_after=cal_brier,
            nll_before=uncal_nll,
            nll_after=cal_nll,
        )
        all_metrics.append(m)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("Calibration Scaling Results")
    print("=" * 82)
    header = (
        f"{'N_calib':>8} | {'Temp':>6} | "
        f"{'ECE(↓)':>14} | {'Brier(↓)':>14} | {'NLL(↓)':>14}"
    )
    print(header)
    print("-" * 82)

    # Uncalibrated baseline row
    print(
        f"{'Uncal':>8} | {'1.000':>6} | "
        f"{'before':>7}{'after':>7} | "
        f"{'before':>7}{'after':>7} | "
        f"{'before':>7}{'after':>7}"
    )
    print(
        f"{'':>8} | {'':>6} | "
        f"{uncal_ece*100:>6.2f}%{'→':>1} | "
        f"{uncal_brier:>6.4f}{'→':>1} | "
        f"{uncal_nll:>6.4f}{'→':>1}"
    )
    print("-" * 82)

    for m in all_metrics:
        ece_delta = "↓" if m.ece_after < m.ece_before else "↑"
        brier_delta = "↓" if m.brier_after < m.brier_before else "↑"
        nll_delta = "↓" if m.nll_after < m.nll_before else "↑"
        print(
            f"{m.n_calib:>8} | {m.temperature:>6.3f} | "
            f"{m.ece_before*100:>5.2f}%→{m.ece_after*100:>5.2f}%{ece_delta} | "
            f"{m.brier_before:>6.4f}→{m.brier_after:>6.4f}{brier_delta} | "
            f"{m.nll_before:>6.4f}→{m.nll_after:>6.4f}{nll_delta}"
        )

    print("=" * 82)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_ece = min(all_metrics, key=lambda m: m.ece_after)
    print(f"\nBest ECE achieved at N_calib={best_ece.n_calib}: "
          f"{best_ece.ece_before*100:.2f}% → {best_ece.ece_after*100:.2f}% "
          f"(T={best_ece.temperature:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration scaling benchmark.")
    parser.add_argument("--eval", type=int, default=1000, dest="n_eval",
                        help="Total evaluation examples (default: 1000)")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[50, 100, 250, 500, 1000],
                        help="Calibration sample sizes to sweep (default: 50 100 250 500 1000)")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="HuggingFace model ID")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n_eval=args.n_eval, calib_sizes=args.sizes, model_name=args.model, seed=args.seed)
