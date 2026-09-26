#!/usr/bin/env python3
# ============================================================
#  MicroGen Decision Engine — Task 7.2
#  Calibration-Set Size Curve
# ============================================================
#
# Investigates how Temperature Scaling behaves as a function of
# calibration-set size, specifically targeting the instability
# observed on the 77-class Banking77 dataset.
#
# Run this via Kaggle T4 or locally.

import time
import json
import math
import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.calibration import TemperatureScaler

def accuracy(preds, labels):
    return float(np.mean([p == l for p, l in zip(preds, labels)])) * 100.0

def ece_score(confidences, correct_flags, n_bins=10):
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
    scores = []
    for probs, idx in zip(prob_lists, true_indices):
        onehot = np.zeros(n_classes)
        onehot[idx] = 1.0
        scores.append(np.sum((np.array(probs) - onehot) ** 2))
    return float(np.mean(scores))

def nll_score(prob_lists, true_indices):
    return float(np.mean([
        -math.log(max(probs[idx], 1e-15))
        for probs, idx in zip(prob_lists, true_indices)
    ]))

def build_prompt(text: str, label_names: list[str]) -> str:
    labels = " | ".join(label_names)
    return (
        f"Classify the text into exactly one of the following categories.\n"
        f"Categories: {labels}\n\n"
        f"Text: {text}\n"
        f"Category:"
    )

def main():
    # ── 1. Environment & Setup ──────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE.type == "cuda" else "HuggingFaceTB/SmolLM-135M"
    
    # Fast parameters for CPU testing, large parameters for GPU paper-run
    if DEVICE.type == "cuda":
        N_EVAL = 1000
        CALIB_SIZES = [50, 100, 250, 500, 1000, 2000]
    else:
        N_EVAL = 100
        CALIB_SIZES = [10, 20, 50, 100]

    print("=" * 70)
    print("  Task 7.2: Calibration-Set Size Curve")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL_ID}")
    print(f"  Eval:   {N_EVAL} test items")
    print(f"  Calib:  {CALIB_SIZES}")
    print("=" * 70)

    # ── 2. Load Model ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    if DEVICE.type == "cpu":
        model.to(DEVICE)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    # ── 3. Dataset Setup (Banking77) ────────────────────────────
    print("\nLoading Banking77 ...")
    ds_train = load_dataset("legacy-datasets/banking77", split="train")
    ds_test = load_dataset("legacy-datasets/banking77", split="test")
    
    label_names = ds_train.features["label"].names
    n_classes = len(label_names)
    schema = ChoiceSchema(name="banking77", options=[Choice(n) for n in label_names])
    
    # Shuffle splits
    np.random.seed(42)
    train_idx = np.random.permutation(len(ds_train))
    test_idx = np.random.permutation(len(ds_test))

    max_calib = max(CALIB_SIZES)
    calib_pool_idx = train_idx[:max_calib]
    eval_pool_idx = test_idx[:N_EVAL]

    # ── 4. Precompute Engine Results ────────────────────────────
    print(f"\nPrecomputing {max_calib} calibration outputs ...")
    calib_results = []
    calib_labels = []
    for rank, idx in enumerate(calib_pool_idx):
        if (rank + 1) % 50 == 0:
            print(f"  {rank+1}/{max_calib} ...")
        text = ds_train[int(idx)]["text"]
        label = label_names[ds_train[int(idx)]["label"]]
        ctx = build_prompt(text, label_names)
        with torch.no_grad():
            res = engine.choose(context=ctx, schema=schema)
        calib_results.append(res)
        calib_labels.append(label)

    print(f"\nPrecomputing {N_EVAL} evaluation outputs ...")
    eval_results = []
    eval_true_labels = []
    eval_true_indices = []
    for rank, idx in enumerate(eval_pool_idx):
        if (rank + 1) % 50 == 0:
            print(f"  {rank+1}/{N_EVAL} ...")
        text = ds_test[int(idx)]["text"]
        l_idx = ds_test[int(idx)]["label"]
        label = label_names[l_idx]
        ctx = build_prompt(text, label_names)
        with torch.no_grad():
            res = engine.choose(context=ctx, schema=schema)
        eval_results.append(res)
        eval_true_labels.append(label)
        eval_true_indices.append(l_idx)

    # ── 5. Run Calibration Curve ────────────────────────────────
    print("\nRunning Calibration-Set Size Curve ...")
    
    # Baseline Uncalibrated
    uncal_preds = [r.choice for r in eval_results]
    uncal_confs = [r.top_probability for r in eval_results]
    uncal_probs = [[r.probabilities.get(n, 0.0) for n in label_names] for r in eval_results]
    uncal_corr  = [p == t for p, t in zip(uncal_preds, eval_true_labels)]
    
    acc_uncal = accuracy(uncal_preds, eval_true_labels)
    ece_uncal = ece_score(uncal_confs, uncal_corr)
    brier_uncal = brier_score(uncal_probs, eval_true_indices, n_classes)
    nll_uncal = nll_score(uncal_probs, eval_true_indices)

    table_data = [
        ["Uncalibrated", "1.000", f"{acc_uncal:.1f}%", f"{ece_uncal:.2f}%", f"{brier_uncal:.4f}", f"{nll_uncal:.4f}"]
    ]

    # Iterative Calibrated
    for n_calib in CALIB_SIZES:
        subset_res = calib_results[:n_calib]
        subset_lbl = calib_labels[:n_calib]
        
        scaler = TemperatureScaler()
        scaler.fit(subset_res, subset_lbl)
        T = scaler.temperature.item()
        
        cal_preds, cal_confs, cal_probs = [], [], []
        for r in eval_results:
            cal_r = scaler.transform(r)
            cal_preds.append(cal_r.choice)
            cal_confs.append(cal_r.top_probability)
            cal_probs.append([cal_r.probabilities.get(n, 0.0) for n in label_names])
            
        cal_corr = [p == t for p, t in zip(cal_preds, eval_true_labels)]
        acc_cal = accuracy(cal_preds, eval_true_labels)
        ece_cal = ece_score(cal_confs, cal_corr)
        brier_cal = brier_score(cal_probs, eval_true_indices, n_classes)
        nll_cal = nll_score(cal_probs, eval_true_indices)
        
        table_data.append([
            f"N = {n_calib}",
            f"{T:.3f}",
            f"{acc_cal:.1f}%",
            f"{ece_cal:.2f}%",
            f"{brier_cal:.4f}",
            f"{nll_cal:.4f}"
        ])

    print("\n" + "=" * 70)
    print("  FINAL CALIBRATION SCALING (Banking77)")
    print("=" * 70)
    headers = ["Calibration Set", "Temperature", "Accuracy", "ECE", "Brier", "NLL"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 70)
    print("Note: ECE naturally drops as calibration density increases.")
    print("For 77 classes, N=2000 yields ~26 examples per class.")

if __name__ == "__main__":
    main()
