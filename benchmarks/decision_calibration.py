"""Calibration Benchmark using BoolQ.

Downloads the BoolQ validation dataset to evaluate the decision engine's
calibration (Expected Calibration Error). Fits the TemperatureScaler on
a subset of the results to show how the confidence distribution improves.
"""

import time
import argparse
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.engine import DecisionEngine
from microgen.decision.calibration import TemperatureScaler


def expected_calibration_error(confidences: list[float], accuracies: list[float], num_bins: int = 10) -> float:
    """Calculate ECE (Expected Calibration Error)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get elements in this bin
        in_bin = []
        for j, conf in enumerate(confidences):
            if bin_lower < conf <= bin_upper:
                in_bin.append(j)
                
        # To cover 0.0 exactly in the first bin
        if i == 0:
            for j, conf in enumerate(confidences):
                if conf == 0.0:
                    in_bin.append(j)
                    
        if not in_bin:
            continue
            
        bin_acc = np.mean([accuracies[j] for j in in_bin])
        bin_conf = np.mean([confidences[j] for j in in_bin])
        bin_weight = len(in_bin) / total_samples
        
        ece += bin_weight * np.abs(bin_acc - bin_conf)
        
    return float(ece)


def run_calibration_benchmark(num_evals: int = 50, model_name: str = "HuggingFaceTB/SmolLM-135M"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    print("Downloading BoolQ dataset (validation split)...")
    dataset = load_dataset("google/boolq", split="validation")
    
    # Take a random subset
    dataset = dataset.shuffle(seed=42).select(range(min(num_evals, len(dataset))))
    
    results_list = []
    true_labels = []
    
    print(f"Running {len(dataset)} evaluations on {device}...")
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        # BoolQ gives answer as boolean True/False.
        # We mapped it to "YES" or "NO"
        true_label = "YES" if item.get("answer", False) else "NO"
        true_labels.append(true_label)
        
        # Construct prompt
        context = f"Passage: {item['passage']}"
        question = item['question']
        
        result = engine.yes_no(context=context, question=question)
        results_list.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(dataset)}...")

    print(f"Finished inferences in {time.time() - start_time:.2f}s")
    
    # 1. Analyze Uncalibrated ECE
    uncal_confidences = []
    uncal_accuracies = []
    
    for res, label in zip(results_list, true_labels):
        uncal_confidences.append(res.top_probability)
        uncal_accuracies.append(1.0 if res.choice == label else 0.0)
        
    uncal_acc = np.mean(uncal_accuracies)
    uncal_ece = expected_calibration_error(uncal_confidences, uncal_accuracies)
    
    print("\n--- Uncalibrated Results ---")
    print(f"Accuracy: {uncal_acc * 100:.2f}%")
    print(f"ECE:      {uncal_ece * 100:.2f}%")
    print(f"Mean Conf:{np.mean(uncal_confidences) * 100:.2f}%")
    
    # 2. Fit Temperature Scaler
    print("\nFitting TemperatureScaler (L-BFGS)...")
    scaler = TemperatureScaler()
    scaler.fit(results_list, true_labels)
    fitted_t = scaler.temperature.item()
    print(f"Optimal Temperature: {fitted_t:.4f}")
    
    # 3. Analyze Calibrated ECE
    cal_confidences = []
    cal_accuracies = []
    
    for res, label in zip(results_list, true_labels):
        cal_res = scaler.transform(res)
        cal_confidences.append(cal_res.top_probability)
        cal_accuracies.append(1.0 if cal_res.choice == label else 0.0)
        
    cal_acc = np.mean(cal_accuracies)
    cal_ece = expected_calibration_error(cal_confidences, cal_accuracies)
    
    print("\n--- Calibrated Results ---")
    print(f"Accuracy: {cal_acc * 100:.2f}%")
    print(f"ECE:      {cal_ece * 100:.2f}%")
    print(f"Mean Conf:{np.mean(cal_confidences) * 100:.2f}%")
    
    print("\nCalibration " + ("IMPROVED" if cal_ece < uncal_ece else "WORSENED") + " the Expected Calibration Error (ECE).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration benchmarks.")
    parser.add_argument("--evals", type=int, default=50, help="Number of evaluations to run.")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="HuggingFace model ID.")
    args = parser.parse_args()
    
    run_calibration_benchmark(num_evals=args.evals, model_name=args.model)
