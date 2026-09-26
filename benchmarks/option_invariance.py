"""Option-Order Invariance Experiment.

Evaluates the robustness of the decision engine to candidate ordering.
In generative LLMs, the order of options in a prompt (e.g. A, B, C, D) 
creates severe position bias. In our decode-free constrained engine,
the order of candidates in the ChoiceSchema should mathematically yield 
zero position bias (KL Divergence = 0) since each candidate is scored 
independently against the same frozen prefill state.
"""

import time
import argparse
import random
import numpy as np
from scipy.stats import entropy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.engine import DecisionEngine


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Calculate D_KL(P || Q) for two probability distributions over the same keys."""
    keys = list(p.keys())
    p_arr = np.array([p[k] for k in keys], dtype=np.float64)
    q_arr = np.array([q[k] for k in keys], dtype=np.float64)
    
    # Avoid log(0)
    p_arr = np.clip(p_arr, 1e-15, 1.0)
    q_arr = np.clip(q_arr, 1e-15, 1.0)
    
    return float(entropy(p_arr, q_arr))


def run_invariance_experiment(num_trials: int = 10, model_name: str = "HuggingFaceTB/SmolLM-135M"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    base_context = "The autonomous vehicle is approaching a busy intersection. The traffic light is yellow. What should the vehicle do? Action:"
    
    original_options = [
        Choice("ACCELERATE"),
        Choice("BRAKE"),
        Choice("MAINTAIN SPEED"),
        Choice("TURN LEFT"),
        Choice("TURN RIGHT"),
    ]
    
    print("\n--- Running Original Schema ---")
    schema_original = ChoiceSchema(name="original", options=original_options)
    result_orig = engine.choose(context=base_context, schema=schema_original)
    
    print("Original Probabilities:")
    for opt, prob in result_orig.probabilities.items():
        print(f"  {opt}: {prob:.4f}")
        
    kl_divergences = []
    
    print(f"\n--- Running {num_trials} Permutations ---")
    start_time = time.time()
    
    for i in range(num_trials):
        # Create a random permutation of the choices
        permuted_options = original_options.copy()
        random.shuffle(permuted_options)
        
        schema_permuted = ChoiceSchema(name=f"permuted_{i}", options=permuted_options)
        result_perm = engine.choose(context=base_context, schema=schema_permuted)
        
        # Calculate KL divergence between original distribution and this permutation's distribution
        kl_div = kl_divergence(result_orig.probabilities, result_perm.probabilities)
        kl_divergences.append(kl_div)
        
    print(f"Completed {num_trials} trials in {time.time() - start_time:.2f}s")
    
    mean_kl = np.mean(kl_divergences)
    max_kl = np.max(kl_divergences)
    
    print("\n--- Results ---")
    print(f"Mean KL Divergence: {mean_kl:.8f}")
    print(f"Max KL Divergence:  {max_kl:.8f}")
    
    if mean_kl < 1e-5:
        print("\n✅ SUCCESS: The decision engine is perfectly robust to option ordering (Zero Position Bias).")
    else:
        print("\n❌ FAILURE: The decision engine exhibits position bias.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run option-order invariance experiment.")
    parser.add_argument("--trials", type=int, default=10, help="Number of random permutations to test.")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="HuggingFace model ID.")
    args = parser.parse_args()
    
    # Set seed for reproducible permutations
    random.seed(42)
    run_invariance_experiment(num_trials=args.trials, model_name=args.model)
