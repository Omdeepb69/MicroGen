import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.huggingface import TransformersDecisionModel

def main():
    print("Loading SmolLM-135M for latency scaling benchmark...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    
    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)
    
    sizes = [2, 4, 6, 10, 20, 50, 77, 100]
    
    # We create artificial candidates of length 5 words each, so they have multi-token lengths
    # and require Trie scoring (some branching).
    
    # Pre-generate some random words
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]
    
    print("\nStarting latency benchmark...")
    print(f"{'Candidates':<15} | {'Median Latency (ms)':<20} | {'p95 Latency (ms)':<20}")
    print("-" * 60)
    
    for n in sizes:
        # Create n choices
        choices = []
        for i in range(n):
            # Create a 4-token sequence (e.g., "apple banana cherry date")
            w = [words[(i + j) % len(words)] for j in range(4)]
            choices.append(Choice(name=f"c_{i}", value=" ".join(w)))
            
        schema = ChoiceSchema(name="scale", options=choices)
        context = "Classify the following text into one of the categories:\nText: The quick brown fox jumps over the lazy dog.\nCategory:"
        
        # Warmup
        engine.choose(context=context, schema=schema)
        
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            engine.choose(context=context, schema=schema)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
        median = np.median(latencies)
        p95 = np.percentile(latencies, 95)
        
        print(f"{n:<15} | {median:<20.2f} | {p95:<20.2f}")

if __name__ == "__main__":
    main()
