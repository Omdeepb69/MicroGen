import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
import time

def main():
    print("Loading SmolLM-135M...")
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Needs to be on CPU for simple local test unless CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded on {device}. Creating DecisionEngine...")
    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)
    
    context = "The drone is flying towards a tall building. It is 5 meters away."
    question = "Is the drone in danger of colliding?"
    
    print("\nRunning engine.yes_no()...")
    start_time = time.time()
    result = engine.yes_no(context=context, question=question)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f}s")
    print(f"Choice: {result.choice}")
    print(f"Top Probability: {result.top_probability:.4f}")
    print(f"Entropy: {result.entropy:.4f} bits")
    print("All Probabilities:")
    for opt, prob in result.probabilities.items():
        print(f"  {opt}: {prob:.4f}")
        
    print("\nRunning multi-token options via choose()...")
    schema = ChoiceSchema(
        name="action",
        options=[
            Choice("TURN LEFT"),
            Choice("PULL UP"),
            Choice("BRAKE"),
            Choice("CONTINUE"),
        ]
    )
    start_time = time.time()
    result2 = engine.choose(context=context + "\nWhat should the drone do next?", schema=schema)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f}s")
    print(f"Choice: {result2.choice}")
    print(f"Top Probability: {result2.top_probability:.4f}")
    print("All Probabilities:")
    for opt, prob in result2.probabilities.items():
        print(f"  {opt}: {prob:.4f}")

if __name__ == "__main__":
    main()
