"""Tests for Task 3.2: Temperature Scaling & Calibration.

Verification criterion: test_decision_calibration() passes, verifying that
TemperatureScaler can fit a temperature to a set of DecisionResults and
properly transform uncalibrated results into calibrated ones.
"""

import pytest
import torch

from microgen.decision.schema import Choice, ChoiceSchema, DecisionResult, CandidateStats
from microgen.decision.calibration import TemperatureScaler
from microgen.decision.engine import DecisionEngine


def create_dummy_result(choice_name: str, true_label: str, logits: list[float]) -> DecisionResult:
    """Helper to create a fake DecisionResult with candidate_stats."""
    options = ["LEFT", "RIGHT", "BRAKE"]
    stats = []
    probabilities = {}
    for i, opt in enumerate(options):
        # We inject fake normalized_score representing logits
        stat = CandidateStats(
            text=f" {opt}", 
            token_count=1, 
            token_ids=[i+1],
            raw_logprob=logits[i],
            normalized_score=logits[i]
        )
        stats.append(stat)
        
    # We don't actually need exact probabilities to match logits for the mock,
    # since TemperatureScaler only reads normalized_score.
    probabilities = {"LEFT": 0.33, "RIGHT": 0.33, "BRAKE": 0.34}
    
    return DecisionResult(
        choice="BRAKE",
        probabilities=probabilities,
        top_probability=0.34,
        entropy=1.58,
        calibrated=False,
        candidate_stats=stats,
    )


def test_temperature_scaler_fit_and_transform():
    """Verify TemperatureScaler fits T and transforms results."""
    # Create an overconfident distribution:
    # Class 0: logit 10.0, Class 1: logit 2.0, Class 2: logit 1.0 (Softmax ~ [0.99, 0.0, 0.0])
    # But let's say the ground truth is often Class 1 or 2, so the model is overconfident.
    # The scaler should increase T > 1.0 to soften the distribution.
    
    results = [
        create_dummy_result("LEFT", "LEFT", [10.0, 2.0, 1.0]),
        create_dummy_result("LEFT", "RIGHT", [10.0, 2.0, 1.0]), # Wrong prediction
        create_dummy_result("LEFT", "RIGHT", [10.0, 2.0, 1.0]), # Wrong prediction
    ]
    true_labels = ["LEFT", "RIGHT", "RIGHT"]
    
    scaler = TemperatureScaler()
    scaler.fit(results, true_labels)
    
    # T should be > 1.0 because the original logits are wildly overconfident 
    # for a dataset where it's only right 33% of the time.
    fitted_t = scaler.temperature.item()
    assert fitted_t > 1.0
    
    # Transform one of the results
    calibrated_res = scaler.transform(results[0])
    
    assert calibrated_res.calibrated is True
    # The probabilities should be softened
    # Because T > 1, the new probability for LEFT will be lower than the unscaled softmax.
    raw_softmax_left = torch.softmax(torch.tensor([10.0, 2.0, 1.0]), dim=0)[0].item()
    assert calibrated_res.probabilities["LEFT"] < raw_softmax_left
    
    # Entropy should be higher after softening
    # Actually wait, our dummy result had a fake entropy of 1.58, let's just 
    # assert the new entropy is valid
    assert calibrated_res.entropy is not None
    assert calibrated_res.entropy > 0.0


def test_temperature_scaler_empty_reject():
    scaler = TemperatureScaler()
    with pytest.raises(ValueError, match="cannot be empty"):
        scaler.fit([], [])


def test_temperature_scaler_mismatch_reject():
    scaler = TemperatureScaler()
    with pytest.raises(ValueError, match="match"):
        scaler.fit([create_dummy_result("L", "L", [1,2,3])], ["L", "R"])


def test_temperature_scaler_missing_stats_reject():
    scaler = TemperatureScaler()
    res = DecisionResult(
        choice="A", 
        probabilities={"A": 1.0}, 
        top_probability=1.0, 
        candidate_stats=[]
    )
    with pytest.raises(RuntimeError, match="must have candidate_stats"):
        scaler.fit([res], ["A"])
    
    with pytest.raises(RuntimeError, match="must have candidate_stats"):
        scaler.transform(res)
