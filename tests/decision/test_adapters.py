"""Tests for the decision model adapters."""

import torch

def test_mock_decision_model_prefill(mock_decision_model):
    """Test that the mock decision model provides deterministic logits."""
    input_ids = torch.tensor([[10, 11, 12]])
    logits, cache = mock_decision_model.prefill(input_ids)
    
    assert logits.shape == (1, 3, 100)
    assert cache == {"mock_cache": True}
    
    # " LEFT" token (id 1) should have the highest logit
    assert logits[0, -1, 1] == 10.0
    assert logits[0, -1, 2] == 2.0
