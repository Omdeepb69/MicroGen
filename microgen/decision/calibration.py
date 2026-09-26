"""Probability calibration via Temperature Scaling.

Implements a TemperatureScaler that fits a single scalar temperature T
to a validation set of DecisionResults, smoothing or sharpening the
normalized scores (logits) to produce calibrated probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Sequence

from microgen.decision.schema import DecisionResult
from microgen.decision.entropy import calculate_entropy


class TemperatureScaler(nn.Module):
    """Calibrates DecisionResult probabilities using temperature scaling.

    Optimizes a single temperature parameter T on a validation set to minimize
    Negative Log Likelihood (NLL). Can then transform uncalibrated
    DecisionResults into calibrated ones.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize temperature to 1.5 as per standard practice, though 1.0 is also fine.
        # It's a learnable parameter.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, results: Sequence[DecisionResult], true_labels: Sequence[str], max_iter: int = 50) -> None:
        """Fit the temperature parameter using L-BFGS on the validation set.

        Args:
            results: A list of uncalibrated DecisionResults containing candidate_stats.
            true_labels: The ground truth choice names for each result.
            max_iter: Maximum iterations for the L-BFGS optimizer.
        
        Raises:
            ValueError: If results list is empty or lengths mismatch.
            RuntimeError: If any DecisionResult lacks candidate_stats.
        """
        if not results:
            raise ValueError("Calibration set (results) cannot be empty.")
        if len(results) != len(true_labels):
            raise ValueError("Length of results and true_labels must match.")

        # Build tensors for logits and labels
        logits_list = []
        labels_list = []

        for result, true_label in zip(results, true_labels):
            if not result.candidate_stats:
                raise RuntimeError(
                    "DecisionResult must have candidate_stats to be used for calibration."
                )
            
            # Reconstruct the order of logits based on candidate_stats
            # DecisionEngine rekeys probabilities back to Choice.name, but candidate_stats.text
            # has the internal tokenizer format. However, we can just extract the scores
            # and map the true_label correctly.
            # Actually, to be safe, we need the scores to match the choices.
            # In candidate_stats, text is the internal representation.
            # Let's map from internal text to choice name.
            # But we don't have the text_to_name map here. 
            # We can reconstruct it because result.probabilities has the choice names,
            # and candidate_stats has the internal text. We assume the order of candidate_stats
            # matches the order of probabilities? Actually dicts are ordered in python 3.7+,
            # but safer to find the true label index based on probability matching or just order.
            
            # Since engine.choose appends to candidate_stats in the exact same order as schema.options,
            # and result.probabilities is a dict created from that same order,
            # we can just zip them.
            choice_names = list(result.probabilities.keys())
            try:
                true_idx = choice_names.index(true_label)
            except ValueError:
                raise ValueError(f"True label '{true_label}' not found in candidates {choice_names}")
            
            scores = []
            for stat in result.candidate_stats:
                if stat.normalized_score is None:
                    raise RuntimeError("candidate_stats must have normalized_score populated.")
                scores.append(stat.normalized_score)
            
            logits_list.append(scores)
            labels_list.append(true_idx)

        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # Optimize temperature w.r.t NLL
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_closure():
            optimizer.zero_grad()
            # T must be positive
            t = self.temperature.clamp(min=1e-3)
            loss = criterion(logits_tensor / t, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

    def transform(self, result: DecisionResult) -> DecisionResult:
        """Apply the fitted temperature to scale the logits and return a new calibrated result.

        Args:
            result: Uncalibrated DecisionResult containing candidate_stats.

        Returns:
            A new DecisionResult with calibrated=True, scaled probabilities, and updated entropy.
        """
        if not result.candidate_stats:
            raise RuntimeError("DecisionResult must have candidate_stats to be transformed.")

        scores = []
        for stat in result.candidate_stats:
            if stat.normalized_score is None:
                raise RuntimeError("candidate_stats must have normalized_score populated.")
            scores.append(stat.normalized_score)

        logits_tensor = torch.tensor(scores, dtype=torch.float32)
        
        with torch.no_grad():
            t = self.temperature.clamp(min=1e-3).item()
            calibrated_probs = torch.softmax(logits_tensor / t, dim=0).tolist()

        choice_names = list(result.probabilities.keys())
        new_probabilities = {
            name: prob for name, prob in zip(choice_names, calibrated_probs)
        }
        
        best_choice = max(new_probabilities, key=new_probabilities.__getitem__)
        top_prob = new_probabilities[best_choice]
        new_entropy = calculate_entropy(new_probabilities)

        return DecisionResult(
            choice=best_choice,
            probabilities=new_probabilities,
            top_probability=top_prob,
            entropy=new_entropy,
            calibrated=True,
            candidate_stats=result.candidate_stats,
        )
