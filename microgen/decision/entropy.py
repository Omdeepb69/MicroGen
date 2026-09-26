"""Entropy metrics for candidate distributions.

This module provides functions to calculate the Shannon entropy of a
decision distribution, which is used to quantify uncertainty across
the constrained candidate space.
"""

import math

def calculate_entropy(probabilities: dict[str, float]) -> float:
    """Calculate the Shannon entropy (in bits) of a probability distribution.

    Args:
        probabilities: Mapping of candidate names to their probabilities.
            Values must be in [0, 1] and sum to 1.0.

    Returns:
        Shannon entropy in bits. Returns 0.0 if the distribution is deterministic
        (one option has probability 1.0) or if the dictionary is empty.
    """
    if not probabilities:
        return 0.0

    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0.0:
            # Add 1e-12 to prevent math domain error just in case, though
            # the > 0.0 check handles exact zeros.
            entropy -= prob * math.log2(prob)
            
    # Float inaccuracies can sometimes result in a very small negative number
    return max(0.0, entropy)
