"""Deterministic test fixtures for the decision module."""

import pytest
import torch
from typing import Any, Tuple


class MockTokenizer:
    def __init__(self):
        # Extremely simplified mock tokenizer.
        # Keys use leading spaces to match BPE mid-sentence representations.
        self.vocab = {
            " LEFT": 1, " RIGHT": 2, " BRAKE": 3, " CLIMB": 4,
            " RETURN": 5, " HOME": 6,
            # yes_no labels
            " YES": 7, " NO": 8,
            # score labels (digits)
            " 1": 11, " 2": 12, " 3": 13, " 4": 14, " 5": 15,
        }
        self.pad_token_id = 0
        self.eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Greedy left-to-right tokenization against the vocab entries.
        # Vocab keys include a leading space (e.g., " LEFT"), so we scan
        # the text for the longest matching key at each position.
        result: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for key in sorted(self.vocab, key=len, reverse=True):
                if text[i:].startswith(key):
                    result.append(self.vocab[key])
                    i += len(key)
                    matched = True
                    break
            if not matched:
                i += 1  # skip unrecognised character
        return result

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        # Reverse vocab lookup
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([inv_vocab.get(tid, "") for tid in token_ids])

    def __call__(self, text: str, return_tensors: str = "pt"):
        class MockOutput:
            def __init__(self, ids: list[int]):
                # Always produce at least one token (sentinel 0) so the engine
                # never hands a zero-length input_ids to prefill.
                if not ids:
                    ids = [0]
                self.input_ids = torch.tensor([ids])
                self.attention_mask = torch.ones_like(self.input_ids)
                
            def get(self, key, default=None):
                return getattr(self, key, default)

        return MockOutput(self.encode(text))


class MockDecisionModel:
    """A deterministic mock implementation of the DecisionModel protocol."""

    def __init__(self):
        self._tokenizer = MockTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device_type(self):
        return "cpu"

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        # Return dummy logits of shape [1, seq_len, vocab_size].
        # seq_len is always >= 1 because MockTokenizer.__call__ injects a sentinel.
        vocab_size = 100
        seq_len = input_ids.shape[1]
        assert seq_len > 0, "prefill received empty input_ids — fixture invariant violated"
        logits = torch.zeros(1, seq_len, vocab_size)
        # Give " LEFT" (id 1) the highest logit at the final position.
        logits[0, -1, 1] = 10.0
        logits[0, -1, 2] = 2.0
        logits[0, -1, 3] = 5.0
        logits[0, -1, 4] = 1.0
        # yes_no: YES (id 7) wins over NO (id 8)
        logits[0, -1, 7] = 8.0
        logits[0, -1, 8] = 2.0
        # score digits: "4" (id 14) wins
        logits[0, -1, 11] = 1.0
        logits[0, -1, 12] = 2.0
        logits[0, -1, 13] = 3.0
        logits[0, -1, 14] = 7.0
        logits[0, -1, 15] = 2.5
        # multi-token: " RETURN" (id 5) gets a moderate logit, " HOME" (id 6) will be scored in decode
        logits[0, -1, 5] = 4.0
        return logits, cache if cache is not None else {"mock_cache": True}

    def decode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        # Dummy decode for multi-token sequence scoring.
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        
        # If the input token is " RETURN" (5), give a very high logit to " HOME" (6)
        if token_ids.shape[-1] > 0 and token_ids[0, -1].item() == 5:
            logits[0, 0, 6] = 15.0
            
        return logits, cache


@pytest.fixture
def mock_decision_model():
    return MockDecisionModel()
