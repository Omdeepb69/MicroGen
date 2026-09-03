"""
Workload generation module for empirical LLM inference benchmarking.

Provides deterministic, reproducible request suites across standardized sequence
length regimes (short: 32-128, medium: 256-512, long: 1024-2048, heterogeneous,
and shared-prefix distributions).
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class WorkloadRequest:
    """Represents a single benchmark request with deterministic tokenized prompt."""
    request_id: str
    prompt_text: str
    prompt_ids: List[int]
    prompt_len: int
    max_new_tokens: int


@dataclass(frozen=True)
class WorkloadSuite:
    """Represents a collection of requests for a specific benchmark scenario."""
    name: str
    requests: List[WorkloadRequest]
    seed: int
    min_prompt_len: int
    max_prompt_len: int
    mean_prompt_len: float

    def get_token_tensors(self, device: str = "cpu") -> List[torch.Tensor]:
        """Returns prompt token IDs as 2D tensors of shape (1, seq_len)."""
        return [
            torch.tensor([req.prompt_ids], dtype=torch.long, device=device)
            for req in self.requests
        ]


# Base vocabulary / text fragments for deterministic prompt construction
_TEXT_CORPUS_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "artificial",
    "intelligence", "inference", "optimization", "latency", "throughput", "memory",
    "attention", "transformer", "neural", "network", "quantization", "paged",
    "cache", "speculative", "decoding", "tensor", "parallelism", "continuous",
    "batching", "profiling", "system", "performance", "scaling", "empirical",
    "benchmark", "evaluation", "framework", "architecture", "hardware", "cuda",
    "bandwidth", "capacity", "eviction", "prefix", "reuse", "token", "generation"
]


class WorkloadGenerator:
    """
    Deterministic workload generator for standardized LLM inference benchmarking.

    Uses a HuggingFace tokenizer (defaults to 'sshleifer/tiny-gpt2') to guarantee
    exact token counts and sequence bounds.
    """

    def __init__(self, tokenizer_name_or_path: str = "sshleifer/tiny-gpt2"):
        self.tokenizer_name = tokenizer_name_or_path
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _build_prompt_for_target_length(self, target_len: int, rng: random.Random) -> Tuple[str, List[int]]:
        """Generates a text prompt whose tokenized length matches target_len."""
        words: List[str] = []
        while True:
            word = rng.choice(_TEXT_CORPUS_WORDS)
            words.append(word)
            candidate_text = " ".join(words)
            tokens = self.tokenizer.encode(candidate_text, add_special_tokens=False)
            if len(tokens) >= target_len:
                # Trim exact tokens to target_len
                exact_tokens = tokens[:target_len]
                exact_text = self.tokenizer.decode(exact_tokens)
                return exact_text, exact_tokens

    def generate_suite(
        self,
        name: str,
        num_requests: int,
        target_len_range: Tuple[int, int],
        max_new_tokens: int = 32,
        seed: int = 42,
    ) -> WorkloadSuite:
        """Generates a workload suite given sequence length bounds."""
        rng = random.Random(seed)
        requests: List[WorkloadRequest] = []
        min_len, max_len = target_len_range

        for i in range(num_requests):
            target_len = rng.randint(min_len, max_len)
            prompt_text, prompt_ids = self._build_prompt_for_target_length(target_len, rng)
            requests.append(
                WorkloadRequest(
                    request_id=f"{name}_{i:03d}",
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    prompt_len=len(prompt_ids),
                    max_new_tokens=max_new_tokens,
                )
            )

        prompt_lens = [req.prompt_len for req in requests]
        return WorkloadSuite(
            name=name,
            requests=requests,
            seed=seed,
            min_prompt_len=min(prompt_lens),
            max_prompt_len=max(prompt_lens),
            mean_prompt_len=sum(prompt_lens) / len(prompt_lens),
        )

    def generate_short_workload(self, num_requests: int = 10, seed: int = 42, max_new_tokens: int = 32) -> WorkloadSuite:
        """Short prompts: 32 - 128 tokens."""
        return self.generate_suite(
            name="short",
            num_requests=num_requests,
            target_len_range=(32, 128),
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    def generate_medium_workload(self, num_requests: int = 10, seed: int = 42, max_new_tokens: int = 64) -> WorkloadSuite:
        """Medium prompts: 256 - 512 tokens."""
        return self.generate_suite(
            name="medium",
            num_requests=num_requests,
            target_len_range=(256, 512),
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    def generate_long_workload(self, num_requests: int = 10, seed: int = 42, max_new_tokens: int = 128) -> WorkloadSuite:
        """Long prompts: 1024 - 2048 tokens."""
        return self.generate_suite(
            name="long",
            num_requests=num_requests,
            target_len_range=(1024, 2048),
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    def generate_heterogeneous_workload(self, num_requests: int = 20, seed: int = 42) -> WorkloadSuite:
        """Heterogeneous prompts combining short, medium, and long sequences."""
        rng = random.Random(seed)
        requests: List[WorkloadRequest] = []
        ranges = [(32, 128), (256, 512), (1024, 2048)]

        for i in range(num_requests):
            min_len, max_len = rng.choice(ranges)
            max_new = rng.choice([16, 32, 64, 128])
            target_len = rng.randint(min_len, max_len)
            prompt_text, prompt_ids = self._build_prompt_for_target_length(target_len, rng)
            requests.append(
                WorkloadRequest(
                    request_id=f"hetero_{i:03d}",
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    prompt_len=len(prompt_ids),
                    max_new_tokens=max_new,
                )
            )

        prompt_lens = [req.prompt_len for req in requests]
        return WorkloadSuite(
            name="heterogeneous",
            requests=requests,
            seed=seed,
            min_prompt_len=min(prompt_lens),
            max_prompt_len=max(prompt_lens),
            mean_prompt_len=sum(prompt_lens) / len(prompt_lens),
        )

    def generate_shared_prefix_workload(
        self,
        num_requests: int = 10,
        total_prompt_len: int = 256,
        prefix_ratio: float = 0.5,
        seed: int = 42,
        max_new_tokens: int = 32,
    ) -> WorkloadSuite:
        """Generates prompts with an exact controlled shared prefix ratio."""
        rng = random.Random(seed)
        prefix_tokens_len = int(total_prompt_len * prefix_ratio)
        suffix_tokens_len = total_prompt_len - prefix_tokens_len

        prefix_text, prefix_ids = self._build_prompt_for_target_length(prefix_tokens_len, rng)
        requests: List[WorkloadRequest] = []

        for i in range(num_requests):
            suffix_text, suffix_ids = self._build_prompt_for_target_length(suffix_tokens_len, rng)
            full_ids = prefix_ids + suffix_ids
            full_text = self.tokenizer.decode(full_ids)
            requests.append(
                WorkloadRequest(
                    request_id=f"shared_prefix_{int(prefix_ratio*100)}_{i:03d}",
                    prompt_text=full_text,
                    prompt_ids=full_ids,
                    prompt_len=len(full_ids),
                    max_new_tokens=max_new_tokens,
                )
            )

        prompt_lens = [req.prompt_len for req in requests]
        return WorkloadSuite(
            name=f"shared_prefix_{int(prefix_ratio*100)}pct",
            requests=requests,
            seed=seed,
            min_prompt_len=min(prompt_lens),
            max_prompt_len=max(prompt_lens),
            mean_prompt_len=sum(prompt_lens) / len(prompt_lens),
        )
