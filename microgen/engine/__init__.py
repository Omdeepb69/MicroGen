"""LLM engine abstractions (LLMEngine and SpeculativeEngine)."""

from microgen.sdk.engine import LLMEngine
from microgen.scheduler.speculative import SpeculativeEngine, SpeculativeResult

__all__ = [
    "LLMEngine",
    "SpeculativeEngine",
    "SpeculativeResult",
]
