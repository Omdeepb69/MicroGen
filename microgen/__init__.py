"""microgen: Lightweight, modular, hardware-aware LLM inference engine."""

__version__ = "1.0.0"

# Apply transformers compatibility patch at package initialization time
import microgen.backends.pytorch  # noqa: F401

# Top-level SDK Engine shortcut
from microgen.sdk.engine import LLMEngine

# Package module re-exports
from microgen import backends
from microgen import memory
from microgen import caching
from microgen import scheduler
from microgen import engine
from microgen import profiling
from microgen import benchmarks

__all__ = [
    "__version__",
    "LLMEngine",
    "backends",
    "memory",
    "caching",
    "scheduler",
    "engine",
    "profiling",
    "benchmarks",
]
