"""microgen: Lightweight, modular, hardware-aware LLM inference engine."""

__version__ = "0.1.0"

# Apply transformers compatibility patch at package initialization time
import microgen.backends.pytorch  # noqa: F401
