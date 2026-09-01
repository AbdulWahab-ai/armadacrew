"""ArmadaCrew — durable multi-agent orchestration for operations copilots.

A LangGraph-style runtime (nodes, edges, reducers, checkpointer, HITL interrupts)
implemented without a LangGraph dependency so the repo runs offline with a mock LLM.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
