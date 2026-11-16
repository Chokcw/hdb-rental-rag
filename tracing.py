"""All tracing lives in this one file.

Decorate a function with @trace("name") and, when LangSmith is switched on, every
call shows up as a step ("span") in the trace: inputs, outputs, timing, errors.
When it is off (no API key), the decorator does nothing and costs nothing.

Switch it on with environment variables (set them *before* running Python):
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=...        # from smith.langchain.com
    export LANGSMITH_PROJECT=hdb-rental-rag

To use a different tool later (e.g. Langfuse), only this file needs to change.
"""
from __future__ import annotations

import config  # noqa: F401  (loads .env before we read LANGSMITH_* variables)

import os

try:
    from langsmith import traceable as _traceable
except ImportError:  # langsmith not installed -> tracing stays off
    _traceable = None


def enabled() -> bool:
    return (
        _traceable is not None
        and os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def trace(name: str | None = None, run_type: str = "chain"):
    """run_type: chain | retriever | embedding | llm | tool"""

    def decorator(fn):
        if not enabled():
            return fn
        return _traceable(name=name or fn.__name__, run_type=run_type)(fn)

    return decorator
