"""Chat model access (answers, and later the eval judge).

Uses the OpenAI SDK pointed at OpenRouter (default if OPENROUTER_API_KEY is set) or
OpenAI direct. Settings come from .env:

    CHAT_MODEL=openai/gpt-4o-mini        # any OpenRouter model id; change freely
    CHAT_PROVIDER=openrouter             # optional: openrouter | openai
"""
from __future__ import annotations

import config  # noqa: F401  (loads .env first)

import os

from embed import make_client
from tracing import wrap_openai

DEFAULT_MODELS = {"openrouter": "openai/gpt-4o-mini", "openai": "gpt-4o-mini"}


def provider() -> str:
    p = (os.getenv("CHAT_PROVIDER") or "").lower()
    if not p:
        p = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    if p not in ("openrouter", "openai"):
        raise SystemExit(f"Unknown CHAT_PROVIDER '{p}' (use openrouter or openai)")
    return p


def chat_model() -> str:
    return os.getenv("CHAT_MODEL") or DEFAULT_MODELS[provider()]


class ChatError(RuntimeError):
    """The chat call failed (bad model id, bad key, provider error)."""


def judge_model() -> str:
    """Model for the eval judge. Prefer a different model from CHAT_MODEL so it is not grading itself."""
    return os.getenv("JUDGE_MODEL") or chat_model()


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1600,
         model: str | None = None) -> tuple[str, dict]:
    """Returns (text, usage). Raises ChatError with a readable message if the call fails."""
    client = wrap_openai(make_client(provider()))
    model = model or chat_model()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:  # surface model-id typos and key problems clearly
        raise ChatError(f"Chat call failed for model '{model}' via {provider()}: {e} "
                        f"(check CHAT_MODEL / JUDGE_MODEL and your API key in .env)") from e
    text = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    usage = {"model": model, "prompt_tokens": getattr(u, "prompt_tokens", None),
             "completion_tokens": getattr(u, "completion_tokens", None)}
    return text, usage
