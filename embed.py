"""Turn text into vectors.

Providers (EMBEDDINGS env var; if unset, picked from whichever key you have):
  openrouter  OpenRouter, model openai/text-embedding-3-small  (OPENROUTER_API_KEY)
  openai      OpenAI direct, model text-embedding-3-small      (OPENAI_API_KEY)
  fake        cheap deterministic vectors, only for testing the plumbing
              (search quality is meaningless; keyword search still works)

Both real providers use the same OpenAI SDK; only the base URL and key differ.
Settings come from the environment or a .env file (see .env.example).
"""
from __future__ import annotations

import config  # noqa: F401  (loads .env first)

import hashlib
import math
import os
import random
import re

from tracing import trace

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODELS = {"openrouter": "openai/text-embedding-3-small", "openai": "text-embedding-3-small"}
FAKE_DIM = 64
BATCH = 96


def get_mode(mode: str | None = None) -> str:
    mode = (mode or os.getenv("EMBEDDINGS") or "").lower()
    if not mode:
        if os.getenv("OPENROUTER_API_KEY"):
            mode = "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            mode = "openai"
        else:
            raise SystemExit("No embeddings key found. Put OPENROUTER_API_KEY=... in .env "
                             "(see .env.example), or run with EMBEDDINGS=fake to test the plumbing.")
    if mode not in ("openrouter", "openai", "fake"):
        raise SystemExit(f"Unknown EMBEDDINGS mode '{mode}' (use openrouter, openai or fake)")
    return mode


def model_name(mode: str) -> str:
    return os.getenv("EMBED_MODEL") or DEFAULT_MODELS.get(mode, "fake")


def embedding_key(mode: str) -> str:
    """Name for the Weaviate collection. Same model => same collection, whichever provider
    serves it, so switching OpenAI <-> OpenRouter does not force a re-embed."""
    if mode == "fake":
        return "fake"
    return re.sub(r"[^0-9A-Za-z]+", "_", model_name(mode).split("/")[-1]).strip("_")


def _fake(text: str) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    v = [rng.gauss(0, 1) for _ in range(FAKE_DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def make_client(mode: str):
    from openai import OpenAI

    if mode == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set (put it in .env, see .env.example).")
        return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL,
                      default_headers={"X-Title": "hdb-rental-rag"})
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env, see .env.example).")
    return OpenAI(api_key=key)


@trace("embed", run_type="embedding")
def embed(texts: list[str], mode: str | None = None) -> list[list[float]]:
    mode = get_mode(mode)
    if mode == "fake":
        return [_fake(t) for t in texts]
    client, model = make_client(mode), model_name(mode)
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=model, input=texts[i : i + BATCH])
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
