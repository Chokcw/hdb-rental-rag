"""Turn text into vectors.

mode "openai": OpenAI text-embedding-3-small (needs OPENAI_API_KEY).
mode "fake":   cheap deterministic vectors, only for testing the plumbing
               (search quality will be meaningless, keyword search still works).

Pick the mode with the EMBEDDINGS environment variable (default: openai).
"""
from __future__ import annotations

import hashlib
import math
import os
import random

from tracing import trace

OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
FAKE_DIM = 64
BATCH = 96


def get_mode(mode: str | None = None) -> str:
    mode = (mode or os.getenv("EMBEDDINGS", "openai")).lower()
    if mode not in ("openai", "fake"):
        raise SystemExit(f"Unknown EMBEDDINGS mode '{mode}' (use openai or fake)")
    return mode


def _fake(text: str) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    v = [rng.gauss(0, 1) for _ in range(FAKE_DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@trace("embed", run_type="embedding")
def embed(texts: list[str], mode: str | None = None) -> list[list[float]]:
    mode = get_mode(mode)
    if mode == "fake":
        return [_fake(t) for t in texts]

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (or run with EMBEDDINGS=fake to test the plumbing).")
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=OPENAI_MODEL, input=texts[i : i + BATCH])
        out.extend(d.embedding for d in resp.data)
    return out
