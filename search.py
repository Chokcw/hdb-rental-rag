"""Retrieval: question in, best chunks out.

This is the only function the rest of the app (answering, evals) calls:

    from search import retrieve
    hits = retrieve("How many tenants can I have in a 4-room flat?", k=5, alpha=0.5)

Command line:
    python search.py "can a PR rent out their HDB flat" --k 5 --alpha 0.5 --audience owner
"""
from __future__ import annotations

import argparse
from functools import lru_cache

from embed import embed, get_mode
from store import WeaviateStore
from tracing import trace


@lru_cache(maxsize=1)
def _store(mode: str) -> WeaviateStore:
    return WeaviateStore(mode)


@trace("retrieve", run_type="retriever")
def retrieve(question: str, k: int = 5, alpha: float = 0.5, audience: str | None = None) -> list[dict]:
    """alpha: 0 = keyword only, 1 = meaning (vector) only. audience: owner | tenant | None."""
    mode = get_mode()
    vector = embed([question], mode)[0]
    return _store(mode).hybrid_search(question, vector, k=k, alpha=alpha, audience=audience)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--audience", choices=["owner", "tenant"])
    args = ap.parse_args()

    for i, h in enumerate(retrieve(args.question, args.k, args.alpha, args.audience), 1):
        snippet = " ".join(h["text"].split())[:220]
        print(f"\n{i}. [{h['score']}] {h['title']} > {h['heading_path']}  (checked {h['fetched_at'][:10]})")
        print(f"   {h['url']}\n   {snippet}...")


if __name__ == "__main__":
    main()
