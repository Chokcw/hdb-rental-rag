"""Chunk the corpus, embed new chunks, and load them into Weaviate.

Safe to re-run any time. A chunk whose text hasn't changed keeps its id, so it is
not embedded again (no API cost); chunks that disappeared from the corpus are deleted.

    python index.py                      # normal run (needs OPENAI_API_KEY)
    EMBEDDINGS=fake python index.py      # test the plumbing without OpenAI
    python index.py --rebuild            # wipe and re-embed everything
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chunker import build_chunks
from embed import embed, get_mode
from store import WeaviateStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--rebuild", action="store_true", help="delete the collection and start again")
    args = ap.parse_args()

    mode = get_mode()
    chunks, skipped = build_chunks(Path(args.corpus))
    print(f"{len(chunks)} chunks from {len({c.page_id for c in chunks})} pages "
          f"({len(skipped)} pages skipped)")

    store = WeaviateStore(mode)
    store.ensure_collection(rebuild=args.rebuild)
    stored = store.existing()
    wanted = {c.uuid: c for c in chunks}

    new = [c for c in chunks if c.uuid not in stored]
    kept = [c for c in chunks if c.uuid in stored]
    stale = [u for u in stored if u not in wanted]

    if new:
        print(f"embedding {len(new)} new chunks ({mode})...")
        store.upsert(new, embed([c.embed_text for c in new], mode))
    store.update_properties(kept)      # refresh "last checked" dates, keep vectors
    store.delete(stale)

    print(f"new {len(new)} | unchanged {len(kept)} | removed {len(stale)} | now in Weaviate: {store.count()}")


if __name__ == "__main__":
    main()
