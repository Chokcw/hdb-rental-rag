"""Retrieval eval: for each question, is a page that contains the answer in the top k?

    python evals/run_retrieval.py --check              # validate the question file (no Weaviate needed)
    python evals/run_retrieval.py                      # hit rate at k=5, alpha=0.5
    python evals/run_retrieval.py --alpha 0 0.3 0.5 0.7 1 --k 3 5
    python evals/run_retrieval.py --misses             # list the questions that missed

A "hit" means at least one retrieved chunk comes from a page listed in `source_pages`.
Questions with no source_pages (SQL routing, out of scope) are skipped here; they are
for the routing and answer evals.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "evals" / "questions.yaml"
CORPUS = ROOT / "corpus"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def load() -> list[dict]:
    return yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]


def check(questions: list[dict]) -> int:
    """Every source page must exist and contain the quoted evidence."""
    from chunker import build_chunks

    chunks, _ = build_chunks(CORPUS)
    indexed = {c.page_id for c in chunks}
    problems = 0
    for q in questions:
        for pid in q.get("source_pages", []):
            if not (CORPUS / f"{pid}.md").exists():
                print(f"[{q['id']}] source page not found: {pid}"); problems += 1
            elif pid not in indexed:
                print(f"[{q['id']}] page exists but is not indexed (skipped/non-canonical): {pid}")
                if q["category"] != "known_gap":
                    problems += 1
        ev = q.get("evidence")
        if ev and q.get("source_pages"):
            texts = [norm((CORPUS / f"{p}.md").read_text(encoding="utf-8")) for p in q["source_pages"]
                     if (CORPUS / f"{p}.md").exists()]
            if not any(norm(ev) in t for t in texts):
                print(f"[{q['id']}] evidence not found verbatim in source pages: {ev!r}"); problems += 1
        if q["expect"] == "answer" and not q.get("source_pages"):
            print(f"[{q['id']}] expects an answer but lists no source pages"); problems += 1
    print(f"{len(questions)} questions checked, {problems} problem(s)")
    return problems


def run(questions, ks, alphas, show_misses):
    from embed import embed, get_mode
    from search import _store

    mode = get_mode()
    scored = [q for q in questions if q.get("source_pages")]
    vectors = embed([q["question"] for q in scored], mode)
    store = _store(mode)
    kmax = max(ks)

    for alpha in alphas:
        by_cat: dict[str, list[bool]] = defaultdict(list)
        overall = {k: 0 for k in ks}
        misses = []
        for q, vec in zip(scored, vectors):
            hits = store.hybrid_search(q["question"], vec, k=kmax, alpha=alpha, audience=q.get("audience"))
            pages = [h["page_id"] for h in hits]
            rank = next((i + 1 for i, p in enumerate(pages) if p in q["source_pages"]), None)
            for k in ks:
                overall[k] += bool(rank and rank <= k)
            by_cat[q["category"]].append(bool(rank and rank <= min(ks)))
            if not rank or rank > min(ks):
                misses.append((q, rank, pages[: min(ks)]))
        n = len(scored)
        print(f"\nalpha={alpha}  ({n} questions)  " + "  ".join(f"hit@{k}: {overall[k]}/{n} = {overall[k]/n:.0%}" for k in ks))
        print("  by category at k=%d: " % min(ks) + ", ".join(f"{c} {sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())))
        if show_misses:
            for q, rank, pages in misses:
                print(f"   miss [{q['id']}] want {q['source_pages']} | got {pages} | first correct rank: {rank}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate the question file and exit")
    ap.add_argument("--k", type=int, nargs="+", default=[5])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5])
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    questions = load()
    if args.check:
        sys.exit(1 if check(questions) else 0)
    run(questions, sorted(args.k), args.alpha, args.misses)


if __name__ == "__main__":
    main()
