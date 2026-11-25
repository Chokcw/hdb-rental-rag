"""Question in, cited answer out.

    python answer.py "Can a PR rent out their whole HDB flat?"
    python answer.py "How many tenants in a 4-room flat?" --audience owner --k 6 --alpha 0.7
    python answer.py "..." --dry-run      # show the exact prompt, don't call the model
    python answer.py "..." --json         # machine-readable result (what the evals use)

Steps (each one is a span in LangSmith):
    retrieve -> build_prompt -> generate -> check_citations
"""
from __future__ import annotations

import argparse
import json
import re
from urllib.parse import urlparse

from llm import ChatError, chat
from search import DEFAULT_ALPHA, retrieve
from tracing import trace

DEFAULT_K = 6

SYSTEM_PROMPT = """You answer questions about renting HDB flats in Singapore, using ONLY the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Do not use outside knowledge, even if you think you know the answer.
2. Cite every factual claim with its source number in square brackets, like [1] or [2][3]. Never cite a number that is not in the list.
3. Prefer the official owner of the topic: HDB for flat rental rules, IRAS for tax and stamp duty. If two sources disagree, give the owner's version and add one sentence saying that the other source (name it) differs and may be out of date. Only mention this when sources actually conflict; never write that a source is "the official owner" or describe this rule.
4. If the sources do not contain the answer, say so plainly. Do not guess and do not apply HDB rules to private property or other situations the sources do not cover. Say what the sources do cover if that helps, and who to ask (HDB, IRAS or CEA).
5. For calculations, show the working using figures from the sources.
6. Include every condition, exception, threshold and time limit in the sources that affects the answer (for example an exception for older flats, or a cap that reverts after a date). Do not add conclusions the sources do not state, including inferences about how one rule limits or overrides another.
7. Some rules have end dates (for example "beyond 31 December 2028"). State the date.
8. Be concise and plain: a short answer first, then bullets only if they help. No legal advice disclaimers beyond what the sources say.

Output format. The first line must be exactly one of:
STATUS: answered      (the sources fully answer the question)
STATUS: partial       (the sources answer only part of it)
STATUS: not_covered   (the sources do not answer it)
Then a blank line, then your answer."""

PUBLISHERS = {"hdb.gov.sg": "HDB", "iras.gov.sg": "IRAS", "cea.gov.sg": "CEA",
              "mnd.gov.sg": "MND", "gov.sg": "gov.sg", "ask.gov.sg": "gov.sg"}


def publisher(url: str) -> str:
    host = (urlparse(url).hostname or "").removeprefix("www.")
    for suffix, name in PUBLISHERS.items():
        if host == suffix or host.endswith("." + suffix):
            return name
    return host or "unknown"


@trace("build_prompt", run_type="chain")
def build_prompt(question: str, hits: list[dict]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        where = f"{h['title']} > {h['heading_path']}" if h["heading_path"] else h["title"]
        blocks.append(f"[{i}] {publisher(h['url'])} - {where}\n"
                      f"URL: {h['url']}\nLast checked: {h['fetched_at'][:10]}\n\n{h['text']}")
    user = "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


STATUS_RE = re.compile(r"^\s*STATUS:\s*(answered|partial|not_covered)\s*\n?", re.I)
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_status(text: str) -> tuple[str, str]:
    m = STATUS_RE.match(text)
    if not m:
        return "unknown", text.strip()
    return m.group(1).lower(), text[m.end():].strip()


@trace("check_citations", run_type="tool")
def check_citations(answer: str, n_sources: int) -> dict:
    cited = [int(n) for grp in CITE_RE.findall(answer) for n in re.split(r"\s*,\s*", grp)]
    return {"cited": sorted(set(c for c in cited if 1 <= c <= n_sources)),
            "invalid": sorted(set(c for c in cited if not 1 <= c <= n_sources))}


@trace("answer", run_type="chain")
def answer(question: str, k: int = DEFAULT_K, alpha: float = DEFAULT_ALPHA,
           audience: str | None = None, dry_run: bool = False) -> dict:
    hits = retrieve(question, k=k, alpha=alpha, audience=audience)
    messages = build_prompt(question, hits)
    if dry_run:
        return {"question": question, "messages": messages, "sources": hits}

    raw, usage = generate(messages)
    status, text = parse_status(raw)
    cites = check_citations(text, len(hits))
    problems = []
    if cites["invalid"]:
        problems.append(f"cites sources that do not exist: {cites['invalid']}")
    if status in ("answered", "partial") and not cites["cited"]:
        problems.append("answer has no citations")
    if status == "unknown":
        problems.append("model did not start with a STATUS line")
    return {
        "question": question, "status": status, "answer": text,
        "sources": [{"n": i, "chunk_id": h["chunk_id"], "page_id": h["page_id"], "publisher": publisher(h["url"]),
                     "title": h["title"], "heading_path": h["heading_path"], "url": h["url"],
                     "last_checked": h["fetched_at"][:10], "text": h["text"]} for i, h in enumerate(hits, 1)],
        "cited": [{"n": n, "page_id": hits[n - 1]["page_id"], "title": hits[n - 1]["title"],
                   "heading_path": hits[n - 1]["heading_path"],
                   "url": hits[n - 1]["url"], "publisher": publisher(hits[n - 1]["url"]),
                   "last_checked": hits[n - 1]["fetched_at"][:10]} for n in cites["cited"]],
        "problems": problems, "retrieved": [h["chunk_id"] for h in hits], "usage": usage,
    }


@trace("generate", run_type="chain")
def generate(messages: list[dict]) -> tuple[str, dict]:
    return chat(messages)


def render(r: dict) -> str:
    out = [f"[{r['status']}]", "", r["answer"]]
    if r["cited"]:
        out += ["", "Sources:"]
        out += [f"  [{c['n']}] {c['publisher']}: {c['title']}"
                + (f" > {c['heading_path']}" if c["heading_path"] else "")
                + f" (checked {c['last_checked']})\n      {c['url']}" for c in r["cited"]]
    if r["problems"]:
        out += ["", "Check: " + "; ".join(r["problems"])]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--audience", choices=["owner", "tenant"])
    ap.add_argument("--dry-run", action="store_true", help="print the prompt instead of calling the model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        r = answer(args.question, args.k, args.alpha, args.audience, args.dry_run)
    except ChatError as e:
        raise SystemExit(str(e))
    if args.dry_run:
        for m in r["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}\n")
    elif args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(render(r))


if __name__ == "__main__":
    main()
