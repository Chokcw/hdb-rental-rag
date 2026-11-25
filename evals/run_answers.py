"""Answer evals: run every eval question through answer.py and score the answers.

    python evals/run_answers.py                      # full run, results go to LangSmith
    python evals/run_answers.py --limit 5            # cheap trial run
    python evals/run_answers.py --only q15-max-tenants-4room-conflict q26-bedroom-fee-conflict
    python evals/run_answers.py --no-upload          # everything stays on your machine
    python evals/run_answers.py --alpha 0.5 --k 8    # try other retrieval settings

Metrics (all 0 to 1):
    status_ok         answered / declined as expected (STATUS line)
    citations_valid   no citations to sources that don't exist, none missing
    cites_right_page  a cited chunk comes from a page that holds the answer
    correct           LLM judge: matches the reference answer
    grounded          LLM judge: every claim is supported by the retrieved sources
    conflict_handled  LLM judge: used the official version and flagged the other source
    pass              strict: every applicable metric above (except cites_right_page) is 1

Each question lands in a LangSmith dataset; each run is an experiment you can compare side by side.
Set JUDGE_MODEL in .env to a different model from CHAT_MODEL so the judge is not grading its own work.
"""
from __future__ import annotations

import os
import sys

if "--no-upload" in sys.argv:  # must happen before tracing is imported
    os.environ["LANGSMITH_TRACING"] = "false"

import argparse
import json
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import warnings  # noqa: E402

import yaml  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*")

DATASET = "hdb-rental-rag-eval"
NS = uuid.UUID("0d5f3c52-7c1e-4e55-9b8b-0f6a2d9e7a11")  # makes example ids stable across runs
QUESTIONS = ROOT / "evals" / "questions.yaml"
RESULTS_DIR = ROOT / "evals" / "results"
METRICS = ["pass", "status_ok", "citations_valid", "cites_right_page", "correct", "grounded", "conflict_handled"]

INCLUDE_SQL = False


# ------------------------------------------------------------------ scoring

def _res(key: str, score: float, comment: str = "") -> dict:
    return {"key": key, "score": score, "comment": comment}


def score_answer(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    from judges import judge_conflict, judge_correct, judge_grounded

    cat, expect = reference_outputs["category"], reference_outputs["expect"]
    status = outputs.get("status", "unknown")
    question, reference = inputs["question"], reference_outputs["expected"]
    text = outputs.get("answer", "")
    results: list[dict] = []

    # status
    if cat == "no_answer":
        ok = 1.0 if status in ("partial", "not_covered") else 0.0
    elif expect in ("decline", "route_sql"):
        ok = 1.0 if status == "not_covered" else 0.0
    else:
        ok = {"answered": 1.0, "partial": 0.5}.get(status, 0.0)
    results.append(_res("status_ok", ok, f"status={status}"))

    problems = outputs.get("problems", [])
    results.append(_res("citations_valid", 0.0 if problems else 1.0, "; ".join(problems)))

    applicable = [ok, 0.0 if problems else 1.0]

    if expect == "answer":
        wanted = set(reference_outputs.get("source_pages", []))
        got = {c["page_id"] for c in outputs.get("cited", [])}
        results.append(_res("cites_right_page", 1.0 if wanted & got else 0.0,
                            f"cited {sorted(got)}; want one of {sorted(wanted)}"))

        if status == "not_covered":
            correct, why = 0.0, "model said the sources do not cover it"
        else:
            correct, why = judge_correct(question, reference, text)
        if correct is not None:
            results.append(_res("correct", correct, why))
            applicable.append(correct)

        if status in ("answered", "partial"):
            grounded, why = judge_grounded(outputs.get("sources", []), text)
            if grounded is not None:
                results.append(_res("grounded", grounded, why))
                applicable.append(grounded)

        if cat == "conflict":
            if status == "not_covered":
                handled, why = 0.0, "model said the sources do not cover it"
            else:
                handled, why = judge_conflict(question, reference, text)
            if handled is not None:
                results.append(_res("conflict_handled", handled, why))
                applicable.append(handled)

    results.append(_res("pass", 1.0 if all(a == 1.0 for a in applicable) else 0.0))
    return {"results": results}


# ------------------------------------------------------------------ data

def load_questions(args) -> list[dict]:
    qs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if not INCLUDE_SQL:
        qs = [q for q in qs if q["category"] != "route_sql"]
    if args.only:
        missing = set(args.only) - {q["id"] for q in qs}
        if missing:
            raise SystemExit(f"Unknown question id(s): {sorted(missing)}")
        qs = [q for q in qs if q["id"] in args.only]
    if args.category:
        qs = [q for q in qs if q["category"] in args.category]
    return qs[: args.limit] if args.limit else qs


def example_parts(q: dict):
    inputs = {"question": q["question"], "audience": q.get("audience")}
    reference = {"expected": q["expected"], "expect": q["expect"], "category": q["category"],
                 "source_pages": q.get("source_pages", []), "evidence": q.get("evidence", ""),
                 "notes": q.get("notes", "")}
    meta = {"id": q["id"], "category": q["category"]}
    return str(uuid.uuid5(NS, q["id"])), inputs, reference, meta


def sync_dataset(client, name: str, all_questions: list[dict]):
    """Create the dataset if needed and upsert every question (ids are stable, so edits update in place)."""
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description="HDB rental RAG eval questions (evals/questions.yaml)")
    parts = [example_parts(q) for q in all_questions]  # (id, inputs, reference, metadata)
    have = {str(e.id) for e in client.list_examples(dataset_name=name)}
    new = [p for p in parts if str(p[0]) not in have]
    old = [p for p in parts if str(p[0]) in have]
    if new:
        kwargs = dict(dataset_name=name, ids=[p[0] for p in new], inputs=[p[1] for p in new],
                      metadata=[p[3] for p in new])
        refs = [p[2] for p in new]
        try:
            client.create_examples(reference_outputs=refs, **kwargs)
        except (TypeError, ValueError):  # older langsmith versions call it `outputs`
            client.create_examples(outputs=refs, **kwargs)
    for id_, inputs, ref, meta in old:  # edits to questions.yaml update in place
        client.update_example(example_id=id_, inputs=inputs, outputs=ref, metadata=meta)


# ------------------------------------------------------------------ reporting

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def summarise(rows: list[dict]):
    by_metric: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m, v in r["scores"].items():
            by_metric[m].append(v["score"])
            by_cat[r["category"]][m].append(v["score"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n{len(rows)} questions\n")
    print(f"{'metric':<18}{'mean':>7}{'n':>5}")
    for m in METRICS:
        if by_metric[m]:
            print(f"{m:<18}{mean(by_metric[m]):>7.2f}{len(by_metric[m]):>5}")
    print("\npass rate by category")
    for c, ms in sorted(by_cat.items()):
        print(f"  {c:<14}{mean(ms['pass']):.2f}  ({len(ms['pass'])})")

    fails = [(r, m, v) for r in rows for m, v in r["scores"].items() if m != "cites_right_page" and v["score"] < 1]
    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"\n{len(errors)} question(s) crashed:")
        for r in errors:
            print(f"  [{r['id']}] {r['error'][:200]}")
    if fails:
        print("\nwhere it fell short (first 25):")
        for r, m, v in fails[:25]:
            if m == "pass":
                continue
            print(f"  [{r['id']}] {m}={v['score']}  {v['comment'][:140]}")


# ------------------------------------------------------------------ main

class Progress:
    """Prints a line as each question is answered and scored, plus a heartbeat so a slow run never looks frozen."""

    def __init__(self, total: int):
        import threading
        self.total, self.answered, self.scored, self.errors = total, 0, 0, 0
        self.t0, self.lock, self.done = time.time(), threading.Lock(), threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()

    def _say(self, msg: str):
        print(f"[{time.time() - self.t0:5.0f}s] {msg}", flush=True)

    def _beat(self):
        while not self.done.wait(20):
            with self.lock:
                self._say(f"...still working: answered {self.answered}/{self.total}, scored {self.scored}/{self.total}")

    def answered_one(self, question: str, secs: float, status: str):
        with self.lock:
            self.answered += 1
            self._say(f"answered {self.answered}/{self.total} ({secs:.0f}s, {status})  {question[:60]}")

    def failed_one(self, question: str, err: Exception):
        with self.lock:
            self.errors += 1
            self._say(f"ERROR answering '{question[:50]}': {err}")

    def scored_one(self):
        with self.lock:
            self.scored += 1
            self._say(f"scored   {self.scored}/{self.total}")

    def finish(self):
        self.done.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=None, help="chunks sent to the model (default: answer.py's)")
    ap.add_argument("--alpha", type=float, default=None, help="keyword/vector mix (default: search.py's)")
    ap.add_argument("--no-audience", action="store_true", help="do not pass each question's owner/tenant hint")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--category", nargs="+")
    ap.add_argument("--include-sql", action="store_true", help="also run the rent-data questions (no router yet)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--no-upload", action="store_true", help="keep everything local; nothing is sent to LangSmith")
    ap.add_argument("--prefix", default="answers")
    args = ap.parse_args()

    global INCLUDE_SQL
    INCLUDE_SQL = args.include_sql

    import config  # noqa: F401
    from answer import DEFAULT_K, answer
    from embed import get_mode
    from llm import chat_model, judge_model
    from search import DEFAULT_ALPHA
    import tracing

    k = args.k or DEFAULT_K
    alpha = DEFAULT_ALPHA if args.alpha is None else args.alpha
    get_mode()  # fail fast if there is no embeddings key
    if not args.no_upload and not tracing.enabled():
        raise SystemExit("LangSmith tracing is off. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env, "
                         "or use --no-upload to run locally.")

    questions = load_questions(args)
    if not questions:
        raise SystemExit("No questions selected.")
    print(f"chat={chat_model()}  judge={judge_model()}  k={k}  alpha={alpha}  questions={len(questions)}")
    if chat_model() == judge_model():
        print("note: judge model == chat model; set JUDGE_MODEL in .env to grade with a different model.")

    progress = Progress(len(questions))

    def target(inputs: dict) -> dict:
        aud = None if args.no_audience else inputs.get("audience")
        t = time.time()
        try:
            out = answer(inputs["question"], k=k, alpha=alpha, audience=aud)
        except Exception as e:
            progress.failed_one(inputs["question"], e)
            raise
        progress.answered_one(inputs["question"], time.time() - t, out.get("status", "?"))
        return out

    def scorer(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        try:
            return score_answer(inputs, outputs, reference_outputs)
        finally:
            progress.scored_one()

    from langsmith import evaluate
    meta = {"alpha": alpha, "k": k, "chat_model": chat_model(), "judge_model": judge_model(),
            "commit": git_commit(), "audience_hint": not args.no_audience}
    prefix = f"{args.prefix}-a{alpha}-k{k}"
    upload = not args.no_upload

    client = None
    if upload:
        from langsmith import Client
        client = Client()
        all_q = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
        print("Syncing questions to LangSmith...", flush=True)
        sync_dataset(client, args.dataset, all_q)
        ids = [example_parts(q)[0] for q in questions]
        data = list(client.list_examples(dataset_name=args.dataset, example_ids=ids))
    else:
        from langsmith import schemas
        ds_id = uuid.uuid4()
        data = []
        for q in questions:
            eid, inputs, ref, m = example_parts(q)
            data.append(schemas.Example(id=eid, dataset_id=ds_id, inputs=inputs, outputs=ref, metadata=m))

    warnings.filterwarnings("ignore", category=DeprecationWarning)  # langsmith re-enables these on import
    warnings.filterwarnings("ignore", message=".*beta.*")
    kw = dict(data=data, evaluators=[scorer], experiment_prefix=prefix, metadata=meta,
              max_concurrency=args.concurrency)
    if client:
        kw["client"] = client
    print(f"Starting: {len(questions)} questions, concurrency {args.concurrency}. A progress line prints as each one finishes.", flush=True)
    try:
        results = evaluate(target, upload_runs=upload, **kw)
    except TypeError:  # older langsmith: parameter was called upload_results
        results = evaluate(target, upload_results=upload, **kw)

    rows = []
    for r in results:
        ex, run = r["example"], r["run"]
        scores = {x.key if hasattr(x, "key") else x["key"]:
                  {"score": x.score if hasattr(x, "score") else x["score"],
                   "comment": (x.comment if hasattr(x, "comment") else x.get("comment")) or ""}
                  for x in r["evaluation_results"]["results"] if (x.score if hasattr(x, "score") else x["score"]) is not None}
        out = run.outputs or {}
        rows.append({"id": (ex.metadata or {}).get("id"), "category": (ex.metadata or {}).get("category"),
                     "question": (ex.inputs or {}).get("question"), "status": out.get("status"),
                     "answer": out.get("answer"), "cited": out.get("cited"), "problems": out.get("problems"),
                     "usage": out.get("usage"), "scores": scores, "error": run.error})
    rows = [r for r in rows if r["scores"] or r["error"]]

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{prefix}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "config": meta}, ensure_ascii=False) + "\n")

    progress.finish()
    summarise([r for r in rows if r["scores"]])
    print(f"\nsaved {path.relative_to(ROOT)}")
    if upload:
        print(f"experiment: {getattr(results, 'experiment_name', prefix)}  "
              f"(LangSmith > Datasets & Experiments > {args.dataset})")


if __name__ == "__main__":
    main()
