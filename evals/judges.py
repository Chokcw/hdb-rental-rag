"""LLM judges for the answer evals.

Each judge returns (score, comment). Scores are 1, 0.5 or 0, or None if the judge's
reply could not be read (so a flaky judge never silently counts as a failure).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import chat, judge_model  # noqa: E402

SYSTEM = ("You are a strict, fair grader for a question-answering system about renting HDB flats "
          "in Singapore. Reply with a single JSON object only: {\"verdict\": \"<one of the allowed words>\", "
          "\"reason\": \"<one short sentence>\"}.")

CORRECT = """TASK: correctness
Compare the candidate answer with the reference answer.
Allowed verdicts: correct, partial, wrong.
- correct: states all the key facts of the reference without contradicting it. Extra detail and different wording are fine.
- partial: states some key facts but misses or blurs an important one (for example a condition, number or date).
- wrong: contradicts the reference, gives the opposite answer, or says it cannot answer.
Ignore citation markers like [1] and formatting.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}"""

GROUNDED = """TASK: groundedness
Check whether the candidate answer is supported by the numbered sources below.
Allowed verdicts: grounded, partial, ungrounded.
- grounded: every factual claim in the answer is stated in the sources.
- partial: mostly supported, but at least one claim (a number, date, condition or rule) is not in the sources.
- ungrounded: key claims are missing from the sources or contradict them.
Do not penalise statements about what the sources do not cover. Ignore citation markers and formatting.

Sources:
{sources}

Candidate answer: {answer}"""

CONFLICT = """TASK: conflict handling
The retrieved sources contain two versions of one fact. The reference answer gives the official (correct) version.
Allowed verdicts: good, partial, bad.
- good: the candidate gives the reference version AND says that another source differs or may be out of date.
- partial: gives the reference version but does not mention that another source differs.
- bad: gives the other version, gives both without choosing, or says it cannot answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}"""

SCORES = {"correct": 1.0, "grounded": 1.0, "good": 1.0, "partial": 0.5,
          "wrong": 0.0, "ungrounded": 0.0, "bad": 0.0}


def _ask(prompt: str) -> tuple[float | None, str]:
    text, _ = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=2000, model=judge_model())  # headroom: reasoning models spend tokens thinking
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        verdict = str(obj.get("verdict", "")).strip().lower()
        return SCORES[verdict], f"{verdict}: {obj.get('reason', '')}".strip()
    except (ValueError, KeyError):
        return None, f"unreadable judge reply: {text[:120]!r}"


def judge_correct(question: str, reference: str, answer: str):
    return _ask(CORRECT.format(question=question, reference=reference, answer=answer))


def judge_grounded(sources: list[dict], answer: str):
    block = "\n\n".join(f"[{s['n']}] {s['publisher']} - {s['title']}"
                        + (f" > {s['heading_path']}" if s["heading_path"] else "") + f"\n{s['text']}"
                        for s in sources)
    return _ask(GROUNDED.format(sources=block, answer=answer))


def judge_conflict(question: str, reference: str, answer: str):
    return _ask(CONFLICT.format(question=question, reference=reference, answer=answer))
