"""Split corpus/*.md into chunks ready for indexing.

Rules (kept deliberately simple):
  1. One chunk is roughly one heading section.
  2. Every chunk remembers its heading path, e.g. "Eligibility > Tenant conditions".
  3. Tiny sections are merged with the next section so no chunk is just a heading.
  4. Long sections are split at paragraph / list-item / table-row boundaries.
  5. A chunk's id is built from its page id + a hash of its text, so unchanged
     chunks keep the same id across re-crawls (and don't need re-embedding).

Usage:
    python chunker.py                 # print a report
    python chunker.py --show 5        # also print the first 5 chunks
    python chunker.py --out chunks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

CORPUS_DIR = Path("corpus")
MIN_PAGE_CHARS = 300   # pages with less body text than this are skipped (title-only pages)
MIN_CHUNK = 350        # sections shorter than this get merged into the next one
TARGET_CHUNK = 1500    # we try to stay under this
MAX_CHUNK = 2200       # hard ceiling for a chunk body

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
NAMESPACE = uuid.UUID("6f1c1c0e-6a8e-4e63-9a55-2f7d0b7f3a10")  # fixed, so ids are stable


@dataclass
class Chunk:
    chunk_id: str        # "<page id>:<hash of text>"
    uuid: str            # deterministic UUID of chunk_id (used as the Weaviate object id)
    page_id: str
    url: str
    title: str
    tier: int
    audience: str
    source: str
    heading_path: str    # "Eligibility > Tenant conditions"
    text: str            # the chunk body, shown to the model and the user
    embed_text: str      # what gets embedded: title + heading path + text
    chunk_hash: str      # hash of embed_text
    position: int        # order within the page
    fetched_at: str      # when the page was last checked
    n_chars: int


# ---------------------------------------------------------------- parsing

def read_page(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return {}, raw
    _, fm, body = raw.split("---", 2)
    return yaml.safe_load(fm) or {}, body.strip()


@dataclass
class Section:
    path: list[str]      # heading path
    heading_line: str    # the original heading line ("" for intro text)
    blocks: list[str]    # paragraphs / lists / tables

    @property
    def size(self) -> int:
        return sum(len(b) for b in self.blocks) + 2 * len(self.blocks)


def split_sections(body: str, title: str) -> list[Section]:
    """Walk the markdown line by line, tracking the heading path."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []          # (level, heading text)
    current = Section(path=[], heading_line="", blocks=[])
    buf: list[str] = []
    in_fence = False

    def flush_buf():
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            current.blocks.append(text)

    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush_buf()
            if current.blocks or current.heading_line:
                sections.append(current)
            level, text = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            # the page's own H1 repeats the title, so leave it out of the path
            path = [t for lv, t in stack if not (lv == 1 and t.lower() == title.lower())]
            current = Section(path=path, heading_line=line.strip(), blocks=[])
        elif not line.strip():
            flush_buf()
        else:
            buf.append(line)
    flush_buf()
    if current.blocks or current.heading_line:
        sections.append(current)
    return [s for s in sections if s.blocks]


# ---------------------------------------------------------------- splitting long blocks

def split_block(block: str, limit: int) -> list[str]:
    """Break one oversized block (list, table or paragraph) into pieces <= limit."""
    if len(block) <= limit:
        return [block]
    lines = block.splitlines()
    if lines and lines[0].lstrip().startswith("|"):                 # table: split by rows
        header, rows = lines[:2], lines[2:]
        pieces, cur = [], list(header)
        for row in rows:
            if len("\n".join(cur + [row])) > limit and len(cur) > len(header):
                pieces.append("\n".join(cur))
                cur = list(header)
            cur.append(row)
        pieces.append("\n".join(cur))
        return pieces
    if len(lines) > 1:                                              # list: split by item
        items: list[str] = []
        for ln in lines:
            if ln.startswith((" ", "\t")) and items:                # nested line stays with parent
                items[-1] += "\n" + ln
            else:
                items.append(ln)
        return pack(items, limit, sep="\n")
    sentences = re.split(r"(?<=[.!?])\s+", block)                   # long paragraph: by sentence
    return pack(sentences, limit, sep=" ")


def pack(parts: list[str], limit: int, sep: str) -> list[str]:
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(sep) + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}{sep}{p}" if cur else p
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- chunking a page

def section_text(sec: Section, with_heading: bool) -> str:
    parts = ([sec.heading_line] if with_heading and sec.heading_line else []) + sec.blocks
    return "\n\n".join(parts)


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def chunk_page(meta: dict, body: str) -> list[tuple[str, str]]:
    """Return [(heading_path, text), ...] for one page."""
    sections = split_sections(body, meta.get("title", ""))
    pieces: list[tuple[list[str], str]] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        path, text = sec.path, section_text(sec, with_heading=False)
        i += 1
        # merge forward while the chunk is tiny and the next section is a close relative
        while (len(text) < MIN_CHUNK and i < len(sections)
               and (not path or sections[i].path[:1] == path[:1] or not sections[i].path)
               and len(text) + len(section_text(sections[i], True)) <= TARGET_CHUNK):
            nxt = sections[i]
            text = f"{text}\n\n{section_text(nxt, with_heading=True)}".strip()
            path = common_prefix(path, nxt.path) if path else nxt.path[:1]
            i += 1
        # split if too long
        if len(text) <= MAX_CHUNK:
            pieces.append((path, text))
        else:
            blocks = [p for b in text.split("\n\n") for p in split_block(b, TARGET_CHUNK)]
            for part in pack(blocks, TARGET_CHUNK, sep="\n\n"):
                pieces.append((path, part))
    return [(" > ".join(p), t) for p, t in pieces if t.strip()]


def sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def seed_canonical(seeds_path: Path = Path("seeds.yaml")) -> dict[str, bool]:
    """seeds.yaml is the source of truth for 'canonical' (page files can be older)."""
    if not seeds_path.exists():
        return {}
    cfg = yaml.safe_load(seeds_path.read_text(encoding="utf-8")) or {}
    default = (cfg.get("defaults") or {}).get("canonical", True)
    return {p["id"]: p.get("canonical", default) for p in cfg.get("pages", [])}


def build_chunks(corpus_dir: Path = CORPUS_DIR):
    """Read every canonical page and return (chunks, skipped)."""
    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    canon = seed_canonical()
    for path in sorted(corpus_dir.glob("*.md")):
        meta, body = read_page(path)
        pid = meta.get("id", path.stem)
        if not canon.get(pid, meta.get("canonical", True)):
            skipped.append((pid, "not canonical"))
            continue
        if len(body) < MIN_PAGE_CHARS:
            skipped.append((pid, f"too little text ({len(body)} chars)"))
            continue
        seen: dict[str, int] = {}
        for pos, (heading_path, text) in enumerate(chunk_page(meta, body)):
            title = meta.get("title", pid)
            header = f"{title}\n{heading_path}" if heading_path else title
            embed_text = f"{header}\n\n{text}"
            cid = f"{pid}:{sha(embed_text)}"
            if cid in seen:                      # identical text twice on one page
                seen[cid] += 1
                cid = f"{cid}-{seen[cid]}"
            else:
                seen[cid] = 0
            chunks.append(Chunk(
                chunk_id=cid, uuid=str(uuid.uuid5(NAMESPACE, cid)), page_id=pid,
                url=meta.get("final_url") or meta.get("url", ""), title=title,
                tier=int(meta.get("tier", 0)), audience=meta.get("audience", "both"),
                source=meta.get("source", "crawl"), heading_path=heading_path,
                text=text, embed_text=embed_text, chunk_hash=sha(embed_text, 16),
                position=pos, fetched_at=str(meta.get("fetched_at", "")), n_chars=len(text),
            ))
    return chunks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--out", help="write chunks to this .jsonl file")
    ap.add_argument("--show", type=int, default=0, help="print the first N chunks")
    args = ap.parse_args()

    chunks, skipped = build_chunks(Path(args.corpus))
    print(f"{len(chunks)} chunks from {len({c.page_id for c in chunks})} pages")
    sizes = sorted(c.n_chars for c in chunks)
    if sizes:
        print(f"chunk size (chars): min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
    for pid, why in skipped:
        print(f"  skipped {pid}: {why}")
    for c in chunks[: args.show]:
        print(f"\n--- {c.chunk_id}  [{c.heading_path or '(intro)'}]  {c.n_chars} chars\n{c.text}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
