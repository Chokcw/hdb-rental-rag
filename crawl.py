"""Polite crawler for the HDB rental RAG corpus.

Reads seeds.yaml, fetches each page (respecting robots.txt), extracts the main
content as markdown with tables kept intact, and writes:

  corpus/<id>.md          markdown with YAML front matter (url, title, hash, fetched_at, ...)
  corpus/manifest.jsonl   one line per page per run: status, hash, changed flag
  corpus/raw/<id>.html    raw HTML, for debugging extraction

Only pages whose content hash changed since the last run are flagged `changed`,
so the embedding step can re-embed just those.

Usage:
  python crawl.py --local manual     # ingest pages you saved by hand into manual/
  python crawl.py                    # crawl everything
  python crawl.py --only hdb-tenant-eligibility
  python crawl.py --tier 1 --delay 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

# Contact goes in the User-Agent so site operators can reach you. Read from the environment so it
# never lands in the repo:  export CRAWLER_CONTACT=you@example.com
CONTACT = os.environ.get("CRAWLER_CONTACT", "").strip()
USER_AGENT = f"hdb-rental-rag-research/0.1 (personal portfolio project; contact: {CONTACT or 'unset'})"
OUT = Path("corpus")
MIN_TEXT_CHARS = 400  # below this, the page is probably rendered client-side

# Elements that are never content
STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe"]
STRIP_SELECTORS = [
    "header", "footer", "nav", "aside",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
    ".breadcrumb", ".breadcrumbs", ".masthead", ".sgds-masthead",
    ".skip-link", ".cookie", ".share", ".social",
    "[class*=Sidebar]", "[class*=Breadcrumb]", "[class*=placeholder]",  # CMS side menus, trails, loading skeletons
]
# Tried in order to find the main content block
MAIN_SELECTORS = ["main", "[role=main]", "article", "#main-content", "#content", ".content"]


def load_seeds(path: Path) -> list[dict]:
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {})
    return [{**defaults, **p} for p in cfg["pages"]]


class Robots:
    """Caches one robots.txt parser per host."""

    def __init__(self, client: httpx.Client):
        self.client = client
        self.cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if host not in self.cache:
            rp = RobotFileParser()
            try:
                r = self.client.get(host + "/robots.txt")
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    self.cache[host] = rp
                elif r.status_code in (401, 403):
                    rp.disallow_all = True  # standard convention: access-denied robots.txt = disallow
                    self.cache[host] = rp
                else:
                    self.cache[host] = None  # no robots.txt -> allowed
            except httpx.HTTPError as e:
                print(f"  ! robots.txt unreachable for {host}: {e}; skipping host", file=sys.stderr)
                rp.disallow_all = True
                self.cache[host] = rp
        rp = self.cache[host]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


def extract(html: str) -> tuple[str, str]:
    """Return (title, markdown) for the page's main content."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find("h1") or soup.title or soup.new_tag("x")).get_text(" ", strip=True)

    for tag in soup(STRIP_TAGS):
        tag.decompose()

    # Component libraries often build headings as <div role="heading" aria-level="3">
    for el in soup.select("[role=heading]"):
        level = el.get("aria-level", "2")
        el.name = f"h{level}" if level in {"1", "2", "3", "4", "5", "6"} else "h2"
        el.attrs = {}

    # Forms and buttons: drop bare controls, but keep their text when substantial. CMS pages can wrap
    # the whole page in a <form>, and accordion section headers are <button>s holding real text.
    for el in soup.find_all(["form", "button"]):
        if len(el.get_text(strip=True)) < 25:
            el.decompose()
        else:
            el.unwrap()

    main = None
    for sel in MAIN_SELECTORS:
        main = soup.select_one(sel)
        if main and len(main.get_text(strip=True)) > MIN_TEXT_CHARS:
            break
        main = None
    main = main or soup.body or soup

    for sel in STRIP_SELECTORS:
        for el in main.select(sel):
            el.decompose()

    md = markdownify(str(main), heading_style="ATX", bullets="-", strip=["img"])
    md = re.sub(r"\n{3,}", "\n\n", md)              # collapse blank runs
    md = re.sub(r"[ \t]+\n", "\n", md).strip() + "\n"  # trailing spaces
    return title, md


def empty_collapsed_sections(html: str) -> int:
    """Count accordion sections whose panel has no text: the page was saved with them closed."""
    soup = BeautifulSoup(html, "html.parser")
    n = 0
    for btn in soup.select("button[aria-controls][aria-expanded]"):
        panel = soup.find(id=btn["aria-controls"])
        if panel is not None and "accordion" in " ".join(btn.get("class", [])) and not panel.get_text(strip=True):
            n += 1
    return n


def content_hash(md: str) -> str:
    # Normalise whitespace so cosmetic changes don't trigger re-embedding
    return hashlib.sha256(re.sub(r"\s+", " ", md).strip().encode()).hexdigest()[:16]


def previous_hashes() -> dict[str, str]:
    manifest = OUT / "manifest.jsonl"
    last: dict[str, str] = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("hash"):
                last[rec["id"]] = rec["hash"]
    return last


def write_page(seed: dict, final_url: str, title: str, md: str, h: str, fetched_at: str,
               source: str = "crawl") -> None:
    front = {
        "id": seed["id"],
        "url": seed["url"],
        "final_url": final_url,
        "title": title,
        "tier": seed["tier"],
        "audience": seed["audience"],
        "canonical": seed["canonical"],
        "source": source,  # "crawl" or "manual"; for manual pages fetched_at is when the file was saved
        "hash": h,
        "fetched_at": fetched_at,
    }
    (OUT / f"{seed['id']}.md").write_text("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n" + md)


def crawl(seeds: list[dict], delay: float) -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    prev = previous_hashes()
    seen_hashes: dict[str, str] = {}  # hash -> first id, to flag duplicate content
    failures = 0

    with httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-SG,en;q=0.9"},
        follow_redirects=True,
        timeout=30,
    ) as client, (OUT / "manifest.jsonl").open("a") as manifest:
        robots = Robots(client)
        for i, seed in enumerate(seeds):
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rec = {"id": seed["id"], "url": seed["url"], "fetched_at": fetched_at}
            print(f"[{i + 1}/{len(seeds)}] {seed['id']}")

            if not robots.allowed(seed["url"]):
                rec["status"] = "robots_disallowed"
                print("  - disallowed by robots.txt")
            else:
                try:
                    r = client.get(seed["url"])
                    r.raise_for_status()
                    (OUT / "raw" / f"{seed['id']}.html").write_text(r.text)
                    title, md = extract(r.text)
                    h = content_hash(md)
                    rec.update(
                        status="ok",
                        http=r.status_code,
                        final_url=str(r.url),
                        hash=h,
                        chars=len(md),
                        changed=prev.get(seed["id"]) != h,
                        duplicate_of=seen_hashes.get(h),
                    )
                    seen_hashes.setdefault(h, seed["id"])
                    if len(md) < MIN_TEXT_CHARS:
                        rec["status"] = "thin_content"
                        print(f"  ! only {len(md)} chars; page may need a headless browser")
                    write_page(seed, str(r.url), title, md, h, fetched_at)
                    flag = "changed" if rec["changed"] else "unchanged"
                    dup = f", duplicate of {rec['duplicate_of']}" if rec["duplicate_of"] else ""
                    print(f"  ok {len(md)} chars, {flag}{dup}")
                except httpx.HTTPError as e:
                    rec["status"] = "error"
                    rec["error"] = str(e)
                    failures += 1
                    print(f"  ! {e}", file=sys.stderr)

            manifest.write(json.dumps(rec) + "\n")
            if i < len(seeds) - 1:
                time.sleep(delay)
    return failures


LOCAL_EXTS = (".html", ".htm", ".md", ".txt")


def ingest_local(seeds: list[dict], folder: Path, all_ids: set[str]) -> int:
    """Ingest pages you saved by hand (for sites that block scripted access).

    Looks in `folder` for <seed id>.html|.htm|.md|.txt. HTML goes through the same
    extraction as crawled pages; .md/.txt are used as-is. The file's modified time
    is recorded as fetched_at, and source is marked "manual".
    """
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    OUT.mkdir(exist_ok=True)
    prev = previous_hashes()
    seen_hashes: dict[str, str] = {}
    problems = 0

    # Files that don't match any seed id are usually typos in the filename
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in LOCAL_EXTS and f.stem not in all_ids:
            print(f"! {f.name} matches no seed id; check the filename")
            problems += 1

    missing: list[str] = []
    with (OUT / "manifest.jsonl").open("a") as manifest:
        for seed in seeds:
            path = next((p for ext in LOCAL_EXTS if (p := folder / f"{seed['id']}{ext}").exists()), None)
            if path is None:
                if seed["canonical"] and seed["tier"] == 1:
                    missing.append(seed["id"])
                continue

            saved_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            text = path.read_text(errors="replace")
            if path.suffix.lower() in (".html", ".htm"):
                title, md = extract(text)
            else:
                m = re.search(r"^#\s+(.+)$", text, re.M)
                title, md = (m.group(1) if m else seed["id"]), text.strip() + "\n"

            h = content_hash(md)
            rec = {
                "id": seed["id"], "url": seed["url"], "fetched_at": saved_at, "source": "manual",
                "file": path.name, "status": "ok", "hash": h, "chars": len(md),
                "changed": prev.get(seed["id"]) != h, "duplicate_of": seen_hashes.get(h),
            }
            seen_hashes.setdefault(h, seed["id"])
            closed = empty_collapsed_sections(text) if path.suffix.lower() in (".html", ".htm") else 0
            if closed:
                rec["status"] = "collapsed_sections"
                problems += 1
                print(f"! {seed['id']}: {closed} section(s) were closed when saved, so their text is missing. "
                      f"Open every section (click each heading), then save again with 'Webpage, Complete'.")
            elif len(md) < MIN_TEXT_CHARS:
                rec["status"] = "thin_content"
                problems += 1
                print(f"! {seed['id']}: only {len(md)} chars. Re-save with 'Webpage, Complete' or paste the text into {seed['id']}.md")
            write_page(seed, seed["url"], title, md, h, saved_at, source="manual")
            flag = "changed" if rec["changed"] else "unchanged"
            print(f"ok {seed['id']}: {len(md)} chars, {flag}")
            manifest.write(json.dumps(rec) + "\n")

    if missing:
        print(f"\nNo local file yet for {len(missing)} tier 1 page(s): " + ", ".join(missing))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=Path("seeds.yaml"))
    ap.add_argument("--local", type=Path, metavar="DIR",
                    help="ingest hand-saved pages from DIR (<seed id>.html/.md) instead of fetching")
    ap.add_argument("--only", nargs="*", help="crawl only these seed ids")
    ap.add_argument("--tier", type=int, nargs="*", help="crawl only these tiers")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests (default 2)")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    all_ids = {s["id"] for s in seeds}
    if args.only:
        seeds = [s for s in seeds if s["id"] in set(args.only)]
    if args.tier:
        seeds = [s for s in seeds if s["tier"] in set(args.tier)]
    if not seeds:
        sys.exit("No seeds matched.")

    if args.local:
        sys.exit(1 if ingest_local(seeds, args.local, all_ids) else 0)

    if not CONTACT:
        sys.exit("Set CRAWLER_CONTACT first, e.g.  export CRAWLER_CONTACT=you@example.com  "
                 "(sent in the User-Agent so site operators can reach you; not stored in the repo).")

    failures = crawl(seeds, args.delay)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
