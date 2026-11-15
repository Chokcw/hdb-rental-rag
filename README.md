# HDB rental RAG — ingestion

Crawls official sources on HDB rentals into markdown for a RAG index, and loads
HDB's rental datasets from data.gov.sg into SQLite for price questions.

## Files

| File | What it does |
| --- | --- |
| `seeds.yaml` | 33 pages (tiers 1, 2, 4) and 3 datasets (tier 3), with audience, canonical flag and refresh cadence |
| `crawl.py` | Fetches pages politely, extracts main content to markdown (tables kept), tracks content hashes |
| `fetch_datasets.py` | Pages through the data.gov.sg API into `data/rentals.db` |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Check robots.txt by hand first: https://www.hdb.gov.sg/robots.txt
# 2. Set a contact for the User-Agent (kept out of the repo):
export CRAWLER_CONTACT=you@example.com
python crawl.py --only hdb-tenant-eligibility --delay 0   # smoke test one page
python crawl.py                                           # full crawl, 2 s between requests
python fetch_datasets.py                                  # rental datasets -> SQLite
```

## Sites that block scripted access (hdb.gov.sg)

hdb.gov.sg returns 403 to non-browser clients, so save those pages by hand instead:

1. Open each tier 1 canonical HDB page in your browser and **click every expandable section open**
   (HDB only loads a section's text when it is opened). Then save the page into `manual/` as `<seed id>.html`
   (e.g. `manual/hdb-tenant-eligibility.html`) using **Webpage, Complete**, not "HTML only". In Safari, use File > Save As > **Web Archive**
   (`<seed id>.webarchive` works too).
   Alternatively, with all sections open, copy the page text into `<seed id>.md`.
2. Run `python crawl.py --local manual`.

It reports filenames that match no seed, pages with too little text, and tier 1 pages you haven't saved yet.
Output records `source: manual` and the file's saved time as `fetched_at`.

## Output

- `corpus/<id>.md`: markdown with front matter (`url`, `title`, `tier`, `audience`, `canonical`, `hash`, `fetched_at`)
- `corpus/manifest.jsonl`: one record per page per run, with `status`, `changed` and `duplicate_of`
- `corpus/raw/<id>.html`: raw HTML, for debugging extraction

What to index: pages with `canonical: true`, `tier` 1 or 2, and `status: ok`. On later runs,
re-embed only records where `changed: true`.

## Status values

| Status | Meaning | Action |
| --- | --- | --- |
| `ok` | Fetched and extracted | Index if canonical |
| `thin_content` | Under 400 chars extracted | Probably rendered client-side; inspect `raw/`, or switch to Playwright |
| `robots_disallowed` | Blocked by robots.txt, or robots.txt unreachable or access-denied | Don't crawl; reconsider the source |
| `error` | HTTP or network error | Check the URL; HDB moves pages between URL trees |

## Known limitations

- Main-content selectors are generic (`main`, `article`, ...). Tune `MAIN_SELECTORS` and
  `STRIP_SELECTORS` after looking at the first real crawl's `raw/` HTML.
- Not tested against the live sites: the build environment couldn't reach them.
  It was tested end to end against a local mock site (tables, robots.txt, duplicates, change detection, 404s).
- ask.gov.sg question URLs are hand-picked; there's no discovery of further questions yet.
