# HDB rental RAG

Crawls official sources on HDB rentals into markdown, splits them into chunks, indexes
them in Weaviate for search, and loads HDB's rental datasets from data.gov.sg into
SQLite for price questions.

## Files

| File | What it does |
| --- | --- |
| `seeds.yaml` | 33 pages (tiers 1, 2, 4) and 3 datasets (tier 3), with audience, canonical flag and refresh cadence |
| `crawl.py` | Fetches pages politely, extracts main content to markdown (tables kept), tracks content hashes |
| `fetch_datasets.py` | Pages through the data.gov.sg API into `data/rentals.db` |
| `chunker.py` | Splits `corpus/*.md` into chunks by heading; keeps the heading path; stable chunk ids |
| `embed.py` | Text to vectors: `text-embedding-3-small` via OpenRouter or OpenAI, or `fake` for testing |
| `config.py` | Loads `.env` (see `.env.example`) |
| `store.py` | All Weaviate code (embedded or Docker). Swap this file to try another store |
| `index.py` | Chunk, embed only new/changed chunks, load into Weaviate |
| `search.py` | `retrieve(question, k, alpha, audience)` — hybrid keyword + meaning search |
| `llm.py` | Chat model via OpenRouter/OpenAI (`CHAT_MODEL` in `.env`) |
| `answer.py` | Question to cited answer: retrieve, build prompt, generate, check citations |
| `tracing.py` | LangSmith tracing in one file; does nothing unless switched on |
| `test_chunker.py` | Chunker tests (`python test_chunker.py`) |
| `evals/questions.yaml` | 46 eval questions with reference answers, source pages and quoted evidence |
| `evals/run_retrieval.py` | Validates the question file; scores retrieval hit rate by category and alpha |

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

## Index and search (Weaviate)

```bash
pip install -r requirements.txt
cp .env.example .env                # then put your OPENROUTER_API_KEY in .env (git ignores it)
python chunker.py --show 3          # look at the chunks first
python index.py                     # first run downloads the Weaviate binary (~165 MB)
python search.py "how many tenants can live in a 4-room flat" --alpha 0.5
```

- **Embedded mode** (default) runs Weaviate inside the Python process: Linux/macOS only,
  experimental, data in `data/weaviate/`, stops when the script exits. Don't run two
  scripts against the same data folder at once. For Docker instead:
  `WEAVIATE_MODE=docker` and start Weaviate yourself.
- `index.py` is safe to re-run. Unchanged chunks are not re-embedded; removed ones are deleted.
  `--rebuild` wipes and re-embeds everything.
- `alpha` controls the search mix: 0 = keyword only, 1 = meaning only. Compare values on the eval set.
- Provider is picked from your key: `OPENROUTER_API_KEY` (default) or `OPENAI_API_KEY`; force one
  with `EMBEDDINGS=openrouter|openai|fake`. Both serve the same model, so they share one Weaviate
  collection. If you change `EMBED_MODEL`, re-run `python index.py --rebuild`.
- `EMBEDDINGS=fake python index.py` tests the plumbing without any key (search quality is meaningless).
- `python index.py` skips pages that are not canonical in `seeds.yaml` or have almost no text.
  Currently skipped: `hdb-owner-flat-application`, `hdb-tenant-eligibility`, `mnd-flat-classification`.

### Answers

```bash
python answer.py "Can a PR rent out their whole HDB flat?"
python answer.py "..." --dry-run     # see the exact prompt, no model call
python answer.py "..." --json        # structured result (used by the evals)
```

The model answers only from the retrieved chunks, cites them as [1], [2], prefers the official
owner of a topic (HDB for flat rules, IRAS for tax) when sources disagree, and starts with a
`STATUS:` line (`answered`, `partial` or `not_covered`) so declines can be scored. Code, not the
model, builds the Sources list and checks that every cited number exists. Set `CHAT_MODEL` in
`.env` (default `openai/gpt-4o-mini`; any OpenRouter model id works).

### Tracing (LangSmith)

Add these to `.env` (or export them):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=hdb-rental-rag
```

Then run `python search.py "can a PR rent out their flat"`.

`python search.py` traces `retrieve` → `embed` + `weaviate_hybrid_search` (query, alpha, filter,
chunks with scores). `python answer.py` traces `answer` → `retrieve`, `build_prompt` (the exact
prompt), `generate` (model, tokens) and `check_citations`.

## Evals

```bash
python evals/run_retrieval.py --check                          # every quoted evidence line exists in the corpus
python evals/run_retrieval.py --alpha 0 0.3 0.5 0.7 1 --k 3 5 --misses
```

`questions.yaml` holds 46 questions: 33 facts, tables and computations, 2 source conflicts
(bedroom fee $9 vs $10; tenant cap date 2028 vs 2026), 2 multi-page questions, 2 that should
route to SQL, and 3 that should be declined or hedged. Each answerable question lists the pages
that contain the answer and a verbatim `evidence` quote. Reference answers are drafts; review
them before trusting scores. `q30` is a known gap until `hdb-owner-flat-application` is re-saved.
Answer-quality (LLM judge) and routing evals come after the answer step exists.

## Sites that block scripted access (hdb.gov.sg)

hdb.gov.sg returns 403 to non-browser clients, so save those pages by hand instead:

1. Open each tier 1 canonical HDB page in your browser and **click every expandable section open**
   (HDB only loads a section's text when it is opened). Then save the page into `manual/` as `<seed id>.html`
   (e.g. `manual/hdb-tenant-eligibility.html`) using **Webpage, Complete**, not "HTML only". In Safari, use File > Save As > **Web Archive**
   (`<seed id>.webarchive` works too).
   Alternatively, with all sections open, copy the page text into `<seed id>.md`.
2. Run `python crawl.py --local manual`.

### Easier: the in-browser helper (`tools/save_hdb_page.js`)

HDB's expandable sections close each other, so saving a page by hand misses text. The helper opens
each section in turn and downloads the page as `<seed id>.md`:

1. Open the HDB page in your browser.
2. Open the browser console (Chrome: Cmd+Option+J. Safari: turn on Settings > Advanced > "Show features
   for web developers", then Option+Cmd+C).
3. Paste the contents of `tools/save_hdb_page.js` and press Enter. (Chrome asks you to type `allow pasting`
   first.) Wait a few seconds; a file downloads and the console prints how many sections were captured.
4. Move the file: `mv ~/Downloads/hdb-*.md manual/`, then `python crawl.py --local manual`.

It sends no requests of its own; it only clicks the sections you could click yourself. If the same page
exists in `manual/` as both `.html` and `.md`, the most recently saved file wins.

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
