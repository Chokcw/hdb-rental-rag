"""Load the data.gov.sg HDB rental datasets in seeds.yaml into SQLite.

Pages through the datastore_search API, retrying with backoff on rate limits,
and replaces each table on every run (the datasets are small).

  python fetch_datasets.py            # writes data/rentals.db
  sqlite3 data/rentals.db "select town, flat_type, median(...)"  # query with SQL
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import yaml

API = "https://data.gov.sg/api/action/datastore_search"
PAGE = 10_000


def fetch_all(client: httpx.Client, dataset_id: str) -> tuple[list[str], list[dict]]:
    rows: list[dict] = []
    fields: list[str] = []
    offset = 0
    while True:
        for attempt in range(6):
            r = client.get(API, params={"resource_id": dataset_id, "limit": PAGE, "offset": offset})
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"  rate limited; waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError("gave up after repeated rate limiting")

        result = r.json()["result"]
        if not fields:
            fields = [f["id"] for f in result["fields"] if f["id"] != "_id"]
        batch = result["records"]
        rows.extend(batch)
        print(f"  {len(rows)} / {result.get('total', '?')} rows")
        if len(batch) < PAGE:
            return fields, rows
        offset += PAGE
        time.sleep(1)


def write_table(db: sqlite3.Connection, table: str, fields: list[str], rows: list[dict]) -> None:
    cols = ", ".join(f'"{f}"' for f in fields)
    db.execute(f'DROP TABLE IF EXISTS "{table}"')
    db.execute(f'CREATE TABLE "{table}" ({cols})')
    db.executemany(
        f'INSERT INTO "{table}" VALUES ({", ".join("?" for _ in fields)})',
        ([row.get(f) for f in fields] for row in rows),
    )
    db.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=Path, default=Path("seeds.yaml"))
    ap.add_argument("--db", type=Path, default=Path("data/rentals.db"))
    args = ap.parse_args()

    datasets = yaml.safe_load(args.seeds.read_text())["datasets"]
    args.db.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(args.db)
    failures = 0
    with httpx.Client(timeout=60) as client:
        for ds in datasets:
            print(f"{ds['id']} ({ds['dataset_id']})")
            try:
                fields, rows = fetch_all(client, ds["dataset_id"])
                write_table(db, ds["id"], fields, rows)
                print(f"  wrote {len(rows)} rows to {ds['id']}")
            except (httpx.HTTPError, RuntimeError, KeyError) as e:
                failures += 1
                print(f"  ! {e}", file=sys.stderr)
    db.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
