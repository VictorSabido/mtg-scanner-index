"""
Daily price table for the app, from Scryfall's `default_cards` bulk file.

Scryfall updates prices once a day and asks apps that look up many prices to use bulk data
instead of the API. The app downloads this file once a day and prices every card offline.

Output (gzip JSON):
  {"version": 1, "updated": "<bulk updated_at>",
   "sets": {"<set>": {"<collector_number>": [eur, eur_foil, eur_etched, usd, usd_foil, usd_etched]}}}
Prices are integer cents; null when Scryfall has none; trailing nulls are dropped. Keyed by
set + collector number (not Scryfall id) so localized printings, which rarely carry their own
prices, take the price of their printing.

Usage: python build_prices.py --out dist/prices.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import requests

USER_AGENT = "MTGScannerIndexBuilder/1.0 (github.com/VictorSabido/mtg-scanner-index)"
FIELDS = ["eur", "eur_foil", "eur_etched", "usd", "usd_foil", "usd_etched"]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json;q=0.9,*/*;q=0.8"})


def cents(v: str | None) -> int | None:
    return None if v in (None, "") else round(float(v) * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="bulk-cache")
    args = ap.parse_args()

    meta = next(
        d for d in session.get("https://api.scryfall.com/bulk-data", timeout=60).json()["data"]
        if d["type"] == "default_cards"
    )
    uri = meta.get("jsonl_download_uri") or meta["download_uri"]
    os.makedirs(args.cache, exist_ok=True)
    path = os.path.join(args.cache, os.path.basename(uri))
    if not os.path.exists(path):
        with session.get(uri, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(path + ".part", path)

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        cards = (json.loads(l) for l in f if l.strip()) if ".jsonl" in path else iter(json.load(f))
        sets: dict[str, dict[str, list]] = {}
        priced = 0
        for c in cards:
            p = c.get("prices") or {}
            row = [cents(p.get(k)) for k in FIELDS]
            while row and row[-1] is None:
                row.pop()
            if not row:
                continue
            bucket = sets.setdefault(c["set"], {})
            # English first in default_cards; keep the first priced printing per number.
            bucket.setdefault(c["collector_number"], row)
            priced += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    doc = {"version": 1, "updated": meta.get("updated_at"), "sets": sets}
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"{priced} priced printings in {len(sets)} sets → {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
