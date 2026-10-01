"""
Offline card metadata for the app, from Scryfall's `all_cards` bulk file (every language).

With it the app builds the card object for a recognized printing — and picks its printed
language from the OCR'd title — without calling the API. Gameplay data changes rarely, so a
weekly rebuild is enough (Scryfall's own advice).

Output (gzip, tab-separated, one record per line, sets first):
  S \t set \t set_name
  C \t set \t collector_number \t lang \t scryfall_id \t rarity \t finishes \t name
    rarity    c/u/r/m/s/b (common, uncommon, rare, mythic, special, bonus)
    finishes  letters n/f/e (nonfoil, foil, etched)
    name      English name for `en` rows; the printed (localized) name otherwise
Only paper cards in the layouts the artwork index covers.

Usage: python build_meta.py --out dist/meta.tsv.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import requests

USER_AGENT = "MTGScannerIndexBuilder/1.0 (github.com/VictorSabido/mtg-scanner-index)"
SKIP_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "vanguard", "scheme", "planar"}
RARITY = {"common": "c", "uncommon": "u", "rare": "r", "mythic": "m", "special": "s", "bonus": "b"}
FINISH = {"nonfoil": "n", "foil": "f", "etched": "e"}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json;q=0.9,*/*;q=0.8"})


def clean(s: str) -> str:
    return s.replace("\t", " ").replace("\n", " ").strip()


def printed(c: dict) -> str:
    if c.get("printed_name"):
        return c["printed_name"]
    faces = [f.get("printed_name") for f in c.get("card_faces") or []]
    if faces and all(faces):
        return " // ".join(faces)
    return c["name"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="bulk-cache")
    args = ap.parse_args()

    meta = next(
        d for d in session.get("https://api.scryfall.com/bulk-data", timeout=60).json()["data"]
        if d["type"] == "all_cards"
    )
    uri = meta.get("jsonl_download_uri") or meta["download_uri"]
    os.makedirs(args.cache, exist_ok=True)
    path = os.path.join(args.cache, os.path.basename(uri))
    if not os.path.exists(path):
        with session.get(uri, stream=True, timeout=1800) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(path + ".part", path)

    sets: dict[str, str] = {}
    rows: list[tuple] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        cards = (json.loads(l) for l in f if l.strip()) if ".jsonl" in path else iter(json.load(f))
        for c in cards:
            if c.get("digital") or c.get("layout") in SKIP_LAYOUTS:
                continue
            if "paper" not in (c.get("games") or []):
                continue
            lang = c.get("lang", "en")
            sets.setdefault(c["set"], c.get("set_name", c["set"].upper()))
            rows.append((
                c["set"], c["collector_number"], lang, c["id"],
                RARITY.get(c.get("rarity", ""), ""),
                "".join(FINISH[x] for x in c.get("finishes") or [] if x in FINISH),
                clean(c["name"] if lang == "en" else printed(c)),
            ))

    rows.sort(key=lambda r: (r[0], r[1], r[2] != "en", r[2]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        for code, name in sorted(sets.items()):
            f.write(f"S\t{code}\t{clean(name)}\n")
        for r in rows:
            f.write("C\t" + "\t".join(r) + "\n")
    langs: dict[str, int] = {}
    for r in rows:
        langs[r[2]] = langs.get(r[2], 0) + 1
    print(f"{len(rows)} printings in {len(sets)} sets → {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.2f} MB); by lang: {dict(sorted(langs.items()))}")


if __name__ == "__main__":
    main()
