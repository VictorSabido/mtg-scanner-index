"""
Build the offline artwork index consumed by modules/card-scanner (IndexStore.kt).

1. Scryfall bulk data: `unique_artwork` (one row per illustration) + `default_cards`
   (to list every printing that shares each illustration).
2. Download each art_crop from cards.scryfall.io (no rate limit on *.scryfall.io).
3. Embed with the same ONNX model + preprocessing as the app (OpenCV DNN, RGB, 256×256 squash).
4. PCA 5120 → DIM, L2-normalize, int8 per-row quantization.

Outputs (in --out):
  manifest.json   version, counts, model id, sha256 + size of each file
  vectors.bin.gz  little-endian:
                    magic "MTGI" | u32 format(1) | u32 count | u32 inDim | u32 dim
                    f32[inDim] pca mean | f32[dim*inDim] pca components (row-major)
                    f32[count] row scale | i8[count*dim] rows
  cards.tsv.gz    one line per row (same order):
                    illustration_id \t name \t scryfall_id \t set \t collector_number \t lang \t printings
                  printings = comma list of "set:collector_number:scryfall_id" sharing the illustration

Usage:
  python scripts/index/build_index.py --model path/to/art_embed_v1.onnx --out dist [--limit 500]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import json
import os
import struct
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import requests

MODEL_ID = "art_embed_v1"
INPUT = 256
DIM = 256
USER_AGENT = "MTGScannerIndexBuilder/1.0 (github.com/VictorSabido/mtg-scanner-index)"
SKIP_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "vanguard", "scheme", "planar"}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json;q=0.9,*/*;q=0.8"})


def bulk_uri(kind: str) -> str:
    data = session.get("https://api.scryfall.com/bulk-data", timeout=60).json()["data"]
    entry = next(d for d in data if d["type"] == kind)
    return entry.get("jsonl_download_uri") or entry["download_uri"]


def load_bulk(kind: str, cache_dir: str) -> list[dict]:
    """Bulk files are served as JSON Lines (gzip); older API versions served a JSON array."""
    uri = bulk_uri(kind)
    path = os.path.join(cache_dir, os.path.basename(uri))
    if not os.path.exists(path):
        print(f"downloading bulk {kind}…", flush=True)
        with session.get(uri, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(path + ".part", path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        if ".jsonl" in path:
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def front(card: dict) -> dict:
    if "image_uris" in card:
        return card
    faces = card.get("card_faces") or [{}]
    return faces[0]


def usable(card: dict) -> bool:
    if card.get("digital") or card.get("layout") in SKIP_LAYOUTS:
        return False
    if "paper" not in (card.get("games") or []):
        return False
    face = front(card)
    return bool(face.get("illustration_id") or card.get("illustration_id")) and bool(
        (face.get("image_uris") or {}).get("art_crop")
    )


def illustration_id(card: dict) -> str:
    return front(card).get("illustration_id") or card.get("illustration_id") or ""


def embed(net: cv2.dnn.Net, bgr: np.ndarray) -> np.ndarray:
    # Same as the app: RGB in [0,1], squash-resize to 256×256 (normalization is in the graph).
    blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (INPUT, INPUT), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward().ravel().astype(np.float32)


def fetch(url: str, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--limit", type=int, default=0, help="only the first N illustrations (testing)")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    arts = [c for c in load_bulk("unique_artwork", args.cache) if usable(c)]
    printings: dict[str, list[str]] = defaultdict(list)
    for c in load_bulk("default_cards", args.cache):
        if usable(c):
            printings[illustration_id(c)].append(f"{c['set']}:{c['collector_number']}:{c['id']}")
    if args.limit:
        arts = arts[: args.limit]
    print(f"{len(arts)} illustrations", flush=True)

    net = cv2.dnn.readNetFromONNX(args.model)
    vecs: list[np.ndarray | None] = [None] * len(arts)

    def work(i: int) -> tuple[int, bytes | None]:
        return i, fetch(front(arts[i])["image_uris"]["art_crop"])

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, data in pool.map(work, range(len(arts))):
            done += 1
            if data:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    vecs[i] = embed(net, img)
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(arts)}  {rate:.1f}/s", flush=True)

    keep = [i for i, v in enumerate(vecs) if v is not None]
    print(f"embedded {len(keep)}/{len(arts)}", flush=True)
    x = np.stack([vecs[i] for i in keep]).astype(np.float32)
    # L2-normalize raw features before PCA so scale differences don't dominate.
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

    mean = x.mean(axis=0)
    dim = min(DIM, x.shape[0], x.shape[1])
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    comps = vt[:dim].astype(np.float32)  # (dim, inDim)
    y = (x - mean) @ comps.T
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    scale = np.abs(y).max(axis=1) / 127.0
    q = np.clip(np.round(y / scale[:, None]), -127, 127).astype(np.int8)

    vec_path = os.path.join(args.out, "vectors.bin.gz")
    with gzip.open(vec_path, "wb", compresslevel=6) as f:
        f.write(b"MTGI")
        f.write(struct.pack("<4I", 1, len(keep), x.shape[1], dim))
        f.write(mean.astype("<f4").tobytes())
        f.write(comps.astype("<f4").tobytes())
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())

    tsv_path = os.path.join(args.out, "cards.tsv.gz")
    with gzip.open(tsv_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for i in keep:
            c = arts[i]
            ill = illustration_id(c)
            name = c["name"].replace("\t", " ")
            prints = ",".join(printings.get(ill) or [f"{c['set']}:{c['collector_number']}:{c['id']}"])
            f.write(
                "\t".join([ill, name, c["id"], c["set"], c["collector_number"], c.get("lang", "en"), prints])
                + "\n"
            )

    version = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
    manifest = {
        "format": 1,
        "version": version,
        "model": MODEL_ID,
        "count": len(keep),
        "dim": dim,
        "files": {
            name: {"size": os.path.getsize(os.path.join(args.out, name)), "sha256": sha256(os.path.join(args.out, name))}
            for name in ("vectors.bin.gz", "cards.tsv.gz")
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
