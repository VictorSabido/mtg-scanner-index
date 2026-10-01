"""
Build the offline artwork index consumed by modules/card-scanner (IndexStore.kt).

1. Scryfall bulk data: `unique_artwork` (one row per illustration) + `default_cards`
   (to list every printing that shares each illustration).
2. Download each art_crop from cards.scryfall.io (no rate limit on *.scryfall.io).
3. Embed with the same ONNX model + preprocessing as the app (OpenCV DNN, RGB, 256×256 squash).
4. PCA 5120 → DIM, L2-normalize, int8 per-row quantization.

Outputs (in --out):
  manifest.json   version, counts, model id, sha256 + size of each file
  hub.bin.gz      f32[count] "hubness" of each row: mean of its top-K cosine similarities to
                  background queries (art-box crops of real cards, upright AND upside down,
                  excluding the row's own artwork). The app scores rows as cos − λ·hub (CSLS-like)
                  so rows that attract everything (text-box-like artworks) stop winning.
  vectors.bin.gz  little-endian:
                    magic "MTGI" | u32 format(1) | u32 count | u32 inDim | u32 dim
                    f32[inDim] pca mean | f32[dim*inDim] pca components (row-major)
                    f32[count] row scale | i8[count*dim] rows
  cards.tsv.gz    one line per row (same order):
                    illustration_id \t name \t scryfall_id \t set \t collector_number \t lang \t printings
                  printings = comma list of "set:collector_number:scryfall_id" sharing the illustration
                  kind = "art" (art_crop) or "b0".."b2" (app art box N on the full card, special layouts)

Incremental mode (--reuse DIR with a previous release): rows of illustrations already in the
previous index are copied as-is and its PCA basis is kept, so only new artworks are downloaded
and embedded. Use a full build (no --reuse) occasionally to refit PCA / refresh changed images.

Usage:
  python scripts/index/build_index.py --model path/to/art_embed_v1.onnx --out dist [--limit 500] [--reuse prev]
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
# Odd artworks that showed up as false top-1 "hubs" in the eval and that nobody scans.
HUB_SETS = {"cmb1", "cmb2"}
HUB_NAMES = {"Double-Faced Substitute Card"}
# Layouts / frames whose art isn't where the standard art boxes look. For these we also index
# what each of the app's art boxes sees on the full card image (kinds b0/b1/b2), so a phone
# crop is compared with exactly the same region. (A whole-card crop was tried and rejected:
# full cards all look alike — frame, text box — and became false-match hubs: 46% top-1.)
FULL_CARD_LAYOUTS = {"split", "battle", "flip", "saga", "class", "case", "aftermath"}
# Must match ArtEmbedder.ART_BOXES (x, y, w, h fractions of the card).
ART_BOXES = [(0.08, 0.11, 0.84, 0.44), (0.12, 0.10, 0.76, 0.42), (0.06, 0.09, 0.88, 0.48)]

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
    if card.get("set") in HUB_SETS or card.get("name") in HUB_NAMES:
        return False
    if "playtest" in (card.get("promo_types") or []):
        return False
    face = front(card)
    return bool(face.get("illustration_id") or card.get("illustration_id")) and bool(
        (face.get("image_uris") or {}).get("art_crop")
    )


def illustration_id(card: dict) -> str:
    return front(card).get("illustration_id") or card.get("illustration_id") or ""


def needs_full_card(card: dict) -> bool:
    # Only layouts whose art really sits elsewhere. Full-art / borderless cards were tried too:
    # their box rows matched the text box of upside-down cards (hubs), and their art_crop
    # already covers the art.
    return bool(
        card.get("layout") in FULL_CARD_LAYOUTS
        or "Battle" in (front(card).get("type_line") or card.get("type_line") or "")
    ) and bool((front(card).get("image_uris") or {}).get("normal"))


def image_url(card: dict, kind: str) -> str:
    uris = front(card)["image_uris"]
    return uris["art_crop"] if kind == "art" else uris["normal"]


def region(img: np.ndarray, kind: str) -> np.ndarray:
    """The pixels an index row embeds: the whole art_crop, or art box N of the full card."""
    if kind == "art":
        return img
    x, y, w, h = ART_BOXES[int(kind[1:])]
    H, W = img.shape[:2]
    return img[int(H * y) : int(H * (y + h)), int(W * x) : int(W * (x + w))]


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


def load_previous(path: str) -> dict | None:
    """Rows (scale, int8 vector) keyed by illustration id + PCA basis of a previous release."""
    vec = os.path.join(path, "vectors.bin.gz")
    tsv = os.path.join(path, "cards.tsv.gz")
    man = os.path.join(path, "manifest.json")
    if not (os.path.exists(vec) and os.path.exists(tsv) and os.path.exists(man)):
        print("no previous index to reuse: full build", flush=True)
        return None
    manifest = json.load(open(man, encoding="utf-8"))
    if manifest.get("model") != MODEL_ID or manifest.get("format") != 1:
        print("previous index built with another model/format: full build", flush=True)
        return None
    raw = gzip.open(vec).read()
    _, count, in_dim, dim = struct.unpack("<4I", raw[4:20])
    o = 20
    mean = np.frombuffer(raw, "<f4", in_dim, o).copy(); o += in_dim * 4
    comps = np.frombuffer(raw, "<f4", dim * in_dim, o).reshape(dim, in_dim).copy(); o += dim * in_dim * 4
    scale = np.frombuffer(raw, "<f4", count, o); o += count * 4
    rows = np.frombuffer(raw, np.int8, count * dim, o).reshape(count, dim)
    # Row key = illustration id + kind (column 8; absent in older releases = 'art').
    ids = []
    for line in gzip.open(tsv, "rt", encoding="utf-8").read().splitlines():
        f = line.split("\t")
        ids.append(f"{f[0]}|{f[7] if len(f) > 7 else 'art'}")
    return {
        "version": manifest.get("version"),
        "mean": mean,
        "comps": comps,
        "in_dim": in_dim,
        "dim": dim,
        "rows": {ill: (float(scale[k]), rows[k].copy()) for k, ill in enumerate(ids)},
    }


HUB_BG_CARDS = int(os.environ.get("HUB_BG_CARDS", "3000"))
HUB_TOP_K = 10


def card_crops(bgr_card: np.ndarray) -> list[np.ndarray]:
    """The app's art boxes on a 744×1039 card, upright and rotated 180°."""
    card = cv2.resize(bgr_card, (744, 1039), interpolation=cv2.INTER_AREA)
    out = []
    for img in (card, cv2.rotate(card, cv2.ROTATE_180)):
        for b in range(len(ART_BOXES)):
            out.append(region(img, f"b{b}"))
    return out


def compute_hub(y: np.ndarray, row_ill: list[str], arts: list[dict], net, mean, comps, workers: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    pick = rng.choice(len(arts), size=min(HUB_BG_CARDS, len(arts)), replace=False)
    ill_index: dict[str, list[int]] = defaultdict(list)
    for k, ill in enumerate(row_ill):
        ill_index[ill].append(k)

    def work(i: int):
        return i, fetch(front(arts[i])["image_uris"]["normal"])

    qs, owners = [], []
    with cf.ThreadPoolExecutor(workers) as pool:
        for i, data in pool.map(work, pick):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            if img is None:
                continue
            for crop in card_crops(img):
                v = embed(net, crop)
                v /= np.linalg.norm(v) + 1e-12
                p = (v - mean) @ comps.T
                qs.append(p / (np.linalg.norm(p) + 1e-12))
                owners.append(illustration_id(arts[i]))
    q = np.stack(qs).astype(np.float32)
    print(f"hubness: {len(q)} background crops from {len(pick)} cards", flush=True)
    top = np.full((y.shape[0], HUB_TOP_K), -1.0, np.float32)
    for start in range(0, len(q), 512):
        sims = y @ q[start : start + 512].T  # (rows, chunk)
        for j, owner in enumerate(owners[start : start + 512]):
            sims[ill_index.get(owner, []), j] = -1.0  # never count a row's own artwork
        both = np.concatenate([top, sims], axis=1)
        top = -np.partition(-both, HUB_TOP_K - 1, axis=1)[:, :HUB_TOP_K]
    return top.mean(axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--limit", type=int, default=0, help="only the first N illustrations (testing)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--reuse", default="", help="dir with a previous release to build incrementally")
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
    items = [(c, "art") for c in arts] + [
        (c, f"b{b}") for c in arts if needs_full_card(c) for b in range(len(ART_BOXES))
    ]
    print(f"{len(arts)} illustrations, {len(items)} index rows", flush=True)

    net = cv2.dnn.readNetFromONNX(args.model)
    model_dim = embed(net, np.zeros((64, 64, 3), np.uint8)).shape[0]
    prev = load_previous(args.reuse) if args.reuse else None
    if prev and prev["in_dim"] != model_dim:
        print(f"previous index has {prev['in_dim']}-d features, model gives {model_dim}: full build", flush=True)
        prev = None
    reused: dict[int, tuple[float, np.ndarray]] = {}
    if prev:
        for i, (c, kind) in enumerate(items):
            hit = prev["rows"].get(f"{illustration_id(c)}|{kind}")
            if hit is not None:
                reused[i] = hit
        print(f"reusing {len(reused)} rows from previous index {prev['version']}", flush=True)

    vecs: list[np.ndarray | None] = [None] * len(items)
    todo = [i for i in range(len(items)) if i not in reused]
    print(f"embedding {len(todo)} rows", flush=True)

    def work(i: int) -> tuple[int, bytes | None]:
        return i, fetch(image_url(*items[i]))

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, data in pool.map(work, todo):
            done += 1
            if data:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    vecs[i] = embed(net, region(img, items[i][1]))
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(todo)}  {rate:.1f}/s", flush=True)

    keep = [i for i in range(len(items)) if i in reused or vecs[i] is not None]
    fresh = [i for i in keep if i not in reused]
    print(f"index rows {len(keep)}/{len(items)} ({len(fresh)} newly embedded)", flush=True)

    if prev:
        mean, comps = prev["mean"], prev["comps"]
        in_dim, dim = prev["in_dim"], prev["dim"]
    else:
        x = np.stack([vecs[i] for i in fresh]).astype(np.float32)
        # L2-normalize raw features before PCA so scale differences don't dominate.
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        mean = x.mean(axis=0)
        in_dim = x.shape[1]
        dim = min(DIM, x.shape[0], in_dim)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        comps = vt[:dim].astype(np.float32)  # (dim, inDim)

    new_scale: dict[int, float] = {}
    new_q: dict[int, np.ndarray] = {}
    if fresh:
        x = np.stack([vecs[i] for i in fresh]).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        y = (x - mean) @ comps.T
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
        sc = np.abs(y).max(axis=1) / 127.0
        qq = np.clip(np.round(y / sc[:, None]), -127, 127).astype(np.int8)
        for j, i in enumerate(fresh):
            new_scale[i], new_q[i] = float(sc[j]), qq[j]
    scale = np.array([reused[i][0] if i in reused else new_scale[i] for i in keep], dtype=np.float32)
    q = np.stack([reused[i][1] if i in reused else new_q[i] for i in keep]).astype(np.int8)

    vec_path = os.path.join(args.out, "vectors.bin.gz")
    with gzip.open(vec_path, "wb", compresslevel=6) as f:
        f.write(b"MTGI")
        f.write(struct.pack("<4I", 1, len(keep), in_dim, dim))
        f.write(mean.astype("<f4").tobytes())
        f.write(comps.astype("<f4").tobytes())
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())

    y_rows = q.astype(np.float32) * scale[:, None]
    row_ill = [illustration_id(items[i][0]) for i in keep]
    hub = compute_hub(y_rows, row_ill, arts, net, mean, comps, args.workers)
    print(f"hubness p50={np.median(hub):.3f} p99={np.percentile(hub, 99):.3f} max={hub.max():.3f}", flush=True)
    with gzip.open(os.path.join(args.out, "hub.bin.gz"), "wb", compresslevel=6) as f:
        f.write(hub.astype("<f4").tobytes())

    tsv_path = os.path.join(args.out, "cards.tsv.gz")
    with gzip.open(tsv_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for i in keep:
            c, kind = items[i]
            ill = illustration_id(c)
            name = c["name"].replace("\t", " ")
            prints = ",".join(printings.get(ill) or [f"{c['set']}:{c['collector_number']}:{c['id']}"])
            f.write(
                "\t".join([ill, name, c["id"], c["set"], c["collector_number"], c.get("lang", "en"), prints, kind])
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
            for name in ("vectors.bin.gz", "cards.tsv.gz", "hub.bin.gz")
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
