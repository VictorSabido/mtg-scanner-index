"""
Accuracy benchmark of the published index, simulating the phone pipeline:
full-card image -> 744x1039 "warp" -> perspective jitter, blur, exposure, noise, white balance
-> 3 art-box crops -> embed -> PCA/int8 search (same as IndexStore.kt) -> best over crops.

Usage: python eval_index.py --model art_embed_v1.onnx --index dist --n 400
"""
import argparse, gzip, random, struct, time

import cv2
import numpy as np
import requests

UA = "MTGScannerIndexEval/1.0 (github.com/VictorSabido/mtg-scanner-index)"
BOXES = [(0.08, 0.11, 0.84, 0.44), (0.12, 0.10, 0.76, 0.42), (0.06, 0.09, 0.88, 0.48)]


def load(index_dir):
    raw = gzip.open(f"{index_dir}/vectors.bin.gz").read()
    assert raw[:4] == b"MTGI"
    _, count, ind, dim = struct.unpack("<4I", raw[4:20]); o = 20
    mean = np.frombuffer(raw, "<f4", ind, o); o += ind * 4
    comps = np.frombuffer(raw, "<f4", dim * ind, o).reshape(dim, ind); o += dim * ind * 4
    scale = np.frombuffer(raw, "<f4", count, o); o += count * 4
    rows = np.frombuffer(raw, np.int8, count * dim, o).reshape(count, dim).astype(np.float32)
    lines = gzip.open(f"{index_dir}/cards.tsv.gz", "rt", encoding="utf-8").read().splitlines()
    return mean, comps, scale, rows, lines


def degrade(img, rng):
    h, w = img.shape[:2]
    j = lambda: rng.uniform(-0.02, 0.02)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * j(), h * j()], [w * (1 + j()), h * j()], [w * (1 + j()), h * (1 + j())], [w * j(), h * (1 + j())]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (5, 5), rng.uniform(0.6, 1.8))
    img = img.astype(np.float32) * rng.uniform(0.6, 1.2) + rng.uniform(-25, 25)
    img += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 7, img.shape)
    img *= np.array([rng.uniform(0.88, 1.12), 1, rng.uniform(0.88, 1.12)])
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    mean, comps, scale, rows, lines = load(a.index)
    net = cv2.dnn.readNetFromONNX(a.model)
    s = requests.Session(); s.headers["User-Agent"] = UA
    rng = random.Random(a.seed)
    sample = rng.sample(range(len(lines)), a.n)
    top1 = top5 = 0; tops = []; seconds = []; fails = []; ms = []
    for k, r in enumerate(sample):
        sid = lines[r].split("\t")[2]
        time.sleep(0.1)  # Scryfall API: ~10 req/s
        c = s.get(f"https://api.scryfall.com/cards/{sid}", timeout=30).json()
        uris = c.get("image_uris") or (c.get("card_faces") or [{}])[0].get("image_uris")
        if not uris:
            continue
        img = cv2.imdecode(np.frombuffer(s.get(uris["large"], timeout=60).content, np.uint8), cv2.IMREAD_COLOR)
        img = degrade(cv2.resize(img, (744, 1039), interpolation=cv2.INTER_AREA), rng)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        best = np.full(len(lines), -9.0, np.float32)
        for x, y, w, h in BOXES:
            crop = rgb[int(1039 * y):int(1039 * (y + h)), int(744 * x):int(744 * (x + w))]
            net.setInput(cv2.dnn.blobFromImage(crop, 1 / 255, (256, 256), swapRB=False, crop=False))
            v = net.forward().ravel(); v /= np.linalg.norm(v)
            p = comps @ (v - mean); p /= np.linalg.norm(p)
            best = np.maximum(best, (rows @ p) * scale)
        ms.append((time.time() - t) * 1000)
        o = np.argsort(-best)[:5]
        top1 += o[0] == r; top5 += r in o
        tops.append(best[o[0]] if o[0] == r else np.nan); seconds.append(best[o[1]])
        if o[0] != r:
            fails.append(f"{lines[r].split(chr(9))[1]} ({lines[r].split(chr(9))[3]}) -> {lines[o[0]].split(chr(9))[1]} ({lines[o[0]].split(chr(9))[3]}) {best[o[0]]:.3f} vs true {best[r]:.3f}")
        if (k + 1) % 50 == 0:
            print(f"{k + 1}: top1 {top1} top5 {top5}", flush=True)
    n = len(tops)
    print(f"\nRESULT n={n} top1={top1} ({100 * top1 / n:.1f}%) top5={top5} ({100 * top5 / n:.1f}%)")
    t = np.array(tops); t = t[~np.isnan(t)]
    print("correct top1 score p5/p50/p95:", np.percentile(t, [5, 50, 95]).round(3))
    print("runner-up score p50/p95/p99:", np.percentile(seconds, [50, 95, 99]).round(3))
    print(f"search+embed ms (runner CPU) p50={np.median(ms):.0f}")
    print("\nFAILURES:"); print("\n".join(fails))


if __name__ == "__main__":
    main()
