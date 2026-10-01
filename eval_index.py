"""
Accuracy benchmark of the published index, simulating the phone pipeline (IndexStore.kt +
CardScannerView.identifyBest + recognize.ts):

full-card image -> 744x1039 "warp" -> perspective jitter, blur, exposure, noise, white balance
(+ a share of cards upside down) -> 4 crops (3 art boxes + whole card) -> embed -> PCA/int8
search -> best per row; retry rotated 180° when the best score < RETRY_BELOW -> collapse rows
of the same illustration (art + whole-card rows) -> top-1 / margin vs runner-up.

Usage: python eval_index.py --model art_embed_v1.onnx --index dist --n 400
"""
import argparse, gzip, random, struct, time

import cv2
import numpy as np
import requests

UA = "MTGScannerIndexEval/1.0 (github.com/VictorSabido/mtg-scanner-index)"
BOXES = [(0.08, 0.11, 0.84, 0.44), (0.12, 0.10, 0.76, 0.42), (0.06, 0.09, 0.88, 0.48), (0.02, 0.02, 0.96, 0.96)]
RETRY_BELOW = 0.5


def load(index_dir):
    raw = gzip.open(f"{index_dir}/vectors.bin.gz").read()
    assert raw[:4] == b"MTGI"
    _, count, ind, dim = struct.unpack("<4I", raw[4:20]); o = 20
    mean = np.frombuffer(raw, "<f4", ind, o); o += ind * 4
    comps = np.frombuffer(raw, "<f4", dim * ind, o).reshape(dim, ind); o += dim * ind * 4
    scale = np.frombuffer(raw, "<f4", count, o); o += count * 4
    rows = np.frombuffer(raw, np.int8, count * dim, o).reshape(count, dim).astype(np.float32)
    lines = [l.split("\t") for l in gzip.open(f"{index_dir}/cards.tsv.gz", "rt", encoding="utf-8").read().splitlines()]
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
    ap.add_argument("--upside-down", type=float, default=0.25, help="share of queries rotated 180°")
    a = ap.parse_args()
    mean, comps, scale, rows, lines = load(a.index)
    ill = np.array([l[0] for l in lines])
    uniq, row_to_ill = np.unique(ill, return_inverse=True)
    art_rows = [k for k, l in enumerate(lines) if (l[7] if len(l) > 7 else "art") == "art"]
    print(f"index rows {len(lines)}, illustrations {len(uniq)}")
    net = cv2.dnn.readNetFromONNX(a.model)
    s = requests.Session(); s.headers["User-Agent"] = UA
    rng = random.Random(a.seed)
    sample = rng.sample(art_rows, a.n)

    def scores_for(rgb):
        best = np.full(len(lines), -9.0, np.float32)
        for x, y, w, h in BOXES:
            crop = rgb[int(1039 * y):int(1039 * (y + h)), int(744 * x):int(744 * (x + w))]
            net.setInput(cv2.dnn.blobFromImage(crop, 1 / 255, (256, 256), swapRB=False, crop=False))
            v = net.forward().ravel(); v /= np.linalg.norm(v)
            p = comps @ (v - mean); p /= np.linalg.norm(p)
            best = np.maximum(best, (rows @ p) * scale)
        return best

    top1 = top5 = 0; decisions = []; fails = []; ms = []; retried = 0; flips = 0; flip_ok = 0
    for k, r in enumerate(sample):
        sid = lines[r][2]
        time.sleep(0.1)  # Scryfall API: ~10 req/s
        c = s.get(f"https://api.scryfall.com/cards/{sid}", timeout=30).json()
        uris = c.get("image_uris") or (c.get("card_faces") or [{}])[0].get("image_uris")
        if not uris:
            continue
        img = cv2.imdecode(np.frombuffer(s.get(uris["large"], timeout=60).content, np.uint8), cv2.IMREAD_COLOR)
        img = degrade(cv2.resize(img, (744, 1039), interpolation=cv2.INTER_AREA), rng)
        upside = rng.random() < a.upside_down
        if upside:
            img = cv2.rotate(img, cv2.ROTATE_180); flips += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        best = scores_for(rgb)
        if best.max() < RETRY_BELOW:
            retried += 1
            best = np.maximum(best, scores_for(cv2.rotate(rgb, cv2.ROTATE_180)))
        ms.append((time.time() - t) * 1000)
        # collapse rows -> illustrations
        per_ill = np.full(len(uniq), -9.0, np.float32)
        np.maximum.at(per_ill, row_to_ill, best)
        o = np.argsort(-per_ill)[:5]
        truth = row_to_ill[r]
        ok = o[0] == truth
        top1 += ok; top5 += truth in o
        if upside: flip_ok += ok
        decisions.append((float(per_ill[o[0]]), float(per_ill[o[0]] - per_ill[o[1]]), bool(ok)))
        if not ok:
            name = lambda i: next(l for l in lines if l[0] == uniq[i])
            got, want = name(o[0]), name(truth)
            fails.append(f"{want[1]} ({want[3]}){' [180]' if upside else ''} -> {got[1]} ({got[3]}) {per_ill[o[0]]:.3f} vs true {per_ill[truth]:.3f}")
        if (k + 1) % 50 == 0:
            print(f"{k + 1}: top1 {top1} top5 {top5}", flush=True)
    n = len(decisions)
    print(f"\nRESULT n={n} top1={top1} ({100 * top1 / n:.1f}%) top5={top5} ({100 * top5 / n:.1f}%)")
    print(f"upside-down queries {flips}: top1 {flip_ok}; 180° retries triggered {retried}")
    print(f"embed+search ms (runner CPU) p50={np.median(ms):.0f} p95={np.percentile(ms, 95):.0f}")
    print("\nFAST PATH (accept without OCR when score>=S and margin>=M):")
    print("   S     M   coverage  wrong-adds")
    for S in (0.5, 0.55, 0.6, 0.65):
        for M in (0.06, 0.08, 0.1, 0.12, 0.15):
            acc = [ok for sc, mg, ok in decisions if sc >= S and mg >= M]
            if acc:
                bad = len(acc) - sum(acc)
                print(f"  {S:.2f}  {M:.2f}  {100 * len(acc) / n:5.1f}%   {bad:3d} ({100 * bad / len(acc):.2f}%)")
    print("\nFAILURES:"); print("\n".join(fails))


if __name__ == "__main__":
    main()
