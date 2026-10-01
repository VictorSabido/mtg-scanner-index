# mtg-scanner-index

Offline artwork index for the MTG Scanner Android app.

Each [release](../../releases/latest) contains:

| File | Content |
|---|---|
| `manifest.json` | version, model id, row count, size + sha256 of each file |
| `vectors.bin.gz` | PCA-projected, int8-quantized artwork embeddings (format documented in `build_index.py`) |
| `cards.tsv.gz` | one row per illustration: illustration id, name, representative printing, and every printing sharing that art |

Built weekly by [`build-index.yml`](.github/workflows/build-index.yml) with `build_index.py` and the
`art_embed_v1.onnx` model (ImageNet MobileNetV2 trunk, 2×2 pooled features).

Card data from [Scryfall](https://scryfall.com) bulk data. No card images are redistributed here:
only numeric embeddings derived from them, plus Scryfall ids. Magic: The Gathering is © Wizards of the Coast.
