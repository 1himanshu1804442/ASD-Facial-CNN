#!/usr/bin/env python3
"""
Near-duplicate audit across train / val / test splits (CPU only, needs Pillow).

Usage:
    python audit_duplicates.py --data-root /path/to/dataset
    python audit_duplicates.py --data-root /path/to/dataset --splits train validation test

Layout expected: <data-root>/<split>/[<class>/]*.jpg   (class sub-folders optional;
the first sub-folder under the split is used as the label).

What it catches: exact copies, resized / re-compressed copies, horizontal flips and
centred crops (90% / 80%) of an image, via 8x8 difference hashes (dHash).
What it does NOT catch: off-centre crops, heavy edits, and *different photos of the
same child* (identity leakage) -- that needs face / CNN embeddings. Always look at the
saved side-by-side pair images before quoting a number: dHash on plain-background
portraits can produce false matches.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def popcount(x: int) -> int:
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


def dhash(img: Image.Image, size: int) -> int:
    small = img.resize((size + 1, size), BILINEAR)
    px = list(small.get_flattened_data()) if hasattr(small, "get_flattened_data") else list(small.getdata())
    bits = 0
    for r in range(size):
        base = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | (1 if px[base + c] > px[base + c + 1] else 0)
    return bits


def hashes_for(path: Path, size: int):
    """Return (full-image hash, set of hashes of flipped / centre-cropped variants)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im.load()
    w, h = im.size
    full = dhash(im, size)
    variants = {full}
    for frac in (1.0, 0.9, 0.8):
        dx, dy = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
        crop = im if frac == 1.0 else im.crop((dx, dy, w - dx, h - dy))
        variants.add(dhash(crop, size))
        variants.add(dhash(ImageOps.mirror(crop), size))
    return full, tuple(variants)


def distance(a: dict, b: dict) -> int:
    d1 = min(popcount(a["full"] ^ v) for v in b["variants"])
    d2 = min(popcount(b["full"] ^ v) for v in a["variants"])
    return min(d1, d2)


def label_of(path: Path, split_dir: Path):
    rel = path.relative_to(split_dir)
    return rel.parts[0] if len(rel.parts) > 1 else None


def side_by_side(p1: Path, p2: Path, out: Path, height: int = 256) -> None:
    ims = []
    for p in (p1, p2):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            ims.append(im.resize((max(1, int(im.width * height / im.height)), height)))
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + 8, height), "white")
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + 8, 0))
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit train/val/test splits for near-duplicate images.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Split folder names, EARLIEST FIRST (default: train val test)")
    ap.add_argument("--max-dist", type=int, default=6, help="Max Hamming distance (default 6 of 64 bits)")
    ap.add_argument("--hash-size", type=int, default=8, help="dHash size (8 -> 64 bits). Scale --max-dist if changed.")
    ap.add_argument("--out-dir", type=Path, default=Path("dup_audit"))
    ap.add_argument("--save-pairs", type=int, default=30, help="Save side-by-side images for the first N cross-split pairs")
    args = ap.parse_args()

    recs = []
    for split in args.splits:
        sdir = args.data_root / split
        if not sdir.is_dir():
            found = sorted(p.name for p in args.data_root.iterdir() if p.is_dir())
            raise SystemExit(f"Split folder not found: {sdir}\nFolders under data-root: {found}")
        files = [p for p in sorted(sdir.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS]
        ok = 0
        for p in files:
            try:
                full, variants = hashes_for(p, args.hash_size)
            except Exception as exc:  # unreadable / corrupt image
                print(f"[WARN] skipped {p}: {exc}")
                continue
            recs.append({"split": split, "path": p, "label": label_of(p, sdir), "full": full, "variants": variants})
            ok += 1
        print(f"[INFO] {split}: {ok} images hashed")

    order = {s: i for i, s in enumerate(args.splits)}
    pairs = []
    n = len(recs)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(recs[i], recs[j])
            if d <= args.max_dist:
                a, b = recs[i], recs[j]
                if order[a["split"]] > order[b["split"]]:
                    a, b = b, a  # a = earlier split
                pairs.append((d, a, b))
    pairs.sort(key=lambda t: t[0])

    cross = [p for p in pairs if p[1]["split"] != p[2]["split"]]
    within = [p for p in pairs if p[1]["split"] == p[2]["split"]]
    conflicts = [p for p in pairs if p[1]["label"] and p[2]["label"] and p[1]["label"] != p[2]["label"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance", "split_a", "label_a", "path_a", "split_b", "label_b", "path_b", "cross_split", "label_conflict"])
        for d, a, b in pairs:
            w.writerow([d, a["split"], a["label"], a["path"], b["split"], b["label"], b["path"],
                        a["split"] != b["split"], bool(a["label"] and b["label"] and a["label"] != b["label"])])
    if args.save_pairs:
        pdir = args.out_dir / "pairs"
        pdir.mkdir(exist_ok=True)
        for k, (d, a, b) in enumerate(cross[: args.save_pairs], 1):
            try:
                side_by_side(a["path"], b["path"], pdir / f"{k:02d}_dist{d}_{a['split']}-{b['split']}.png")
            except Exception as exc:
                print(f"[WARN] could not render pair {k}: {exc}")

    print("\n=== Duplicate audit summary ===")
    print(f"Hamming <= {args.max_dist}/{args.hash_size ** 2}; flip + centre-crop (90%, 80%) tolerant")
    sizes = {s: sum(1 for r in recs if r["split"] == s) for s in args.splits}
    for later in args.splits[1:]:
        for earlier in args.splits[: order[later]]:
            hit = {id(b) for _, a, b in cross if a["split"] == earlier and b["split"] == later}
            print(f"{later:>6} images with a near-duplicate in {earlier}: {len(hit)} / {sizes[later]}"
                  f" ({100 * len(hit) / max(1, sizes[later]):.1f}%)")
    print(f"Cross-split pairs: {len(cross)}   |   within-split pairs: {len(within)}")
    print(f"Pairs whose class labels DISAGREE: {len(conflicts)}")
    print(f"Wrote {args.out_dir / 'pairs.csv'}" + (f" and {min(len(cross), args.save_pairs)} pair images" if args.save_pairs else ""))
    for d, a, b in cross[:10]:
        print(f"  [dist={d}] {a['split']}/{a['path'].name} ({a['label']})  <->  {b['split']}/{b['path'].name} ({b['label']})")


if __name__ == "__main__":
    main()
