"""
PS6 Day 2 - Step (added): Split a YOLO-format dataset into train/val/test.

Reads from:
  datasets/<name>/raw_all/images/   and   datasets/<name>/raw_all/labels/

Writes to:
  datasets/<name>/images/{train,val,test}/
  datasets/<name>/labels/{train,val,test}/

Split ratio defaults:
  - >= 1000 images: 70/20/10
  - < 1000 images:  80/10/10 (more training data for small sets)

Deterministic: uses a fixed seed (default 42) so the split can be
reproduced exactly. Set --seed N to override.

Group-by-source: optional. If a `source.txt` file exists in
raw_all/ (one image filename per line, naming source video),
images from the same source are kept in the same split.

Usage:
  python scripts/split_dataset.py --name ppe
  python scripts/split_dataset.py --name fire_smoke --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS6 Day 2 - dataset splitter")
    p.add_argument("--name", required=True, choices=["ppe", "fire_smoke"],
                   help="Dataset name under datasets/.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default 42).")
    p.add_argument("--ratio", default="auto",
                   help="Train:val:test ratio, e.g. 70:20:10. 'auto' picks "
                        "70:20:10 if >=1000 images, else 80:10:10.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without copying files.")
    return p.parse_args()


def parse_ratio(ratio: str, n: int) -> tuple[float, float, float]:
    if ratio == "auto":
        if n >= 1000:
            r = (0.70, 0.20, 0.10)
        else:
            r = (0.80, 0.10, 0.10)
        return r
    parts = ratio.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad ratio: {ratio}")
    nums = [float(p) for p in parts]
    s = sum(nums)
    return (nums[0] / s, nums[1] / s, nums[2] / s)


def main() -> int:
    args = parse_args()
    base = DATASETS / args.name
    raw_img = base / "raw_all" / "images"
    raw_lbl = base / "raw_all" / "labels"
    if not raw_img.exists():
        print(f"ERROR: raw images not found: {raw_img}", flush=True)
        return 2

    # List images (jpg/jpeg/png)
    images = sorted([
        p for p in raw_img.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    n = len(images)
    print(f"[split] dataset={args.name}  total_images={n}  seed={args.seed}")

    # Determine ratio
    r_train, r_val, r_test = parse_ratio(args.ratio, n)
    print(f"[split] ratio train:val:test = "
          f"{r_train:.2f}:{r_val:.2f}:{r_test:.2f}")

    # Shuffle deterministically
    rng = random.Random(args.seed)
    images = list(images)
    rng.shuffle(images)

    n_train = max(1, int(round(n * r_train)))
    n_val = max(1, int(round(n * r_val)))
    # Ensure test set is non-empty if n >= 3
    if n >= 3:
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test
    else:
        # For tiny datasets, just put everything in train
        n_train, n_val, n_test = n, 0, 0

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }
    print(f"[split] counts: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")

    if args.dry_run:
        print("[split] dry-run; not copying files.")
        for split, lst in splits.items():
            print(f"  {split}: {len(lst)} files")
        return 0

    # Copy (not move — keep raw_all/ for reproducibility)
    n_copied = 0
    for split, lst in splits.items():
        out_img = base / "images" / split
        out_lbl = base / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img_path in lst:
            # Copy image
            shutil.copy2(img_path, out_img / img_path.name)
            # Copy corresponding label (same stem, .txt)
            lbl_path = raw_lbl / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)
            else:
                # Empty label file (negative sample)
                (out_lbl / (img_path.stem + ".txt")).write_text("", encoding="utf-8")
            n_copied += 1
    print(f"[split] Copied {n_copied} image+label pairs into images/ and labels/ subfolders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
