"""
PS6 Day 2 - Step 14: Dataset validator.

Verifies a YOLO-format dataset for:
  1. Image exists.
  2. Label exists where required.
  3. Label syntax is valid.
  4. Class ID is valid (against the expected class set).
  5. x_center is 0..1.
  6. y_center is 0..1.
  7. width  is >0 and <=1.
  8. height is >0 and <=1.
  9. Bounding box remains valid.
 10. Image can be opened.
 11. No obvious image/label mismatch (file-name based).
 12. Class counts are calculated.

Also runs perceptual-hash duplicate detection (imagehash) across all
images in the dataset, to flag near-duplicate images which can cause
train/test leakage.

Usage:
  python scripts/validate_dataset.py --name ppe
  python scripts/validate_dataset.py --name fire_smoke --no-duplicate-check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"

# Expected class indices for each dataset (from class_mapping.yaml)
EXPECTED_CLASSES = {
    "ppe": {0: "person"},                # MVP
    "fire_smoke": {0: "fire", 1: "smoke"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS6 Day 2 - dataset validator")
    p.add_argument("--name", required=True, choices=["ppe", "fire_smoke"])
    p.add_argument("--no-duplicate-check", action="store_true",
                   help="Skip the perceptual-hash duplicate check (faster).")
    return p.parse_args()


def validate_one_image(img_path: Path, label_path: Path,
                       expected_classes: dict) -> tuple[dict, list[str]]:
    info = {"image": str(img_path), "label": str(label_path), "errors": []}
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            info["errors"].append("cannot_open_image")
            info["shape"] = None
        else:
            info["shape"] = list(img.shape[:2])  # H, W
    except Exception as e:
        info["errors"].append(f"image_read_exception:{type(e).__name__}")
        info["shape"] = None

    if not label_path.exists():
        info["errors"].append("missing_label_file")
        info["boxes"] = []
        info["label_lines"] = 0
        return info, info["errors"]

    text = label_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    info["label_lines"] = len(lines)
    info["boxes"] = []
    if len(lines) == 0:
        # Empty label = valid negative sample
        return info, info["errors"]
    for ln_idx, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 5:
            info["errors"].append(f"line_{ln_idx}_bad_field_count:{len(parts)}")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:])
        except ValueError:
            info["errors"].append(f"line_{ln_idx}_non_numeric")
            continue
        if cls not in expected_classes:
            info["errors"].append(f"line_{ln_idx}_bad_class:{cls}")
        if not (0.0 <= xc <= 1.0):
            info["errors"].append(f"line_{ln_idx}_xc_out_of_range:{xc}")
        if not (0.0 <= yc <= 1.0):
            info["errors"].append(f"line_{ln_idx}_yc_out_of_range:{yc}")
        if not (0.0 < w <= 1.0):
            info["errors"].append(f"line_{ln_idx}_w_out_of_range:{w}")
        if not (0.0 < h <= 1.0):
            info["errors"].append(f"line_{ln_idx}_h_out_of_range:{h}")
        info["boxes"].append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h})
    return info, info["errors"]


def find_duplicates(images: list[Path]) -> list[tuple[str, str]]:
    """Perceptual hash to detect near-duplicate images."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("  [warn] imagehash not installed; skipping duplicate check")
        return []
    hashes: dict[str, str] = {}  # filename -> hash
    duplicates: list[tuple[str, str]] = []
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                h = str(imagehash.phash(im))
        except Exception:
            continue
        for other_name, other_h in hashes.items():
            if h == other_h:
                duplicates.append((img_path.name, other_name))
                break
        else:
            hashes[img_path.name] = h
    return duplicates


def main() -> int:
    args = parse_args()
    base = DATASETS / args.name
    expected = EXPECTED_CLASSES[args.name]
    print("=" * 60)
    print(f"DATASET VALIDATION: {args.name}")
    print(f"Expected classes: {expected}")
    print("=" * 60)

    all_class_counts: Counter = Counter()
    all_errors: list[str] = []
    all_records: list[dict] = []
    per_split_stats = {}
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            print(f"\n[{split}] SKIP (directory does not exist: {img_dir})")
            continue
        images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        n_imgs = len(images)
        n_lbls = sum(1 for p in lbl_dir.iterdir() if p.suffix == ".txt") if lbl_dir.exists() else 0
        print(f"\n[{split}] images={n_imgs}  labels={n_lbls}")
        split_class_counts: Counter = Counter()
        split_errors = 0
        split_invalid_imgs = 0
        for img_path in images:
            label_path = lbl_dir / (img_path.stem + ".txt")
            info, errs = validate_one_image(img_path, label_path, expected)
            all_records.append({"split": split, **info})
            if errs:
                split_errors += len(errs)
                split_invalid_imgs += 1
                for e in errs:
                    all_errors.append(f"{split}/{img_path.name}: {e}")
            for box in info["boxes"]:
                split_class_counts[box["cls"]] += 1
                all_class_counts[box["cls"]] += 1
        # Class distribution for this split
        for cls_id, cnt in sorted(split_class_counts.items()):
            cls_name = expected.get(cls_id, f"unknown_{cls_id}")
            print(f"  class {cls_id} ({cls_name}): {cnt}")
        print(f"  invalid images : {split_invalid_imgs}")
        print(f"  total errors   : {split_errors}")
        per_split_stats[split] = {
            "images": n_imgs,
            "labels": n_lbls,
            "class_counts": dict(split_class_counts),
            "invalid_images": split_invalid_imgs,
            "total_errors": split_errors,
        }

    print("\n--- DUPLICATE CHECK ---")
    if not args.no_duplicate_check:
        all_images = []
        for split in ("train", "val", "test"):
            d = base / "images" / split
            if d.exists():
                all_images += [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        dups = find_duplicates(all_images)
        if dups:
            print(f"  FOUND {len(dups)} near-duplicate pairs (LEAKAGE RISK):")
            for a, b in dups:
                print(f"    {a}  <->  {b}")
        else:
            print("  No near-duplicate images found.")
    else:
        dups = []
        print("  Skipped (--no-duplicate-check)")

    # Overall status
    status = "PASS" if (not all_errors and not dups) else "FAIL"
    print("\n" + "=" * 60)
    print(f"STATUS: {status}")
    print("=" * 60)
    print("\nClass counts (whole dataset):")
    for cls_id, cnt in sorted(all_class_counts.items()):
        cls_name = expected.get(cls_id, f"unknown_{cls_id}")
        print(f"  {cls_id} ({cls_name}): {cnt}")
    if all_errors:
        print(f"\nFirst 10 errors (out of {len(all_errors)}):")
        for e in all_errors[:10]:
            print(f"  - {e}")

    # Write JSON report
    report = {
        "dataset": args.name,
        "expected_classes": expected,
        "per_split": per_split_stats,
        "overall_class_counts": {int(k): int(v) for k, v in all_class_counts.items()},
        "duplicate_pairs": [(a, b) for a, b in dups],
        "total_errors": len(all_errors),
        "errors_sample": all_errors[:20],
        "status": status,
    }
    out = base / "validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[OK] Validation report written to {out}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
