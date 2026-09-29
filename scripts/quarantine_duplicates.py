"""
PS6 Day 2 - Quarantine duplicate images identified by the validator.

Reads:
  datasets/<name>/validation_report.json
  (uses the duplicate_pairs field)

Moves one image from each duplicate pair from
  datasets/<name>/images/{train,val,test}/<file>
  datasets/<name>/labels/{train,val,test}/<file>.txt
to
  datasets/invalid/<name>/duplicates/<file>

Then prints a summary. After running this, re-split the dataset to
rebalance the train/val/test counts:
  python scripts/split_dataset.py --name <name> --seed 42
  python scripts/validate_dataset.py --name <name>

Usage:
  python scripts/quarantine_duplicates.py --name ppe
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"
INVALID_ROOT = DATASETS / "invalid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS6 Day 2 - quarantine duplicates")
    p.add_argument("--name", required=True, choices=["ppe", "fire_smoke"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = DATASETS / args.name
    report_path = base / "validation_report.json"
    if not report_path.exists():
        print(f"ERROR: no validation report at {report_path}")
        print("Run scripts/validate_dataset.py --name", args.name, "first.")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pairs = report.get("duplicate_pairs", [])
    if not pairs:
        print(f"[{args.name}] No duplicates to quarantine.")
        return 0

    # Move the SECOND element of each pair (keep the first occurrence)
    # Avoid double-move if the same file appears in multiple pairs.
    to_quarantine: set[str] = set()
    for a, b in pairs:
        # Strip the dirname if any (validator stores as (name, name))
        a_name = a.split("/")[-1]
        b_name = b.split("/")[-1]
        if a_name in to_quarantine:
            # 'a' is already being removed, move 'b' instead
            if b_name not in to_quarantine:
                to_quarantine.add(b_name)
        else:
            # Prefer to keep 'a' and remove 'b'
            to_quarantine.add(b_name)
    print(f"[{args.name}] Quarantining {len(to_quarantine)} duplicate images:")
    out_dir = INVALID_ROOT / args.name / "duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find them in whichever split they live in
    manifest = []
    for fname in sorted(to_quarantine):
        for split in ("train", "val", "test"):
            img_src = base / "images" / split / fname
            lbl_src = base / "labels" / split / (Path(fname).stem + ".txt")
            if img_src.exists():
                img_dst = out_dir / f"{split}_{fname}"
                lbl_dst = out_dir / f"{split}_{Path(fname).stem}.txt"
                shutil.copy2(img_src, img_dst)
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dst)
                img_src.unlink()
                if lbl_src.exists():
                    lbl_src.unlink()
                manifest.append({
                    "file": fname,
                    "from_split": split,
                    "reason": "near_duplicate_perceptual_hash",
                    "moved_to": str(img_dst),
                })
                print(f"  - {fname} (from {split})")
                break
    # Write manifest
    mf_path = out_dir / "quarantine_manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[OK] Manifest written to {mf_path}")
    print("Re-run split + validator to rebalance and re-validate:")
    print(f"  python scripts/split_dataset.py --name {args.name}")
    print(f"  python scripts/validate_dataset.py --name {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
