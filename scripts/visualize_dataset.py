"""
PS6 Day 2 - Step 15: Visual annotation check.

Randomly selects sample images from a YOLO dataset, draws their
bounding boxes on top, and saves the rendered PNGs under
outputs/dataset_preview/<dataset>/<split>/<filename>.png.

Usage:
  python scripts/visualize_dataset.py --name ppe --per-split 5
  python scripts/visualize_dataset.py --name fire_smoke --per-split 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"
OUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_preview"

CLASS_COLORS = [
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
]

EXPECTED_CLASSES = {
    "ppe": {0: "person"},
    "fire_smoke": {0: "fire", 1: "smoke"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS6 Day 2 - dataset visualizer")
    p.add_argument("--name", required=True, choices=["ppe", "fire_smoke"])
    p.add_argument("--per-split", type=int, default=5,
                   help="Number of images to render per split (default 5).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def draw_boxes(img_path: Path, label_path: Path, expected: dict) -> "cv2.Mat":
    img = cv2.imread(str(img_path))
    if img is None:
        # Return a placeholder black image
        return cv2.rectangle(
            cv2.putText(
                cv2.rectangle(
                    np_zeros := __import__('numpy').zeros((100, 200, 3), dtype='uint8'),
                    (0, 0), (200, 100), (0, 0, 255), -1
                ),
                "READ FAIL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            ),
            (0, 0), (200, 100), (0, 0, 255), 1
        )
    h, w = img.shape[:2]
    if label_path.exists():
        text = label_path.read_text(encoding="utf-8").strip()
        for ln in text.splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            cls_name = expected.get(cls, f"cls{cls}")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls_name, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Filename + split label in top-left corner
    label = f"{img_path.parent.name}/{img_path.name}"
    cv2.rectangle(img, (0, 0), (len(label) * 9 + 10, 22), (0, 0, 0), -1)
    cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return img


def main() -> int:
    args = parse_args()
    base = DATASETS / args.name
    expected = EXPECTED_CLASSES[args.name]
    out_root = OUT_ROOT / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rendered = 0
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            continue
        images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if not images:
            continue
        n_pick = min(args.per_split, len(images))
        picks = rng.sample(images, n_pick)
        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        for img_path in picks:
            label_path = lbl_dir / (img_path.stem + ".txt")
            out_img = draw_boxes(img_path, label_path, expected)
            out_path = out_split / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), out_img)
            print(f"  -> {out_path}")
            rendered += 1
    print(f"\n[OK] Rendered {rendered} sample images to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
