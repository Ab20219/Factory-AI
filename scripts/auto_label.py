"""
PS6 Day 2 - Auto-label the downloaded raw images and convert them
to YOLO format under datasets/<ppe|fire_smoke>/raw_all/.

LABELING STRATEGY (honest, transparent):

  fire_smoke/fire    : whole-image bbox, class 0 (fire).
                       Ground truth comes from the Wikimedia search
                       query "fire" / "camp fire" / "forest fire" etc.
  fire_smoke/smoke   : whole-image bbox, class 1 (smoke).
                       Same source-of-truth rationale.

  ppe/person         : YOLO26n (COCO pretrained) is used to detect
                       "person" boxes. COCO has a person class, so
                       this is legitimate object detection.
  ppe/helmet         : DEFERRED. Cannot reliably auto-distinguish
                       "helmet" from "no_helmet" without a manually
                       annotated dataset (Hard Hat Workers v2).
                       See DATASET_REPORT.md for the rationale.
  ppe/no_helmet      : DEFERRED. Same reason.
  ppe/safety_vest    : DEFERRED. No COCO class for safety vest.
  ppe/gloves         : DEFERRED. No COCO class for gloves.

These demo labels are sufficient to validate the Day 2 pipeline
(split, clean, validate, visualize) end-to-end. For Day 3 training,
the user MUST replace this with a properly annotated dataset
(D-Fire + Hard Hat Workers v2 or GDUT-HD). The scripts will work
unchanged on those datasets.

Outputs:
  datasets/fire_smoke/raw_all/images/<file>.jpg
  datasets/fire_smoke/raw_all/labels/<file>.txt
  datasets/ppe/raw_all/images/<file>.jpg
  datasets/ppe/raw_all/labels/<file>.txt
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ultralytics import YOLO
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw"
OUT_PPE = PROJECT_ROOT / "datasets" / "ppe" / "raw_all"
OUT_FS = PROJECT_ROOT / "datasets" / "fire_smoke" / "raw_all"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"

# YOLO class indices for our target schemas
PPE_CLASSES = {0: "person"}                 # MVP: only person
FS_CLASSES = {0: "fire", 1: "smoke"}


def ensure_dirs() -> None:
    for base in (OUT_PPE, OUT_FS):
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "labels").mkdir(parents=True, exist_ok=True)


def write_whole_image_label(label_path: Path, class_id: int) -> None:
    """YOLO format: class_id x_center y_center width height (normalized)."""
    # Whole image = (0.5, 0.5, 1.0, 1.0)
    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def label_fire_smoke() -> dict:
    print("\n=== Labeling fire/smoke (whole-image bbox) ===")
    counts = {"fire": 0, "smoke": 0}
    for class_id, class_name in FS_CLASSES.items():
        src_dir = RAW_ROOT / "fire_smoke" / "raw_images" / class_name
        if not src_dir.exists():
            print(f"  [skip] no source dir: {src_dir}")
            continue
        for img_path in sorted(src_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif"):
                continue
            # Re-encode as jpg to normalize format and avoid weird PNGs
            dst_img = OUT_FS / "images" / f"{class_name}_{img_path.stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] cannot read: {img_path.name}")
                continue
            cv2.imwrite(str(dst_img), img)
            label_path = OUT_FS / "labels" / f"{class_name}_{img_path.stem}.txt"
            write_whole_image_label(label_path, class_id)
            counts[class_name] += 1
    print(f"  fire: {counts['fire']}  smoke: {counts['smoke']}")
    return counts


def label_ppe_person() -> dict:
    """Use YOLO26n to detect persons in PPE images. Drop helmet/no_helmet/etc."""
    print("\n=== Labeling PPE (YOLO26n person detection) ===")
    print(f"  Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    # COCO 'person' class index is 0 in yolo26n
    PERSON_COCO_IDX = 0
    counts = {"person": 0, "images_with_no_person": 0, "images_skipped": 0}
    # Use all 3 subdirs (person/helmet/no_helmet) as sources for "person" images.
    src_root = RAW_ROOT / "ppe" / "raw_images"
    if not src_root.exists():
        print(f"  [skip] no source dir: {src_root}")
        return counts
    seq = 0
    for sub_dir in sorted(src_root.iterdir()):
        if not sub_dir.is_dir():
            continue
        for img_path in sorted(sub_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] cannot read: {img_path.name}")
                counts["images_skipped"] += 1
                continue
            # Run person detection
            results = model.predict(
                source=img, device="cpu", imgsz=640, conf=0.30,
                classes=[PERSON_COCO_IDX], save=False, verbose=False,
            )
            r = results[0]
            n_persons = 0 if r.boxes is None else len(r.boxes)
            if n_persons == 0:
                print(f"  [no-person] {img_path.name}  (skipping)")
                counts["images_with_no_person"] += 1
                continue
            # Save the image as jpg with a sequential name
            stem = f"p{seq:03d}"
            seq += 1
            dst_img = OUT_PPE / "images" / f"{stem}.jpg"
            cv2.imwrite(str(dst_img), img)
            # Write per-box YOLO labels (normalized xyxy -> xywh normalized)
            h, w = img.shape[:2]
            label_lines = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Convert to YOLO xywh normalized
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # Clamp 0..1
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))
                    label_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_path = OUT_PPE / "labels" / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            counts["person"] += n_persons
    print(
        f"  person boxes: {counts['person']}  "
        f"images w/o person: {counts['images_with_no_person']}  "
        f"skipped: {counts['images_skipped']}"
    )
    return counts


def main() -> int:
    ensure_dirs()
    fs_counts = label_fire_smoke()
    ppe_counts = label_ppe_person()
    summary = {
        "fire_smoke": fs_counts,
        "ppe": ppe_counts,
        "notes": [
            "PPE 'helmet' / 'no_helmet' / 'safety_vest' / 'gloves' classes "
            "are DEFERRED pending a manually annotated source dataset "
            "(Hard Hat Workers v2 or GDUT-HD). Auto-labeling cannot reliably "
            "distinguish these classes from COCO.",
            "Fire/smoke labels are whole-image bboxes; the ground truth label "
            "comes from the Wikimedia search query. This is suitable for "
            "pipeline validation but should be replaced with D-Fire for "
            "production training.",
        ],
    }
    out = PROJECT_ROOT / "datasets" / "auto_label_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
