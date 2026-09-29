"""
PS6 Day 1 - Step 10: Reusable detection script.

Loads YOLO26n and runs inference on:
  - an image file      : --source path/to/image.jpg
  - a video file       : --source path/to/video.mp4
  - the default webcam: --source 0

Usage examples:
  python scripts/detect.py --source images/test/bus.jpg
  python scripts/detect.py --source videos/test/test.mp4
  python scripts/detect.py --source 0
  python scripts/detect.py --source images/test/bus.jpg --conf 0.4 --imgsz 640

All annotated outputs are written under outputs/<image_test|video_test|webcam>/.
A short JSON-style summary is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PS6 Day 1 reusable YOLO26n detector (CPU inference)."
    )
    p.add_argument(
        "--source",
        required=True,
        help="Image path, video path, or '0' for default webcam.",
    )
    p.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to YOLO model .pt file (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device (default: cpu). Use 'cuda:0' if GPU is available.",
    )
    p.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated output (default: True).",
    )
    p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save annotated output.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display annotated frames in a window (disabled on headless servers).",
    )
    return p.parse_args()


def classify_source(src: str) -> str:
    """Return 'webcam', 'image', or 'video' based on the source string."""
    if src.isdigit():
        return "webcam"
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        return "image"
    if ext in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"):
        return "video"
    raise ValueError(f"Unrecognized source extension: {ext} for {src}")


def output_dir_for(kind: str) -> Path:
    d = PROJECT_ROOT / "outputs" / f"{kind}_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_image(src: str, model, args) -> dict:
    out_dir = output_dir_for("image")
    t0 = time.perf_counter()
    results = model.predict(
        source=src,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        project=str(out_dir),
        name="detect",
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    r = results[0]
    detections = []
    if r.boxes is not None:
        for c, cf, bb in zip(
            r.boxes.cls.tolist(),
            r.boxes.conf.tolist(),
            r.boxes.xyxy.tolist(),
        ):
            detections.append(
                {
                    "class": r.names[int(c)],
                    "confidence": round(float(cf), 4),
                    "bbox": [round(float(v), 1) for v in bb],
                }
            )
    saved_files = []
    if args.save:
        for root, _, files in os.walk(out_dir):
            for f in files:
                fp = Path(root) / f
                if fp.stat().st_mtime >= t0:
                    saved_files.append(str(fp))
    return {
        "kind": "image",
        "source": src,
        "image_shape_hw": list(r.orig_shape),
        "num_detections": len(detections),
        "detections": detections[:20],
        "speed_ms": {
            "preprocess": round(r.speed["preprocess"], 2),
            "inference": round(r.speed["inference"], 2),
            "postprocess": round(r.speed["postprocess"], 2),
        },
        "wall_time_ms": round(elapsed * 1000, 2),
        "saved_files": saved_files,
    }


def run_video(src: str, model, args) -> dict:
    out_dir = output_dir_for("video")
    t0 = time.perf_counter()
    results = model.predict(
        source=src,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        project=str(out_dir),
        name="detect",
        exist_ok=True,
        verbose=False,
        stream=False,
    )
    elapsed = time.perf_counter() - t0
    # results is a list of per-frame Results objects
    n_frames = len(results)
    total_inference_ms = sum(r.speed["inference"] for r in results)
    fps_inference = (
        1000.0 / (total_inference_ms / n_frames) if n_frames > 0 and total_inference_ms > 0 else 0.0
    )
    fps_wall = n_frames / elapsed if elapsed > 0 else 0.0
    class_counts: dict[str, int] = {}
    for r in results:
        if r.boxes is None:
            continue
        for c in r.boxes.cls.tolist():
            name = r.names[int(c)]
            class_counts[name] = class_counts.get(name, 0) + 1
    saved_files = []
    if args.save:
        for root, _, files in os.walk(out_dir):
            for f in files:
                fp = Path(root) / f
                if fp.stat().st_mtime >= t0:
                    saved_files.append(str(fp))
    # First frame shape as proxy for resolution
    first_shape = list(results[0].orig_shape) if results else None
    return {
        "kind": "video",
        "source": src,
        "num_frames_processed": n_frames,
        "frame_shape_hw": first_shape,
        "class_counts": class_counts,
        "total_inference_ms": round(total_inference_ms, 2),
        "avg_inference_ms_per_frame": round(total_inference_ms / n_frames, 2) if n_frames else 0.0,
        "fps_inference_only": round(fps_inference, 2),
        "fps_wall_including_io": round(fps_wall, 2),
        "wall_time_s": round(elapsed, 2),
        "saved_files": saved_files,
    }


def run_webcam(src: str, model, args) -> dict:
    try:
        import cv2
    except ImportError:
        return {"kind": "webcam", "source": src, "result": "WEBCAM TEST SKIPPED (cv2 not available)"}
    cap = cv2.VideoCapture(int(src))
    if not cap.isOpened():
        return {
            "kind": "webcam",
            "source": src,
            "result": "WEBCAM TEST SKIPPED (could not open webcam)",
        }
    # Read 30 frames max for the test
    frame_count = 0
    inference_ms_total = 0.0
    t0 = time.perf_counter()
    while frame_count < 30:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(
            source=frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            save=False,
            verbose=False,
        )
        inference_ms_total += res[0].speed["inference"]
        frame_count += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    if frame_count == 0:
        return {
            "kind": "webcam",
            "source": src,
            "result": "WEBCAM TEST SKIPPED (no frames read)",
        }
    return {
        "kind": "webcam",
        "source": src,
        "frames_captured": frame_count,
        "avg_inference_ms_per_frame": round(inference_ms_total / frame_count, 2),
        "fps_inference_only": round(1000.0 / (inference_ms_total / frame_count), 2),
        "fps_wall": round(frame_count / elapsed, 2),
        "wall_time_s": round(elapsed, 2),
    }


def main() -> int:
    args = parse_args()

    # Load model
    print(f"[detect] Loading model: {args.model}")
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"[detect] Model loaded. Classes: {len(model.names)}")

    # Classify source
    try:
        kind = classify_source(args.source)
    except (FileNotFoundError, ValueError) as e:
        print(f"[detect] ERROR: {e}", file=sys.stderr)
        return 2
    print(f"[detect] Source kind: {kind}")
    print(f"[detect] Device: {args.device}  imgsz: {args.imgsz}  conf: {args.conf}")

    # Run
    if kind == "image":
        summary = run_image(args.source, model, args)
    elif kind == "video":
        summary = run_video(args.source, model, args)
    else:
        summary = run_webcam(args.source, model, args)

    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
