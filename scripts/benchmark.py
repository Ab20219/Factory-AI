"""
PS6 Day 1 - Step 11: Performance benchmark.

Measures:
  - Model loading time
  - Image inference time (single + repeat)
  - Approximate FPS (warm inference)
  - Image size
  - Device

Usage:
  python scripts/benchmark.py                      # uses default test image
  python scripts/benchmark.py --source path/to/img
  python scripts/benchmark.py --runs 50           # change warmup+run count
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "test" / "bus.jpg"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS6 Day 1 benchmark.")
    p.add_argument("--source", default=str(DEFAULT_IMAGE), help="Test image path.")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Model .pt path.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    p.add_argument("--device", default="cpu", help="Device (default cpu).")
    p.add_argument("--warmup", type=int, default=3, help="Warmup runs.")
    p.add_argument("--runs", type=int, default=30, help="Measured runs.")
    return p.parse_args()


def get_torch_device_info() -> dict:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
        }
    except Exception as e:
        return {"error": str(e)}


def get_ultralytics_version() -> str:
    try:
        import ultralytics
        return ultralytics.__version__
    except Exception as e:
        return f"ERROR: {e}"


def get_opencv_version() -> str:
    try:
        import cv2
        return cv2.__version__
    except Exception as e:
        return f"ERROR: {e}"


def get_image_size(path: str) -> tuple[int, int]:
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return (0, 0)
        h, w = img.shape[:2]
        return (h, w)
    except Exception:
        return (0, 0)


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("PS6 Day 1 - Performance Benchmark")
    print("=" * 60)

    print("\n[ENV]")
    print(f"  python       : {platform.python_version()}")
    print(f"  ultralytics  : {get_ultralytics_version()}")
    print(f"  opencv       : {get_opencv_version()}")
    info = get_torch_device_info()
    print(f"  torch        : {info.get('torch_version', '?')}")
    print(f"  cuda avail.  : {info.get('cuda_available', '?')}")
    print(f"  num threads  : {info.get('num_threads', '?')}")
    print(f"  device       : {args.device}")
    print(f"  imgsz        : {args.imgsz}")
    print(f"  model        : {args.model}")
    print(f"  source       : {args.source}")

    h, w = get_image_size(args.source)
    print(f"  image (HxW)  : {h} x {w}")

    # 1. Model loading time
    print("\n[1] Model loading time")
    from ultralytics import YOLO
    t0 = time.perf_counter()
    model = YOLO(args.model)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  load time    : {load_ms:.1f} ms")

    # 2. Warmup
    print(f"\n[2] Warmup ({args.warmup} runs)")
    for _ in range(args.warmup):
        model.predict(
            source=args.source, device=args.device, imgsz=args.imgsz,
            save=False, verbose=False,
        )

    # 3. Measured runs
    print(f"\n[3] Measured runs ({args.runs})")
    times_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        results = model.predict(
            source=args.source, device=args.device, imgsz=args.imgsz,
            save=False, verbose=False,
        )
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    fps_mean = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    # Ultralytics built-in speed (from last run)
    last = results[0].speed
    last_total_ms = sum(last.values())

    print(f"  mean wall    : {mean_ms:.2f} ms")
    print(f"  median wall  : {median_ms:.2f} ms")
    print(f"  stdev        : {stdev_ms:.2f} ms")
    print(f"  min / max    : {min_ms:.2f} / {max_ms:.2f} ms")
    print(f"  FPS (mean)   : {fps_mean:.2f}")
    print(f"\n  (last-run Ultralytics speed)")
    print(f"  preprocess   : {last['preprocess']:.2f} ms")
    print(f"  inference    : {last['inference']:.2f} ms")
    print(f"  postprocess  : {last['postprocess']:.2f} ms")
    print(f"  total (UL)   : {last_total_ms:.2f} ms")

    summary = {
        "env": {
            "python": platform.python_version(),
            "ultralytics": get_ultralytics_version(),
            "opencv": get_opencv_version(),
            "torch": info.get("torch_version"),
            "cuda_available": info.get("cuda_available"),
            "num_threads": info.get("num_threads"),
            "platform": platform.platform(),
            "device_used": args.device,
        },
        "model": {
            "path": args.model,
            "load_ms": round(load_ms, 2),
            "num_classes": len(model.names),
        },
        "image": {
            "path": args.source,
            "height": h,
            "width": w,
        },
        "inference": {
            "imgsz": args.imgsz,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "wall_ms_mean": round(mean_ms, 2),
            "wall_ms_median": round(median_ms, 2),
            "wall_ms_stdev": round(stdev_ms, 2),
            "wall_ms_min": round(min_ms, 2),
            "wall_ms_max": round(max_ms, 2),
            "fps_mean_wall": round(fps_mean, 2),
            "ultralytics_last_speed_ms": {
                "preprocess": round(last["preprocess"], 2),
                "inference": round(last["inference"], 2),
                "postprocess": round(last["postprocess"], 2),
                "total": round(last_total_ms, 2),
            },
        },
    }

    out_path = PROJECT_ROOT / "outputs" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Full benchmark saved to {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
