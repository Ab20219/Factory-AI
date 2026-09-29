"""
PS6 Day 1 - Step 5: Environment check.

Reports:
  - Python version
  - PyTorch version
  - Ultralytics version
  - OpenCV version
  - CUDA availability
  - Selected device
  - CPU information

Usage:
  python scripts/check_environment.py
"""

from __future__ import annotations

import platform
import sys


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def section_python() -> None:
    banner("PYTHON")
    print(f"  python version   : {platform.python_version()}")
    print(f"  implementation    : {platform.python_implementation()}")
    print(f"  executable        : {sys.executable}")
    print(f"  platform          : {platform.platform()}")


def section_torch() -> None:
    banner("PYTORCH")
    try:
        import torch
        print(f"  torch version     : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  cuda available    : {cuda_available}")
        if cuda_available:
            print(f"  cuda device count : {torch.cuda.device_count()}")
            print(f"  cuda device name  : {torch.cuda.get_device_name(0)}")
        else:
            print("  cuda device count : 0 (CPU-only build)")
        print(f"  threads           : {torch.get_num_threads()}")
    except Exception as e:
        print(f"  ERROR importing torch: {type(e).__name__}: {e}")


def section_ultralytics() -> None:
    banner("ULTRALYTICS")
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"  ultralytics ver.  : {ultralytics.__version__}")
        # Sanity check: can we construct the YOLO class?
        print(f"  YOLO class loaded : {YOLO.__name__}")
    except Exception as e:
        print(f"  ERROR importing ultralytics: {type(e).__name__}: {e}")


def section_opencv() -> None:
    banner("OPENCV")
    try:
        import cv2
        print(f"  opencv version    : {cv2.__version__}")
        print(f"  build information  : {cv2.getBuildInformation().splitlines()[0]}")
    except Exception as e:
        print(f"  ERROR importing cv2: {type(e).__name__}: {e}")


def section_device() -> None:
    banner("SELECTED DEVICE")
    try:
        import torch
        if torch.cuda.is_available():
            print("  selected device   : cuda:0")
        else:
            print("  selected device   : cpu")
            print("  reason            : CUDA not available (CPU-only PyTorch build)")
            print("  target match      : Intel UHD laptop has no NVIDIA CUDA support")
            print("  inference will use CPU. This is the intended Day 1 configuration.")
    except Exception as e:
        print(f"  ERROR determining device: {type(e).__name__}: {e}")


def section_cpu() -> None:
    banner("CPU INFORMATION")
    try:
        import multiprocessing
        print(f"  cpu count (log.)  : {multiprocessing.cpu_count()}")
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
                model_lines = [ln for ln in lines if ln.startswith("model name")]
                if model_lines:
                    print(f"  model name        : {model_lines[0].split(':', 1)[1].strip()}")
        except FileNotFoundError:
            print("  /proc/cpuinfo not available (non-Linux)")
    except Exception as e:
        print(f"  ERROR reading CPU: {type(e).__name__}: {e}")


def main() -> int:
    print("PS6 Day 1 - Environment Check")
    section_python()
    section_torch()
    section_ultralytics()
    section_opencv()
    section_device()
    section_cpu()
    print("\n" + "=" * 60)
    print("Environment check complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
