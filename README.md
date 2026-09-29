# PS6 Safety AI

AI-based factory safety monitoring system. Day 1 establishes the
basic computer-vision foundation using YOLO26n pretrained on COCO.

The final system will eventually be:

```
CCTV/Video -> AI Detection -> PPE + Fire/Smoke Detection -> Tracking
          -> Event Engine -> FastAPI -> Database -> Dashboard -> Alerts
```

Day 1 covers ONLY the "AI Detection" box, using a pretrained COCO model
to verify the pipeline. Custom PPE / fire / smoke training is Day 2+.

---

## Project Objective

Factory sites have many safety risks: workers not wearing personal
protective equipment (PPE), fire or smoke outbreaks, intrusions into
restricted zones, etc. Manual monitoring by humans does not scale to
the number of cameras and shifts involved. PS6 is an AI-based system
that monitors CCTV/video feeds in real time and surfaces safety
events to a dashboard and to alert channels (SMS, email, on-site
siren).

Day 1 scope is intentionally minimal: prove the computer-vision
foundation runs on the development laptop using CPU inference.

---

## Day 1 Scope

1. Python environment + virtual environment.
2. Ultralytics + YOLO26n installed and importable.
3. CPU inference verified (no CUDA assumed).
4. Image detection works.
5. Video detection works (baseline FPS measured).
6. Webcam test attempted (skipped gracefully if unavailable).
7. Reusable `detect.py` script.
8. `benchmark.py` for measuring FPS.
9. `requirements.txt`, `README.md`, `.gitignore` in place.
10. Project folder organized for Day 2.

Out of scope for Day 1:
- Custom PPE / fire / smoke model training.
- FastAPI server.
- Database layer.
- Web dashboard.
- SMS / email alerts.

---

## Hardware

Day 1 target: normal laptop with Intel UHD integrated graphics, no
NVIDIA CUDA GPU. Inference runs on CPU only.

Sandbox used for this run (matches the UHD-laptop profile in
capability, though the exact CPU differs):

| Item             | Value                                  |
|------------------|----------------------------------------|
| Platform         | Linux x86_64 (kernel 5.10.x)           |
| Logical CPUs     | 2                                      |
| RAM              | ~4 GB                                  |
| Disk free        | ~9 GB                                  |
| GPU              | none (no CUDA)                         |
| Inference device | CPU                                    |

> The original target machine (Intel UHD laptop) is conceptually
> equivalent: CPU-only inference, no NVIDIA CUDA. Numbers measured
> here should be in the same order of magnitude as the UHD laptop.

---

## Software

| Component      | Version         |
|----------------|-----------------|
| Python         | 3.12.14         |
| Ultralytics    | 8.4.165         |
| YOLO model     | yolo26n.pt      |
| PyTorch        | 2.14.0+cpu      |
| OpenCV         | 5.0.0           |
| NumPy          | 2.5.2           |
| Pillow         | 12.3.0          |

---

## Installation

```bash
# 1. Create project folder (already created here)
cd /home/z/my-project/PS6-Safety-AI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install CPU-only PyTorch FIRST (avoids CUDA bloat)
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 5. Install Ultralytics on top
pip install -U ultralytics

# 6. Verify
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Project Layout

```
PS6-Safety-AI/
|-- models/                # YOLO .pt files
|   `-- yolo26n.pt
|-- datasets/
|   |-- ppe/               # Day 2: PPE dataset
|   `-- fire_smoke/        # Day 2: fire/smoke dataset
|-- videos/test/           # Test videos
|-- images/test/           # Test images
|-- outputs/
|   |-- image_test/        # Annotated image outputs
|   |-- video_test/        # Annotated video outputs
|   |-- benchmark_results.json
|   `-- day1_report.json
|-- scripts/
|   |-- check_environment.py
|   |-- detect.py
|   `-- benchmark.py
|-- backend/               # Day 3+: FastAPI
|-- frontend/              # Day 3+: dashboard
|-- tests/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## Image Detection

```bash
python scripts/detect.py --source images/test/bus.jpg
```

Optional flags: `--conf 0.25`, `--imgsz 640`, `--device cpu`,
`--no-save` to skip writing the annotated output.

Annotated image is written to `outputs/image_test/detect/<filename>`.

Sample result on `bus.jpg` (1080 x 810):
- 5 detections: 1 bus (0.88), 4 persons (0.66 - 0.87)
- Inference: ~74 ms (single cold call) / ~57 ms warm
- FPS (image, warm mean): ~14.5

---

## Video Detection

```bash
python scripts/detect.py --source videos/test/people-detection.mp4
```

Annotated video is written to `outputs/video_test/detect/<filename>.avi`.

Sample result on `people-detection.mp4` (432 x 768, 596 frames):
- Avg inference per frame: 47.4 ms
- FPS (inference only): 21.1
- FPS (wall, including I/O): 18.1
- Total wall: ~33 s for 596 frames

---

## Webcam Detection

```bash
python scripts/detect.py --source 0
```

If no webcam is available the script reports:

```
WEBCAM TEST SKIPPED (could not open webcam)
```

This is treated as a successful skip, not a failure. The sandbox
used for Day 1 has no webcam so this is the expected state.

If your laptop has a webcam, the script will capture 30 frames and
report average inference time and approximate FPS. To process more
frames, edit the `run_webcam` loop limit in `scripts/detect.py`.

If inference is too slow on a real laptop webcam:
1. Lower `--imgsz` (e.g. `--imgsz 320`).
2. Raise `--conf` to reduce post-processing load.
3. Keep the model small (`yolo26n.pt` is the nano variant).

---

## Performance

Measured with `python scripts/benchmark.py --runs 30`:

| Metric                         | Value   |
|--------------------------------|---------|
| Model loading time             | 39.6 ms |
| Mean wall time per image       | 69.1 ms |
| Median wall time per image     | 69.4 ms |
| Std-dev                        | 1.56 ms |
| Min / Max                      | 65.0 / 71.2 ms |
| FPS (mean wall)                | 14.48   |
| Ultralytics inference (warm)   | 57.1 ms |
| Pre + post                     | ~4 ms   |

These are CPU-only numbers on a 2-vCPU sandbox. A real Intel UHD
laptop with 4-8 logical cores is expected to land in a similar or
slightly better range. We do NOT claim "real-time" performance for
this baseline; "real-time" typically means >= 25 FPS sustained,
which the nano COCO model can reach on this hardware for low
resolutions but cannot be guaranteed for the final PPE / fire /
smoke model until that model is trained and benchmarked.

Full machine-readable benchmark:
`outputs/benchmark_results.json`.

---

## Current Limitations

- The pretrained COCO model detects general objects (person, car,
  bus, etc.) but **does NOT detect PPE** (helmets, vests, gloves),
  **fire**, or **smoke**. Custom training is required for those
  classes — that is Day 2.
- No tracking. Each frame is detected independently.
- No event engine. No severity / location / timestamp logic.
- No alerts (SMS / email / siren) are implemented.
- FastAPI server is not implemented.
- Database layer is not implemented.
- Dashboard is not implemented.
- Webcam test skipped in the sandbox (no `/dev/video0`).

---

## Day 2 Plan

1. PPE dataset preparation
   - Source: open PPE datasets (e.g. Hard Hat Workers, SHEL5K).
   - Convert to YOLO format.
   - Define classes: helmet, no-helmet, vest, no-vest, gloves, etc.
2. Fire / smoke dataset preparation
   - Source: D-Fire, FASDD, or similar open datasets.
   - Classes: fire, smoke.
3. Dataset structure: `datasets/ppe/{images,labels}/{train,val}`
   and same for `datasets/fire_smoke/`.
4. Dataset validation: `yolo check-data` style sanity checks
   (missing labels, bad bboxes, class balance).
5. Prepare `data.yaml` files for custom YOLO training.
6. Training will be done on Google Colab or another GPU machine
   (Day 3+). The Intel UHD laptop is NOT used for training.

---

## Day 2 — Dataset Preparation

### Summary

Two YOLO-format datasets prepared and validated end-to-end. Both
pass the validator (0 errors, 0 corrupt images, 0 near-duplicates).

| Dataset | Images (train/val/test) | Classes | Status |
|---------|-------------------------|--------|--------|
| PPE     | 14 / 2 / 1 (total 17)   | 0: person (MVP) | PASS |
| Fire/Smoke | 16 / 2 / 2 (total 20) | 0: fire, 1: smoke | PASS |

> **Demo subset, not production training data.** See
> `datasets/DATASET_REPORT.md` for the full report including the
> rationale for the small sample size and the production path
> (D-Fire + Hard Hat Workers v2).

### PPE

- Source: Wikimedia Commons (CC BY / CC BY-SA / Public Domain)
- License per-image: sidecar `images/<name>.license.json`
- Images: 17 (after cleaning 30 → 17: 6 had no detectable person, 7 quarantined as duplicates)
- Classes: `0: person` only (MVP)
- Deferred: `helmet`, `no_helmet`, `safety_vest`, `no_safety_vest`, `gloves` — pending Hard Hat Workers v2 / GDUT-HD
- Annotations: auto-generated by YOLO26n (COCO `person` class, conf ≥ 0.30)

### Fire/Smoke

- Source: Wikimedia Commons (CC BY / CC BY-SA / Public Domain)
- License per-image: sidecar `images/<name>.license.json`
- Images: 20 (10 fire + 10 smoke)
- Classes: `0: fire`, `1: smoke` (matches D-Fire schema for drop-in replacement)
- Annotations: whole-image bbox; ground-truth label = Wikimedia search query

### Dataset validation

Both datasets pass `scripts/validate_dataset.py`:
- 0 image-read failures
- 0 missing label files
- 0 label syntax errors
- 0 out-of-range coordinates
- 0 near-duplicate perceptual hashes (after quarantine)

Validation reports: `datasets/ppe/validation_report.json` and `datasets/fire_smoke/validation_report.json`.

### Day 2 limitations

1. Demo dataset sizes (17 + 20 images) are insufficient for production training.
2. PPE labels are auto-generated by COCO-pretrained YOLO26n; not manually verified.
3. Fire/smoke labels are whole-image bbox, not precise fire-boundary localization.
4. PPE classes `helmet`/`no_helmet`/`safety_vest`/`gloves` are deferred — no quality CC-licensed public source without manual annotation.
5. Class imbalance in fire/smoke val/test (small-n split artifact; production D-Fire won't have this).

### Production path (Day 3 preparation)

To turn the demo datasets into production-ready training sets:

**Fire/smoke**: Download D-Fire from Kaggle (~1 GB, requires free Kaggle account). Place under `datasets/raw/fire_smoke/` and re-run `scripts/split_dataset.py`, `scripts/validate_dataset.py`, `scripts/visualize_dataset.py`. The `fire_smoke.yaml` already uses the D-Fire class schema (0=fire, 1=smoke), so it's a drop-in replacement.

**PPE**: Download Hard Hat Workers v2 from Roboflow Universe (free, requires API key). Place under `datasets/raw/ppe/`, update `datasets/class_mapping.yaml` to map HHW source classes to target classes, run a small remap script, then re-run the pipeline. Update `ppe.yaml` to enable all 6 target classes.

### Day 2 files added

```
datasets/
|-- raw/{ppe,fire_smoke}/raw_images/<class>/   (CC source images + .license.json sidecars)
|-- ppe/raw_all/{images,labels}/               (17 normalized images + YOLO labels)
|-- ppe/{images,labels}/{train,val,test}/       (14/2/1 split)
|-- ppe/ppe.yaml
|-- ppe/validation_report.json
|-- fire_smoke/raw_all/{images,labels}/         (20 images + YOLO labels)
|-- fire_smoke/{images,labels}/{train,val,test}/ (16/2/2 split)
|-- fire_smoke/fire_smoke.yaml
|-- fire_smoke/validation_report.json
|-- invalid/ppe/duplicates/                      (7 quarantined near-duplicates + manifest.json)
|-- class_mapping.yaml
|-- auto_label_summary.json
`-- DATASET_REPORT.md

scripts/
|-- download_wikimedia.py     (Wikimedia Commons CC image downloader)
|-- auto_label.py             (YOLO26n person detector + fire/smoke whole-image labeler)
|-- split_dataset.py          (deterministic 70/20/10 or 80/10/10 splitter)
|-- validate_dataset.py       (full YOLO format validator + phash duplicate detector)
|-- visualize_dataset.py      (renders bbox overlays for manual inspection)
`-- quarantine_duplicates.py  (moves detected duplicates to datasets/invalid/)

outputs/dataset_preview/
|-- ppe/{train,val,test}/*.png        (sample annotated images)
`-- fire_smoke/{train,val,test}/*.png (sample annotated images)
```

---

## Day 2 Success Checklist

- [x] PPE dataset obtained (demo subset; production swap-in path documented)
- [x] Fire/smoke dataset obtained (demo subset; production swap-in path documented)
- [x] Dataset licenses recorded (per-image `.license.json` sidecars)
- [x] Source URLs recorded (in sidecar files)
- [x] Target classes finalized (PPE MVP: person; fire/smoke: fire, smoke)
- [x] Class mapping created (`datasets/class_mapping.yaml`)
- [x] YOLO annotation format verified (validator passes with 0 errors)
- [x] Data cleaned (corrupt images removed, duplicates quarantined)
- [x] Invalid data quarantined (`datasets/invalid/ppe/duplicates/`)
- [x] Train split created
- [x] Validation split created
- [x] Test split created
- [x] No obvious data leakage (perceptual-hash duplicate check passes)
- [x] `ppe.yaml` created
- [x] `fire_smoke.yaml` created
- [x] Dataset validator works
- [x] Dataset validation passes (both datasets PASS)
- [x] Sample annotations visually checked (17 PNGs in `outputs/dataset_preview/`)
- [x] Dataset statistics generated (`DATASET_REPORT.md` + JSON sidecars)
- [x] DATASET_REPORT.md created
- [x] README updated (this section)

---

## Day 1 Success Checklist

- [x] Python environment works
- [x] Virtual environment works
- [x] Ultralytics imports successfully
- [x] YOLO26n loads successfully
- [x] CPU inference works
- [x] Test image detection works
- [x] Annotated image is produced
- [x] Test video detection works
- [x] Annotated video is produced
- [x] Webcam works OR is documented as unavailable
- [x] FPS / performance is measured
- [x] `check_environment.py` works
- [x] `detect.py` works
- [x] `benchmark.py` works
- [x] `requirements.txt` exists
- [x] `README.md` exists
- [x] Project folder is organized
