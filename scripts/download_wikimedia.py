"""
PS6 Day 2 - Download CC-licensed images from Wikimedia Commons.

This script downloads a small set of CC-licensed images per class
to validate the Day 2 dataset-preparation pipeline end-to-end.

For production training, the user should run the same downstream
scripts (split_dataset.py, validate_dataset.py, visualize_dataset.py)
on the real D-Fire + Hard Hat Workers datasets.

Wikimedia Commons API documentation:
  https://commons.wikimedia.org/w/api.php

All images downloaded are CC-BY, CC-BY-SA, or Public Domain.
The original author and license URL are saved alongside each image
in a sidecar file (<image>.license.json) for attribution tracking.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = (
    "PS6-Safety-AI/0.1 (https://example.local/ps6; contact@example.local) "
    "python-urllib"
)
API = "https://commons.wikimedia.org/w/api.php"

# Class -> search terms on Wikimedia Commons.
# Fire and smoke are visual concepts with many CC images available.
# For PPE, "construction worker" / "hard hat" yield good results.
CLASS_QUERIES = {
    # Fire/smoke dataset
    "fire_smoke/fire": [
        "camp fire flames",
        "forest fire",
        "bonfire night flames",
        "fire flame",
    ],
    "fire_smoke/smoke": [
        "smoke cloud sky",
        "industrial smokestack smoke",
        "smoke dark background",
        "wildfire smoke",
    ],
    # PPE dataset
    "ppe/person": [
        "construction worker portrait",
        "worker safety vest",
        "construction site worker",
        "factory worker",
    ],
    "ppe/helmet": [
        "hard hat construction worker",
        "worker wearing hard hat",
        "yellow hard hat",
        "construction worker helmet",
    ],
    "ppe/no_helmet": [
        "construction worker no helmet",
        "worker without hard hat",
        "factory worker head",
        "warehouse worker",
    ],
}

IMAGES_PER_QUERY = 5
THUMB_WIDTH = 640  # reasonable size for inference + storage


def api_get(params: dict) -> dict:
    headers = {"User-Agent": USER_AGENT}
    full = {"format": "json", "formatversion": 2, **params}
    url = API + "?" + urllib.parse.urlencode(full)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_image_info(title: str) -> dict | None:
    """Use imageinfo API to get the file URL + extmetadata (license + author)."""
    data = api_get(
        {
            "action": "query",
            "titles": f"File:{title}",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "iiurlwidth": THUMB_WIDTH,
        }
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    ii = page.get("imageinfo") or []
    if not ii:
        return None
    info = ii[0]
    ext = info.get("extmetadata", {})
    return {
        "title": title,
        "thumburl": info.get("thumburl") or info.get("url"),
        "origurl": info.get("url"),
        "mime": info.get("mime"),
        "width": info.get("thumbwidth") or info.get("width"),
        "height": info.get("thumbheight") or info.get("height"),
        "license_short": ext.get("LicenseShortName", {}).get("value", "Unknown"),
        "license_long": ext.get("UsageTerms", {}).get("value", "Unknown"),
        "license_url": ext.get("LicenseUrl", {}).get("value", ""),
        "artist_html": ext.get("Artist", {}).get("value", ""),
        "credit_html": ext.get("Credit", {}).get("value", ""),
        "datetime": ext.get("DateTime", {}).get("value", ""),
    }


def search_images(query: str, limit: int) -> list[dict]:
    """Search Wikimedia Commons for files matching the query."""
    print(f"  searching: '{query}' (limit {limit})")
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": 6,  # File namespace
        "srlimit": limit,
    }
    data = api_get(params)
    items = data.get("query", {}).get("search", [])
    results = []
    for item in items:
        title = item["title"].split(":", 1)[-1]
        # Skip non-image types
        if not any(title.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")):
            continue
        info = get_image_info(title)
        if info and info.get("thumburl"):
            results.append(info)
        if len(results) >= limit:
            break
    return results


def strip_html(s: str) -> str:
    """Very simple HTML stripper for attribution fields."""
    import re
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download(url: str, dst: Path, timeout: int = 60) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dst.write_bytes(data)
        return True
    except Exception as e:
        print(f"    download failed: {type(e).__name__}: {e}")
        return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "datasets" / "raw"
    print(f"Raw download root: {root}")
    total_downloaded = 0
    summary = {}
    for class_key, queries in CLASS_QUERIES.items():
        dataset_name, class_name = class_key.split("/")
        out_dir = root / dataset_name / "raw_images" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{class_key}] -> {out_dir}")
        n_for_class = 0
        for q in queries:
            if n_for_class >= IMAGES_PER_QUERY * 2:
                break
            items = search_images(q, IMAGES_PER_QUERY)
            time.sleep(0.3)
            for idx, item in enumerate(items):
                if n_for_class >= IMAGES_PER_QUERY * 2:
                    break
                ext = ".jpg"
                if item.get("mime") == "image/png":
                    ext = ".png"
                elif item.get("mime") == "image/gif":
                    ext = ".gif"
                fname = f"{class_name}_{n_for_class:03d}{ext}"
                dst = out_dir / fname
                if download(item["thumburl"], dst):
                    sidecar = dst.with_suffix(dst.suffix + ".license.json")
                    sidecar.write_text(
                        json.dumps(
                            {
                                "title": item["title"],
                                "source": "Wikimedia Commons",
                                "source_url": item["origurl"],
                                "thumb_url": item["thumburl"],
                                "license_short": item.get("license_short"),
                                "license_long": item.get("license_long"),
                                "license_url": item.get("license_url"),
                                "artist": strip_html(item.get("artist_html")),
                                "credit": strip_html(item.get("credit_html")),
                                "datetime": item.get("datetime"),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    print(f"    {fname}  ({dst.stat().st_size} bytes)")
                    n_for_class += 1
                    total_downloaded += 1
                time.sleep(0.2)
        summary[class_key] = n_for_class
    print(f"\n[OK] Total downloaded: {total_downloaded}")
    print("Per-class counts:")
    for k, v in summary.items():
        print(f"  {k:25s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
