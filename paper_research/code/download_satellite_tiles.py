#!/usr/bin/env python3
"""Download satellite tiles from Esri World Imagery for exemplar locations.

Reads satellite_metadata.json and downloads tiles at the maximum resolution
supported by the Esri World Imagery REST API (no API key required).

    uv run python paper_research/code/download_satellite_tiles.py
    uv run python paper_research/code/download_satellite_tiles.py --only LHL
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from pyproj import Transformer

META_PATH = Path(__file__).resolve().parent.parent / "outputs" / "atlas" / "satellite_metadata.json"
SAT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "atlas" / "satellites"

# Esri World Imagery export endpoint
ESRI_URL = (
    "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
)

# Approximate metres per side at the exemplar zoom level
EXTENT_M = 1000
# Max resolution the API serves in Europe is ~0.3 m/pixel (zoom 19)
TILE_SIZE = 1500  # 1000m / 1500px ≈ 0.67 m/pixel

# EPSG:3857 for web mercator bbox
tr_4326_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)


def download_tile(lat: float, lon: float, out_path: Path) -> bool:
    """Download a single satellite tile centred on lat/lon."""
    cx, cy = tr_4326_to_3857.transform(lon, lat)
    half = EXTENT_M / 2
    bbox = f"{cx - half},{cy - half},{cx + half},{cy + half}"

    params = (
        f"?bbox={bbox}"
        f"&bboxSR=3857"
        f"&imageSR=3857"
        f"&size={TILE_SIZE},{TILE_SIZE}"
        f"&format=png"
        f"&f=image"
    )
    url = ESRI_URL + params

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 * attempt)
        try:
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 5000:
                continue
            out_path.write_bytes(data)
            return True
        except Exception as e:
            if attempt == 2:
                print(f" [{e}]", end="", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only download tiles for this octant (e.g. LHL)")
    parser.add_argument("--force", action="store_true", help="Re-download existing tiles")
    args = parser.parse_args()

    with open(META_PATH) as f:
        meta = json.load(f)

    SAT_DIR.mkdir(parents=True, exist_ok=True)

    for m in meta:
        octant = m["octant"]
        exemplar = m["exemplar"]

        if args.only and octant != args.only:
            continue

        safe_city = m["city"].replace(" ", "_").replace("/", "_")
        out = SAT_DIR / f"sat_{octant}_{safe_city}.png"

        if out.exists() and not args.force:
            print(f"  {out.name} — exists, skipping")
            continue

        print(f"  {out.name} — downloading ({m['lat']}, {m['lon']})...", end="", flush=True)
        ok = download_tile(m["lat"], m["lon"], out)
        if ok:
            print(f" OK ({out.stat().st_size / 1024:.0f} KB)")
        else:
            print(" FAILED")

        time.sleep(1)

    print("Done.")


if __name__ == "__main__":
    main()
