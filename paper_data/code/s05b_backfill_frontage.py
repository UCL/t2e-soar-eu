#!/usr/bin/env python3
"""Backfill bilateral street-frontage ratio onto existing processed GeoPackages.

Computes edge-based bilateral frontage for each street segment:
- Buffers each street by 35m to define a corridor
- For each building, only edges fully contained in the corridor contribute
- Edges are classified left/right by cross product with street direction
- Adaptive end-trimming avoids junction contamination
- Score = max(left_coverage, right_coverage)

Also updates the parquet cache used by atlas scripts.

Examples:
    uv run python paper_data/code/s05b_backfill_frontage.py --limit 5 --dry-run
    uv run python paper_data/code/s05b_backfill_frontage.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from config import BOUNDS_PATH, PROCESSED_DIR

# Atlas cache directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper_research" / "code"))
from shared_config import SHARED_CACHE_DIR

# Frontage computation lives in the main processing module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "processing"))
from processors import compute_street_frontage

# Must match the canonical default in processors.compute_street_frontage (35 m),
# which is what generate_metrics uses and what the data paper documents.
BUFFER_DIST = 35.0


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------


def _process_city(task: dict) -> dict:
    bounds_fid = str(task["bounds_fid"])
    gpkg_path = Path(str(task["metrics_path"]))
    dry_run = task.get("dry_run", False)
    buffer_dist = float(task.get("buffer_dist", BUFFER_DIST))
    cache_dir = Path(str(task.get("cache_dir", "")))

    result = {"bounds_fid": bounds_fid, "status": "skipped", "n_streets": 0, "mean_frontage": np.nan}

    if not gpkg_path.exists():
        result["status"] = "missing"
        return result

    try:
        available = {name for name, _ in pyogrio.list_layers(gpkg_path)}
    except Exception:
        result["status"] = "error_layers"
        return result

    if "streets" not in available or "buildings" not in available:
        result["status"] = "missing_layers"
        return result

    try:
        streets_gdf = gpd.read_file(gpkg_path, layer="streets")
        bldgs_gdf = gpd.read_file(gpkg_path, layer="buildings")
    except Exception as exc:
        result["status"] = f"error_read: {exc}"
        return result

    if streets_gdf.empty:
        result["status"] = "empty_streets"
        return result

    try:
        fr_df = compute_street_frontage(streets_gdf, bldgs_gdf, buffer_dist=buffer_dist)

        result["n_streets"] = len(streets_gdf)
        result["mean_frontage"] = float(fr_df["frontage_max"].mean()) if not fr_df["frontage_max"].isna().all() else np.nan

        if dry_run:
            result["status"] = "dry_run"
            return result

        # Note: a rerun refreshes only the four frontage ratio columns, in
        # both the cache and the gpkg streets layer.  frontage_edges_left/
        # right (also returned by compute_street_frontage) are not written
        # here: in the gpkg they keep the values from the original
        # generate_metrics run, and the parquet cache does not carry them.

        # Write to parquet cache
        cache_file = cache_dir / f"city_{bounds_fid}.parquet"
        if cache_file.exists():
            cache_df = pd.read_parquet(cache_file)
            n = len(cache_df)
            for col in ("frontage_max", "frontage_avg", "frontage_left", "frontage_right"):
                cache_df[col] = fr_df[col].values[:n]
            cache_df.drop(columns=["frontage_ratio"], errors="ignore", inplace=True)
            cache_df.to_parquet(cache_file, index=False)

        # Write to GeoPackage (handles .gpkg.zip)
        for col in ("frontage_max", "frontage_avg", "frontage_left", "frontage_right"):
            streets_gdf[col] = fr_df[col].values
        # Remove legacy column
        streets_gdf.drop(columns=["frontage_ratio"], errors="ignore", inplace=True)

        is_zip = str(gpkg_path).endswith(".gpkg.zip")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            if is_zip:
                # Extract .gpkg from zip
                with zipfile.ZipFile(gpkg_path, "r") as zf:
                    gpkg_names = [n for n in zf.namelist() if n.endswith(".gpkg")]
                    zf.extractall(tmp_dir)
                inner_gpkg = tmp_dir / gpkg_names[0]
            else:
                inner_gpkg = tmp_dir / gpkg_path.name
                shutil.copy2(gpkg_path, inner_gpkg)

            # Read other layers, rewrite all
            other_layers = {}
            for lyr in available:
                if lyr != "streets":
                    other_layers[lyr] = gpd.read_file(inner_gpkg, layer=lyr)
            inner_gpkg.unlink()
            streets_gdf.to_file(inner_gpkg, driver="GPKG", layer="streets")
            for lyr, lyr_gdf in other_layers.items():
                lyr_gdf.to_file(inner_gpkg, driver="GPKG", layer=lyr, mode="a")

            if is_zip:
                # Re-zip
                tmp_zip = tmp_dir / "output.gpkg.zip"
                with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(inner_gpkg, gpkg_names[0])
                shutil.move(str(tmp_zip), str(gpkg_path))
            else:
                shutil.move(str(inner_gpkg), str(gpkg_path))

        result["status"] = "ok"

    except Exception as exc:
        result["status"] = f"error_compute: {exc}"
        traceback.print_exc(limit=1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N cities.")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write.")
    parser.add_argument(
        "--buffer", type=float, default=BUFFER_DIST, help=f"Buffer distance in metres (default {BUFFER_DIST})."
    )
    args = parser.parse_args(argv)

    buffer_dist = args.buffer

    bounds = pyogrio.read_dataframe(BOUNDS_PATH, columns=["bounds_fid", "label", "country"], read_geometry=False)
    bounds["bounds_fid"] = bounds["bounds_fid"].astype(str)

    tasks = [
        {
            "bounds_fid": row.bounds_fid,
            "city_label": row.label,
            "country": row.country,
            "metrics_path": str(PROCESSED_DIR / f"metrics_{row.bounds_fid}.gpkg.zip"),
            "dry_run": args.dry_run,
            "buffer_dist": buffer_dist,
            "cache_dir": str(SHARED_CACHE_DIR),
        }
        for row in bounds.itertuples(index=False)
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    total = len(tasks)
    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Backfilling frontage_max for {total} cities ({mode}, buffer={buffer_dist}m, workers={args.workers})")

    results = []
    completed = 0
    t0 = time.time()

    if args.workers <= 1:
        for task in tasks:
            r = _process_city(task)
            results.append(r)
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{total}  ({rate:.1f} cities/s)  last: {r['bounds_fid']} {r['status']}")
    else:
        max_workers = min(args.workers, max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process_city, t): t for t in tasks}
            for future in as_completed(future_map):
                try:
                    r = future.result()
                    results.append(r)
                except Exception as exc:
                    task = future_map[future]
                    results.append({"bounds_fid": task["bounds_fid"], "status": f"error: {exc}"})
                completed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  {completed}/{total}  ({rate:.1f} cities/s)")

    elapsed = time.time() - t0

    results_df = pd.DataFrame(results)
    ok = (results_df["status"] == "ok").sum()
    dry = (results_df["status"] == "dry_run").sum()
    errors = results_df["status"].str.startswith("error").sum()
    mean_fr = results_df["mean_frontage"].mean()

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  OK:       {ok}")
    if dry > 0:
        print(f"  Dry run:  {dry}")
    print(f"  Errors:   {errors}")
    print(f"  Mean frontage ratio across cities: {mean_fr:.3f}")

    if errors > 0:
        print("\n  Errors:")
        for _, row in results_df[results_df["status"].str.startswith("error")].iterrows():
            print(f"    {row['bounds_fid']}: {row['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
