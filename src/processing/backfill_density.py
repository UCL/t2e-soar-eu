"""Backfill the street-level population density column with the land-surface floor.

Recomputes ``density`` (persons per km2 of land surface) for every processed
city using the guard added to generate_metrics: census ``land_surface`` is
floored at MIN_LAND_SURFACE_KM2 so populated cells with sliver or zero land
recordings cannot yield impossible densities. The interpolation mirrors
generate_metrics exactly (linear griddata from valid cell centroids to street
midpoints). Only the ``density`` column in the streets layer and the parquet
cache is replaced.

Usage:
    uv run python -m src.processing.backfill_density --workers 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src import tools
from src.processing.generate_metrics import CLIP_DIST, MIN_LAND_SURFACE_KM2, WORKING_CRS

logger = tools.get_logger(__name__)


def _process_city(task: dict) -> dict:
    bounds_fid = task["bounds_fid"]
    result: dict = {"bounds_fid": bounds_fid, "status": "skipped"}
    metrics_path = Path(task["metrics_path"])
    if not metrics_path.exists():
        result["status"] = "missing_input"
        return result
    try:
        stats_gdf = task["stats_gdf"].copy()
        # clean t exactly as generate_metrics does
        invalid = ~np.isfinite(stats_gdf["t"]) | (stats_gdf["t"] < 0)
        stats_gdf.loc[invalid, "t"] = np.nan
        stats_gdf["density"] = stats_gdf["t"] / stats_gdf["land_surface"].clip(lower=MIN_LAND_SURFACE_KM2)

        streets_gdf = gpd.read_file(metrics_path, layer="streets")
        grid_values = stats_gdf["density"].to_numpy(dtype=float)
        valid_mask = np.isfinite(grid_values) & (grid_values >= 0)
        target_coords = np.column_stack((streets_gdf["x"], streets_gdf["y"]))
        if not np.any(valid_mask):
            new_density = np.full(len(streets_gdf), np.nan)
        else:
            grid_coords = np.array([(p.x, p.y) for p in stats_gdf.geometry.centroid])
            new_density = griddata(
                grid_coords[valid_mask],
                grid_values[valid_mask],
                target_coords,
                method="linear",
                fill_value=np.nan,
            )
        old = streets_gdf["density"].to_numpy(dtype=float)
        streets_gdf["density"] = new_density
        result["n_streets"] = len(streets_gdf)
        result["n_changed"] = int(np.sum(~np.isclose(old, new_density, equal_nan=True)))
        result["max_density"] = float(np.nanmax(new_density)) if len(new_density) else np.nan
        if task.get("dry_run"):
            result["status"] = "dry_run"
            return result
        # patch the parquet cache in place if present
        cache_file = Path(task["cache_dir"]) / f"city_{bounds_fid}.parquet"
        if cache_file.exists():
            cache_df = pd.read_parquet(cache_file)
            if "density" in cache_df.columns and len(cache_df) == len(streets_gdf):
                cache_df["density"] = new_density
                cache_df.to_parquet(cache_file, index=False)
        # rewrite the streets layer, preserving all other layers
        import pyogrio

        available = {name for name, _ in pyogrio.list_layers(metrics_path)}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with zipfile.ZipFile(metrics_path, "r") as zf:
                gpkg_names = [n for n in zf.namelist() if n.endswith(".gpkg")]
                zf.extractall(tmp_dir)
            inner_gpkg = tmp_dir / gpkg_names[0]
            other_layers = {lyr: gpd.read_file(inner_gpkg, layer=lyr) for lyr in available if lyr != "streets"}
            inner_gpkg.unlink()
            streets_gdf.to_file(inner_gpkg, driver="GPKG", layer="streets")
            for lyr, lyr_gdf in other_layers.items():
                lyr_gdf.to_file(inner_gpkg, driver="GPKG", layer=lyr, mode="a")
            tmp_zip = tmp_dir / "output.gpkg.zip"
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(inner_gpkg, gpkg_names[0])
            shutil.move(str(tmp_zip), str(metrics_path))
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - operational tool
        import traceback

        result["status"] = f"error: {exc}"
        traceback.print_exc(limit=3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fids", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_dir = tools.get_data_dir()
    bounds_gdf = gpd.read_file(data_dir / "datasets" / "boundaries.gpkg", layer="bounds").to_crs(WORKING_CRS)
    logger.info("Loading population statistics grid...")
    stats_full = gpd.read_file(data_dir / "Eurostat_Census-GRID_2021_V2" / "ESTAT_Census_2021_V2.gpkg")
    stats_full = stats_full.to_crs(WORKING_CRS)
    stats_full = stats_full.rename(columns={c: c.lower() for c in stats_full.columns})
    sindex = stats_full.sindex
    cache_dir = data_dir / "temp_egs" / "shared_cache"

    tasks = []
    for _, row in bounds_gdf.iterrows():
        fid = row["bounds_fid"]
        if args.fids and str(fid) not in {s.strip() for s in args.fids.split(",")}:
            continue
        hit_idx = sindex.query(row.geometry.buffer(CLIP_DIST), predicate="intersects")
        tasks.append(
            {
                "bounds_fid": fid,
                "stats_gdf": stats_full.iloc[hit_idx],
                "metrics_path": str(data_dir / "cities_data" / "processed" / f"metrics_{fid}.gpkg.zip"),
                "cache_dir": str(cache_dir),
                "dry_run": args.dry_run,
            }
        )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    total = len(tasks)
    print(f"Backfilling density for {total} cities (workers={args.workers}, dry_run={args.dry_run})")
    results = []
    completed = 0
    t0 = time.time()
    if args.workers <= 1:
        for task in tasks:
            r = _process_city(task)
            results.append(r)
            completed += 1
            print(f"  {completed}/{total}  {r['bounds_fid']}: {r['status']}", flush=True)
    else:
        max_workers = min(args.workers, max(1, (os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process_city, t): t for t in tasks}
            for future in as_completed(future_map):
                try:
                    r = future.result()
                except Exception as exc:
                    r = {"bounds_fid": future_map[future]["bounds_fid"], "status": f"error: {exc}"}
                results.append(r)
                completed += 1
                elapsed = time.time() - t0
                print(
                    f"  {completed}/{total} ({elapsed / completed:.0f}s/city avg)  "
                    f"{r['bounds_fid']}: {r['status']}",
                    flush=True,
                )
    results_df = pd.DataFrame(results)
    errors = results_df["status"].str.startswith("error").sum()
    print(f"\nDone in {time.time() - t0:.0f}s: {(results_df['status'].isin(['ok', 'dry_run'])).sum()} ok, {errors} errors")
    if "max_density" in results_df.columns:
        print(f"Max density across cities after fix: {results_df['max_density'].max():.0f}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
