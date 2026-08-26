"""Backfill corrected green/trees and block metrics onto processed GeoPackages.

Recomputes, per city, the metric families affected by three pipeline fixes:

- green/trees ``data_id`` namespacing: a shared RangeIndex let a nearer tree
  point mask a green polygon (and vice versa), corrupting all ``cc_green_*``
  and ``cc_trees_*`` street columns
- ``block_far`` no longer records 0 for blocks whose buildings all lack
  height data (previously ``fillna(0)`` defeated the ``min_count=1`` guard)
- ``block_mean_height`` divides by the area of height-valid buildings only

The street network is rebuilt deterministically from the raw Overture edges;
it is geometry-identical to the stored streets layer, so new values are
joined back by geometry WKB (``ns_node_idx`` is not stable across builds).
Only ``cc_block_*``, ``cc_green_*``, and ``cc_trees_*`` street columns plus
the blocks layer are replaced. The buildings layer is reused as stored, so
building heights are not resampled. Cities missing any of the 11 standard
POI category column families (schema drift before accessibility keys were
fixed) additionally get those columns computed and added.

Usage:
    uv run python -m src.processing.backfill_green_blocks --limit 2 --dry-run
    uv run python -m src.processing.backfill_green_blocks --workers 8
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
from cityseer.network import CityNetwork

from src import landuse_categories, tools
from src.processing import processors
from src.processing.generate_metrics import (
    CLIP_DIST,
    NON_BUILT_CLASSES,
    WORKING_CRS,
    build_green_gdf,
)

logger = tools.get_logger(__name__)

REPLACE_PREFIXES = ("cc_block_", "cc_green_", "cc_trees_")


def _patch_missing_categories(
    cn: CityNetwork, streets_cols: set[str], overture_path: Path
) -> tuple[CityNetwork, list[str]]:
    """Compute accessibility columns for standard POI categories absent from the stored streets.

    Returns the network and the list of newly computed categories (empty if none).
    """
    missing = [
        cat
        for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES
        if not any(c.startswith(f"cc_{cat}_") for c in streets_cols)
    ]
    if not missing:
        return cn, []
    places_gdf = gpd.read_file(overture_path, layer="places").to_crs(WORKING_CRS)
    places_gdf = landuse_categories.merge_landuse_categories(places_gdf)
    cn = cn.compute_accessibilities(
        places_gdf,  # type: ignore
        landuse_column_label="merged_cats",
        accessibility_keys=missing,
        distances=processors.DISTANCES_LU,
        decay_fn=processors.DECAYS,
    )
    processors._restore_column_order(cn.nodes_gdf)
    return cn, missing


def _process_city(task: dict) -> dict:
    bounds_fid = task["bounds_fid"]
    result: dict = {"bounds_fid": bounds_fid, "status": "skipped"}
    metrics_path = Path(task["metrics_path"])
    overture_path = Path(task["overture_path"])
    if not metrics_path.exists() or not overture_path.exists():
        result["status"] = "missing_input"
        return result
    try:
        # stored layers
        streets_gdf = gpd.read_file(metrics_path, layer="streets")
        bldgs_gdf = gpd.read_file(metrics_path, layer="buildings")
        # network rebuild (deterministic geometry)
        clean_edges_gdf = gpd.read_file(overture_path, layer="clean_edges").to_crs(WORKING_CRS)
        boundary = gpd.GeoSeries.from_wkb([task["boundary_wkb"]], crs=WORKING_CRS).iloc[0]
        cn = CityNetwork.from_geopandas(clean_edges_gdf, crs=WORKING_CRS, boundary=boundary)
        # blocks
        blocks_bbox = boundary.buffer(CLIP_DIST).bounds
        blocks_gdf = gpd.read_file(task["blocks_path"], bbox=blocks_bbox).to_crs(WORKING_CRS)
        morph_blocks_gdf = blocks_gdf[~blocks_gdf["class_2021"].isin(NON_BUILT_CLASSES)].copy()
        # block metrics require centroid geometry on buildings
        bldgs_centroids = bldgs_gdf.copy()
        bldgs_centroids["centroid"] = bldgs_centroids.geometry.centroid
        bldgs_centroids.set_geometry("centroid", inplace=True)
        cn, new_blocks_gdf = processors.process_blocks(cn, bldgs_centroids, morph_blocks_gdf)
        # green + trees
        green_gdf = build_green_gdf(blocks_gdf, bldgs_gdf)
        trees_gdf = gpd.read_file(task["trees_path"], bbox=blocks_bbox).to_crs(WORKING_CRS)
        trees_gdf.geometry = trees_gdf.geometry.simplify(2.0)
        cn = processors.process_green(cn, green_gdf, trees_gdf)
        # patch categories missing from the stored schema (pre-fix key drift)
        cn, patched_cats = _patch_missing_categories(cn, set(streets_gdf.columns), overture_path)
        # join new values back onto the stored streets by geometry
        new_streets = cn.to_geopandas()
        new_live = new_streets[new_streets["live"]]
        if len(new_live) != len(streets_gdf):
            result["status"] = f"error_align: {len(new_live)} rebuilt vs {len(streets_gdf)} stored"
            return result
        wkb_to_pos = {geom.wkb: i for i, geom in enumerate(new_live.geometry)}
        positions = []
        for geom in streets_gdf.geometry:
            pos = wkb_to_pos.get(geom.wkb)
            if pos is None:
                result["status"] = "error_align: unmatched geometry"
                return result
            positions.append(pos)
        replace_cols = [
            c
            for c in new_live.columns
            if c.startswith(REPLACE_PREFIXES) or any(c.startswith(f"cc_{cat}_") for cat in patched_cats)
        ]
        aligned = new_live.iloc[positions]
        for col in replace_cols:
            streets_gdf[col] = aligned[col].values
        result["n_streets"] = len(streets_gdf)
        result["n_replaced_cols"] = len(replace_cols)
        result["patched_cats"] = ",".join(patched_cats)
        result["median_far_400"] = float(streets_gdf["cc_block_far_median_400_wt"].median())
        result["median_green_1600"] = float(streets_gdf["cc_green_nearest_max_1600"].median())
        if task.get("dry_run"):
            result["status"] = "dry_run"
            return result
        # write back: replace streets and blocks, keep everything else
        new_blocks_gdf["bounds_fid"] = bounds_fid
        import pyogrio

        available = {name for name, _ in pyogrio.list_layers(metrics_path)}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with zipfile.ZipFile(metrics_path, "r") as zf:
                gpkg_names = [n for n in zf.namelist() if n.endswith(".gpkg")]
                zf.extractall(tmp_dir)
            inner_gpkg = tmp_dir / gpkg_names[0]
            other_layers = {
                lyr: gpd.read_file(inner_gpkg, layer=lyr) for lyr in available if lyr not in ("streets", "blocks")
            }
            inner_gpkg.unlink()
            streets_gdf.to_file(inner_gpkg, driver="GPKG", layer="streets")
            new_blocks_gdf.to_file(inner_gpkg, driver="GPKG", layer="blocks", mode="a")
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
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cities.")
    parser.add_argument("--fids", type=str, default=None, help="Comma-separated bounds_fids to process.")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write.")
    args = parser.parse_args(argv)

    data_dir = tools.get_data_dir()
    bounds_gdf = gpd.read_file(data_dir / "datasets" / "boundaries.gpkg", layer="bounds").to_crs(WORKING_CRS)
    tasks = []
    for _, row in bounds_gdf.iterrows():
        fid = row["bounds_fid"]
        tasks.append(
            {
                "bounds_fid": fid,
                "boundary_wkb": row.geometry.wkb,
                "metrics_path": str(data_dir / "cities_data" / "processed" / f"metrics_{fid}.gpkg.zip"),
                "overture_path": str(data_dir / "cities_data" / "overture" / f"overture_{fid}.gpkg.zip"),
                "blocks_path": str(data_dir / "datasets" / "blocks.gpkg"),
                "trees_path": str(data_dir / "datasets" / "tree_canopies.gpkg"),
                "dry_run": args.dry_run,
            }
        )
    if args.fids:
        keep = {s.strip() for s in args.fids.split(",")}
        tasks = [t for t in tasks if str(t["bounds_fid"]) in keep]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    total = len(tasks)
    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Backfilling green/trees + block metrics for {total} cities ({mode}, workers={args.workers})")

    results = []
    completed = 0
    t0 = time.time()
    if args.workers <= 1:
        for task in tasks:
            r = _process_city(task)
            results.append(r)
            completed += 1
            elapsed = time.time() - t0
            print(f"  {completed}/{total} ({elapsed / completed:.0f}s/city)  {r['bounds_fid']}: {r['status']}")
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
    ok = (results_df["status"].isin(["ok", "dry_run"])).sum()
    errors = results_df["status"].str.startswith("error").sum()
    print(f"\nDone in {time.time() - t0:.0f}s: {ok} ok, {errors} errors")
    if errors:
        for _, row in results_df[results_df["status"].str.startswith("error")].iterrows():
            print(f"  {row['bounds_fid']}: {row['status']}")
    out_csv = data_dir / "paper_data_outputs" / "backfill_green_blocks_report.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Report: {out_csv}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
