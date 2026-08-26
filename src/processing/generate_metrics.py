""" """

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from cityseer.network import CityNetwork
from scipy.interpolate import griddata
from tqdm import tqdm

from src import tools
from src.processing import processors

logger = tools.get_logger(__name__)

# Layers expected to be present in each per-boundary GeoPackage
REQUIRED_LAYERS = [
    "buildings",
    "blocks",
    "streets",
]
WORKING_CRS = 3035
CLIP_DIST = 2000  # Distance (m) to clip stats, blocks, trees around respective bounds
MIN_LAND_SURFACE_KM2 = 0.2  # Floor for census land surface in density computation (bounds the adjustment at 5x)

# Urban Atlas classes excluded from urban morphological (block) metrics
NON_BUILT_CLASSES = {
    "Arable land (annual crops)",
    "Complex and mixed cultivation patterns",
    "Forests",
    "Green urban areas (Public access)",
    "Green urban areas (Private access)",
    "Green urban areas (Unknown access conditions)",
    "Herbaceous vegetation associations (natural grassland, moors...)",
    "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)",
    "Orchards at the fringe of urban classes",
    "Pastures",
    "Permanent crops (vineyards, fruit trees, olive groves)",
    "Sports and leisure facilities",
    "Water",
    "Wetlands",
}

# Urban Atlas classes always treated as green/blue space
GREEN_CLASSES = [
    "Arable land (annual crops)",
    "Complex and mixed cultivation patterns",
    "Forests",
    "Green urban areas (Public access)",
    # "Green urban areas (Private access)",
    "Green urban areas (Unknown access conditions)",
    "Herbaceous vegetation associations (natural grassland, moors...)",
    "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)",
    "Orchards at the fringe of urban classes",
    "Pastures",
    "Permanent crops (vineyards, fruit trees, olive groves)",
    "Water",
    "Wetlands",
]


def build_green_gdf(blocks_gdf: gpd.GeoDataFrame, bldgs_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assemble the green/blue layer from Urban Atlas blocks.

    Includes the always-green classes plus sports/leisure facilities that are
    mostly open (< 20% building coverage).
    """
    always_green = blocks_gdf[blocks_gdf["class_2021"].isin(GREEN_CLASSES)]
    sports_blocks = blocks_gdf[blocks_gdf["class_2021"] == "Sports and leisure facilities"].copy()
    if not sports_blocks.empty and not bldgs_gdf.empty:
        sports_blocks["_block_area"] = sports_blocks.geometry.area
        bldg_in_sports = gpd.sjoin(
            bldgs_gdf[["geometry"]],
            sports_blocks[["geometry", "_block_area"]],
            how="inner",
            predicate="intersects",
        )
        bldg_area_per_block = bldg_in_sports.groupby("index_right").apply(
            lambda g: g.geometry.area.sum(), include_groups=False
        )
        sports_blocks["_bldg_coverage"] = (
            bldg_area_per_block.reindex(sports_blocks.index, fill_value=0).values / sports_blocks["_block_area"].values
        )
        open_sports = sports_blocks[sports_blocks["_bldg_coverage"] < 0.2].drop(
            columns=["_block_area", "_bldg_coverage"]
        )
        logger.info(
            f"Sports/leisure: {len(open_sports)}/{len(sports_blocks)} blocks "
            f"below 20% building coverage, included as green"
        )
    else:
        # no buildings data — include all sports blocks (or empty passthrough)
        open_sports = sports_blocks
    if not open_sports.empty:
        return gpd.GeoDataFrame(pd.concat([always_green, open_sports], ignore_index=True))
    return always_green.copy()


# Census statistic columns interpolated onto street segments
CENSUS_COLS = [
    "density",
    "t",
    "m",
    "f",
    "y_lt15",
    "y_1564",
    "y_ge65",
    "emp",
    "nat",
    "eu_oth",
    "oth",
    "same",
    "chg_in",
    "chg_out",
]


def _process_city(task: dict) -> str:
    """Compute and write the metrics GeoPackage for one city. Returns a status string."""
    bounds_fid = task["bounds_fid"]
    overwrite = task["overwrite"]
    zip_output = task["zip_output"]
    boundary = gpd.GeoSeries.from_wkb([task["boundary_wkb"]], crs=WORKING_CRS).iloc[0]
    stats_gdf = task["stats_gdf"].copy()
    logger.info(f"Processing metrics for bounds fid: {bounds_fid}")
    try:
        overture_path = tools.resolve_gpkg_path(Path(task["overture_data_dir"]) / f"overture_{bounds_fid}.gpkg")
    except FileNotFoundError:
        logger.warning(f"Missing overture file for bounds fid {bounds_fid}, skipping")
        return "missing_overture"
    output_path = Path(task["processed_data_dir"]) / f"metrics_{bounds_fid}.gpkg"
    # NETWORK
    clean_edges_gdf = gpd.read_file(overture_path, layer="clean_edges")
    clean_edges_gdf = clean_edges_gdf.to_crs(WORKING_CRS)
    cn = CityNetwork.from_geopandas(clean_edges_gdf, crs=WORKING_CRS, boundary=boundary)
    # process centrality
    cn = processors.process_centrality(cn)
    # POI
    places_gdf = gpd.read_file(overture_path, layer="places")
    places_gdf = places_gdf.to_crs(WORKING_CRS)
    # infrast
    infrast_gdf = gpd.read_file(overture_path, layer="infrastructure")
    infrast_gdf = infrast_gdf.to_crs(WORKING_CRS)
    cn = processors.process_places(cn, places_gdf, infrast_gdf)
    # buildings
    bldgs_gdf = gpd.read_file(overture_path, layer="buildings")
    bldgs_gdf = bldgs_gdf.to_crs(WORKING_CRS)
    # blocks - load per-boundary with bbox filter to limit memory
    blocks_bbox = boundary.buffer(CLIP_DIST).bounds
    blocks_gdf = gpd.read_file(task["blocks_path"], bbox=blocks_bbox)
    blocks_gdf = blocks_gdf.to_crs(WORKING_CRS)
    logger.info(f"Loaded {len(blocks_gdf)} blocks for bounds fid {bounds_fid}")
    # filter blocks to built-up / urban classes for morphology metrics
    # (green, water, agricultural classes are handled separately via green_gdf)
    morph_blocks_gdf = blocks_gdf[~blocks_gdf["class_2021"].isin(NON_BUILT_CLASSES)].copy()
    logger.info(
        f"Filtered to {len(morph_blocks_gdf)} built blocks "
        f"(excluded {len(blocks_gdf) - len(morph_blocks_gdf)} non-built)"
    )
    # process
    hts_path = Path(task["hts_raster_data_dir"]) / f"bldg_hts_{bounds_fid}.tif"
    if not hts_path.exists():
        logger.warning(
            "Missing building heights raster for bounds fid %s, continuing without height sampling", bounds_fid
        )
        hts_path = None
    cn, bldgs_gdf, morph_blocks_gdf = processors.process_blocks_buildings(cn, bldgs_gdf, morph_blocks_gdf, hts_path)
    if not bldgs_gdf.empty:
        bldgs_gdf["bounds_fid"] = bounds_fid
        if overwrite is True:
            tools.remove_layer_if_exists(output_path, "buildings")
        bldgs_gdf.to_file(output_path, driver="GPKG", layer="buildings")
    if not morph_blocks_gdf.empty:
        morph_blocks_gdf["bounds_fid"] = bounds_fid
        if overwrite is True:
            tools.remove_layer_if_exists(output_path, "blocks")
        morph_blocks_gdf.to_file(output_path, driver="GPKG", layer="blocks")
    # green spaces — include sports/leisure only if mostly open (< 20% building coverage)
    green_gdf = build_green_gdf(blocks_gdf, bldgs_gdf)
    # trees - load per-boundary with bbox filter to limit memory
    trees_gdf = gpd.read_file(task["trees_path"], bbox=blocks_bbox)
    trees_gdf = trees_gdf.to_crs(WORKING_CRS)
    trees_gdf.geometry = trees_gdf.geometry.simplify(2.0)
    logger.info(f"Loaded {len(trees_gdf)} tree canopy features for bounds fid {bounds_fid}")
    cn = processors.process_green(cn, green_gdf, trees_gdf)
    # stats
    logger.info("Computing stats")
    logger.info(f"Received {len(stats_gdf)} stat grid cells within {CLIP_DIST}m of boundary")
    cols = list(CENSUS_COLS)
    # Clean sentinel values (-9999, negatives) BEFORE computing ratios
    # All census statistics should be non-negative counts
    raw_cols = [c for c in cols if c != "density"]  # density computed below
    for col in raw_cols:
        if col in stats_gdf.columns:
            invalid_mask = ~np.isfinite(stats_gdf[col]) | (stats_gdf[col] < 0)
            if invalid_mask.any():
                logger.info(f"Cleaning {invalid_mask.sum()} invalid values in {col}")
                stats_gdf.loc[invalid_mask, col] = np.nan
    # ratios - now safe to compute since negatives/sentinels are NaN.
    # land_surface is floored: some populated cells record near-zero land
    # (sliver/recording errors), which would otherwise yield impossible
    # densities (> 1e9 persons/km2 in the raw grid).
    stats_gdf["density"] = stats_gdf["t"] / stats_gdf["land_surface"].clip(lower=MIN_LAND_SURFACE_KM2)
    perc_cols = []
    for col in cols:
        if col == "density" or col == "t":
            continue
        col_perc = f"{col}_%"
        stats_gdf[col_perc] = stats_gdf[col] / stats_gdf["t"]
        perc_cols.append(col_perc)
    cols.extend(perc_cols)
    # interpolate
    nodes_gdf = cn.nodes_gdf
    grid_coords = np.array([(point.x, point.y) for point in stats_gdf.geometry.centroid])  # type: ignore
    target_coords = np.column_stack((nodes_gdf.x, nodes_gdf.y))  # type: ignore
    for col in tqdm(cols):
        grid_values = stats_gdf[col].values  # type: ignore
        # Report data quality issues
        n_nan = np.sum(~np.isfinite(grid_values))
        n_sentinel = np.sum(grid_values == -9999)  # Common NaN sentinel value
        n_negative = np.sum(np.isfinite(grid_values) & (grid_values < 0) & (grid_values != -9999))
        if n_nan > 0 or n_sentinel > 0 or n_negative > 0:
            logger.info(
                f"Column {col}: {n_nan} NaN/inf, {n_sentinel} sentinel (-9999 as NaN), {n_negative} other negative"
            )
        # Filter out invalid values (NaN, inf, and negative sentinel values like -9999, -9902, etc.)
        # All statistics should be non-negative (counts, densities, percentages)
        valid_mask = np.isfinite(grid_values) & (grid_values >= 0)
        if not np.any(valid_mask):
            logger.warning(f"No valid values for column {col}, skipping interpolation")
            nodes_gdf[col] = np.nan
            continue
        # Only use valid grid points for interpolation
        valid_grid_coords = grid_coords[valid_mask]
        valid_grid_values = grid_values[valid_mask]
        # use linear because cubic goes negative
        # fill_value=np.nan ensures out-of-bounds points get NaN rather than extrapolated values
        nodes_gdf[col] = griddata(
            valid_grid_coords, valid_grid_values, target_coords, method="linear", fill_value=np.nan
        )  # type: ignore
    # export as LineString geometry (street segments) and keep only live
    streets_gdf = cn.to_geopandas()
    if not streets_gdf.empty:
        logger.info("Computing street-frontage metrics")
        frontage_df = processors.compute_street_frontage(streets_gdf, bldgs_gdf)
        for col in frontage_df.columns:
            streets_gdf[col] = frontage_df[col]
        streets_gdf["bounds_fid"] = bounds_fid
        streets_live = streets_gdf.loc[streets_gdf["live"]].copy()
        if not streets_live.empty:
            if overwrite is True:
                tools.remove_layer_if_exists(output_path, "streets")
            streets_live.to_file(output_path, driver="GPKG", layer="streets")
    # compress after all layers written
    if zip_output and output_path.exists():
        tools.compress_gpkg(output_path)
    return "ok"


def process_metrics(
    bounds_in_path: str,
    overture_data_dir: str,
    blocks_path: str,
    trees_path: str,
    hts_raster_data_dir: str,
    stats_path: str,
    processed_data_dir: str,
    overwrite: bool = False,
    zip_output: bool = False,
    workers: int = 1,
    fids: list | None = None,
    limit: int | None = None,
):
    """ """
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(overture_data_dir)
    tools.validate_directory(hts_raster_data_dir)
    tools.validate_directory(processed_data_dir, create=True)
    tools.validate_filepath(blocks_path)
    tools.validate_filepath(trees_path)
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.set_index("bounds_fid")
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    # Load stats once; each task receives its pre-filtered subset
    tools.validate_filepath(stats_path)
    logger.info("Loading population statistics grid...")
    stats_gdf_full = gpd.read_file(stats_path)
    stats_gdf_full = stats_gdf_full.to_crs(WORKING_CRS)
    stats_gdf_full = stats_gdf_full.rename(columns={col: col.lower() for col in stats_gdf_full.columns})
    logger.info(f"Loaded {len(stats_gdf_full)} stat grid cells")
    stats_sindex = stats_gdf_full.sindex

    tasks = []
    for bounds_fid, bounds_row in bounds_gdf.iterrows():
        if fids is not None and str(bounds_fid) not in {str(f) for f in fids}:
            continue
        output_path = Path(processed_data_dir) / f"metrics_{bounds_fid}.gpkg"
        zip_path = output_path.with_suffix(".gpkg.zip")
        if zip_path.exists() and not overwrite:
            logger.info(f"Skipping existing zipped file: {zip_path}")
            continue
        if output_path.exists() and not overwrite:
            if tools.gpkg_has_all_layers(str(output_path), REQUIRED_LAYERS):
                if zip_output and not zip_path.exists():
                    logger.info(f"Zipping existing complete file in place: {output_path}")
                    tools.compress_gpkg(output_path)
                    continue
                logger.info(f"Skipping existing file with all layers: {output_path}")
                continue
            logger.info(f"File missing some layers, will overwrite: {output_path}")
        buffered = bounds_row.geometry.buffer(CLIP_DIST)
        hit_idx = stats_sindex.query(buffered, predicate="intersects")
        tasks.append(
            {
                "bounds_fid": bounds_fid,
                "boundary_wkb": bounds_row.geometry.wkb,
                "stats_gdf": stats_gdf_full.iloc[hit_idx],
                "overture_data_dir": overture_data_dir,
                "blocks_path": blocks_path,
                "trees_path": trees_path,
                "hts_raster_data_dir": hts_raster_data_dir,
                "processed_data_dir": processed_data_dir,
                "overwrite": overwrite,
                "zip_output": zip_output,
            }
        )
    if limit is not None:
        tasks = tasks[:limit]

    total = len(tasks)
    logger.info(f"Processing {total} cities with {workers} worker(s)")
    statuses: list[str] = []
    completed = 0
    t0 = time.time()
    if workers <= 1:
        for task in tasks:
            statuses.append(_process_city(task))
            completed += 1
            logger.info(f"[{completed}/{total}] fid {task['bounds_fid']}: {statuses[-1]}")
    else:
        max_workers = min(workers, max(1, (os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process_city, t): t for t in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    status = future.result()
                except Exception as exc:
                    status = f"error: {exc}"
                statuses.append(status)
                completed += 1
                elapsed = time.time() - t0
                logger.info(
                    f"[{completed}/{total}] fid {task['bounds_fid']}: {status} "
                    f"({elapsed / completed:.0f}s/city avg)"
                )
    n_errors = sum(1 for s in statuses if s.startswith("error"))
    logger.info(f"Done: {total - n_errors} ok, {n_errors} errors in {time.time() - t0:.0f}s")
    if n_errors:
        for task, status in zip(tasks, statuses, strict=False):
            if status.startswith("error"):
                logger.error(f"  fid {task['bounds_fid']}: {status}")


if __name__ == "__main__":
    """
    # All paths default to T2E_DATA_DIR (from .env). No args needed:
    #   python -m src.processing.generate_metrics --zip --overwrite --workers 8
    """
    data_dir = Path(os.environ.get("T2E_DATA_DIR", ""))
    defaults = {
        "bounds_in_path": str(data_dir / "datasets" / "boundaries.gpkg"),
        "overture_data_dir": str(data_dir / "cities_data" / "overture"),
        "blocks_path": str(data_dir / "datasets" / "blocks.gpkg"),
        "trees_path": str(data_dir / "datasets" / "tree_canopies.gpkg"),
        "hts_raster_data_dir": str(data_dir / "cities_data" / "heights"),
        "stats_path": str(data_dir / "Eurostat_Census-GRID_2021_V2" / "ESTAT_Census_2021_V2.gpkg"),
        "processed_data_dir": str(data_dir / "cities_data" / "processed"),
    }
    parser = argparse.ArgumentParser(description="Compute per-city street metrics.")
    parser.add_argument("bounds_in_path", type=str, nargs="?", default=defaults["bounds_in_path"])
    parser.add_argument("overture_data_dir", type=str, nargs="?", default=defaults["overture_data_dir"])
    parser.add_argument("blocks_path", type=str, nargs="?", default=defaults["blocks_path"])
    parser.add_argument("trees_path", type=str, nargs="?", default=defaults["trees_path"])
    parser.add_argument("hts_raster_data_dir", type=str, nargs="?", default=defaults["hts_raster_data_dir"])
    parser.add_argument("stats_path", type=str, nargs="?", default=defaults["stats_path"])
    parser.add_argument("processed_data_dir", type=str, nargs="?", default=defaults["processed_data_dir"])
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--zip", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1, help="Parallel city workers.")
    parser.add_argument("--fids", type=str, default=None, help="Comma-separated bounds_fids to process.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cities.")
    args = parser.parse_args()
    process_metrics(
        bounds_in_path=args.bounds_in_path,
        overture_data_dir=args.overture_data_dir,
        blocks_path=args.blocks_path,
        trees_path=args.trees_path,
        hts_raster_data_dir=args.hts_raster_data_dir,
        stats_path=args.stats_path,
        processed_data_dir=args.processed_data_dir,
        overwrite=args.overwrite,
        zip_output=args.zip,
        workers=args.workers,
        fids=args.fids.split(",") if args.fids else None,
        limit=args.limit,
    )
