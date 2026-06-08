""" """

import argparse
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
):
    """ """
    tools.validate_filepath(bounds_in_path)
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.set_index("bounds_fid")
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    # Load stats once outside loop for efficiency
    tools.validate_filepath(stats_path)
    logger.info("Loading population statistics grid...")
    stats_gdf_full = gpd.read_file(stats_path)
    stats_gdf_full = stats_gdf_full.to_crs(WORKING_CRS)
    stats_gdf_full = stats_gdf_full.rename(columns={col: col.lower() for col in stats_gdf_full.columns})
    logger.info(f"Loaded {len(stats_gdf_full)} stat grid cells")
    # Blocks loaded per-boundary inside loop (bbox filter) to avoid OOM
    tools.validate_filepath(blocks_path)
    # Trees loaded per-boundary inside loop (bbox filter) to avoid OOM
    tools.validate_filepath(trees_path)
    # process each boundary
    for bounds_fid, bounds_row in bounds_gdf.iterrows():
        logger.info(f"\n\nProcessing metrics for bounds fid: {bounds_fid}")
        tools.validate_directory(overture_data_dir)
        try:
            overture_path = tools.resolve_gpkg_path(Path(overture_data_dir) / f"overture_{bounds_fid}.gpkg")
        except FileNotFoundError:
            logger.warning(f"Missing overture file for bounds fid {bounds_fid}, skipping")
            continue
        # output path
        tools.validate_directory(processed_data_dir, create=True)
        output_path = Path(processed_data_dir) / f"metrics_{bounds_fid}.gpkg"
        zip_path = output_path.with_suffix(".gpkg.zip")
        # check if already exists
        if zip_path.exists() and not overwrite:
            logger.info(f"Skipping existing zipped file: {zip_path}")
            continue
        if output_path.exists() and not overwrite:
            has_all = tools.gpkg_has_all_layers(str(output_path), REQUIRED_LAYERS)
            if has_all:
                if zip_output and not zip_path.exists():
                    logger.info(f"Zipping existing complete file in place: {output_path}")
                    tools.compress_gpkg(output_path)
                    continue
                logger.info(f"Skipping existing file with all layers: {output_path}")
                continue
            else:
                logger.info(f"File missing some layers, will overwrite: {output_path}")
        # NETWORK
        clean_edges_gdf = gpd.read_file(overture_path, layer="clean_edges")
        clean_edges_gdf = clean_edges_gdf.to_crs(WORKING_CRS)
        cn = CityNetwork.from_geopandas(clean_edges_gdf, crs=WORKING_CRS, boundary=bounds_row.geometry)
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
        blocks_bbox = bounds_row.geometry.buffer(CLIP_DIST).bounds
        blocks_gdf = gpd.read_file(blocks_path, bbox=blocks_bbox)
        blocks_gdf = blocks_gdf.to_crs(WORKING_CRS)
        logger.info(f"Loaded {len(blocks_gdf)} blocks for bounds fid {bounds_fid}")
        # filter blocks to built-up / urban classes for morphology metrics
        # (green, water, agricultural classes are handled separately via green_gdf)
        non_built_classes = {
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
        morph_blocks_gdf = blocks_gdf[~blocks_gdf["class_2021"].isin(non_built_classes)].copy()
        logger.info(
            f"Filtered to {len(morph_blocks_gdf)} built blocks "
            f"(excluded {len(blocks_gdf) - len(morph_blocks_gdf)} non-built)"
        )
        # process
        tools.validate_directory(hts_raster_data_dir)
        hts_path = Path(hts_raster_data_dir) / f"bldg_hts_{bounds_fid}.tif"
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
        green_classes = [
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
        always_green = blocks_gdf[blocks_gdf["class_2021"].isin(green_classes)]
        # sports/leisure: compute building coverage and include only open facilities
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
                bldg_area_per_block.reindex(sports_blocks.index, fill_value=0).values
                / sports_blocks["_block_area"].values
            )
            open_sports = sports_blocks[sports_blocks["_bldg_coverage"] < 0.2].drop(
                columns=["_block_area", "_bldg_coverage"]
            )
            logger.info(
                f"Sports/leisure: {len(open_sports)}/{len(sports_blocks)} blocks "
                f"below 20% building coverage, included as green"
            )
        elif not sports_blocks.empty:
            # no buildings data — include all sports blocks
            open_sports = sports_blocks
        else:
            open_sports = sports_blocks  # empty
        green_gdf = (
            gpd.GeoDataFrame(pd.concat([always_green, open_sports], ignore_index=True))
            if not open_sports.empty
            else always_green.copy()
        )
        # trees - load per-boundary with bbox filter to limit memory
        tree_bbox = bounds_row.geometry.buffer(CLIP_DIST).bounds
        trees_gdf = gpd.read_file(trees_path, bbox=tree_bbox)
        trees_gdf = trees_gdf.to_crs(WORKING_CRS)
        trees_gdf.geometry = trees_gdf.geometry.simplify(2.0)
        logger.info(f"Loaded {len(trees_gdf)} tree canopy features for bounds fid {bounds_fid}")
        cn = processors.process_green(cn, green_gdf, trees_gdf)
        # stats
        logger.info("Computing stats")
        # Filter stats to within buffer of boundary to focus interpolation on locally-relevant data
        stats_gdf = stats_gdf_full[stats_gdf_full.intersects(bounds_row.geometry.buffer(CLIP_DIST))].copy()
        logger.info(f"Retained {len(stats_gdf)} stat grid cells within {CLIP_DIST}m of boundary")
        cols = [
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
        # Clean sentinel values (-9999, negatives) BEFORE computing ratios
        # All census statistics should be non-negative counts
        raw_cols = [c for c in cols if c != "density"]  # density computed below
        for col in raw_cols:
            if col in stats_gdf.columns:
                invalid_mask = ~np.isfinite(stats_gdf[col]) | (stats_gdf[col] < 0)
                if invalid_mask.any():
                    logger.info(f"Cleaning {invalid_mask.sum()} invalid values in {col}")
                    stats_gdf.loc[invalid_mask, col] = np.nan
        # ratios - now safe to compute since negatives/sentinels are NaN
        stats_gdf["density"] = stats_gdf["t"] / stats_gdf["land_surface"]
        for col in cols:
            if col == "density" or col == "t" or "%" in col:
                continue
            col_perc = f"{col}_%"
            stats_gdf[col_perc] = stats_gdf[col] / stats_gdf["t"]
            cols.append(col_perc)  # guard against re-adding
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


if __name__ == "__main__":
    """
    # All paths default to T2E_DATA_DIR (from .env). No args needed:
    #   python -m src.processing.generate_metrics --zip --overwrite
    """
    if True:
        import os

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
        parser = argparse.ArgumentParser(description="Load overture networks.")
        parser.add_argument(
            "bounds_in_path",
            type=str,
            nargs="?",
            default=defaults["bounds_in_path"],
            help="Input data directory with boundary GPKG.",
        )
        parser.add_argument(
            "overture_data_dir",
            type=str,
            nargs="?",
            default=defaults["overture_data_dir"],
            help="Input data directory for overture GPKG files.",
        )
        parser.add_argument(
            "blocks_path",
            type=str,
            nargs="?",
            default=defaults["blocks_path"],
            help="Input data directory with Urban Atlas blocks GPKG.",
        )
        parser.add_argument(
            "trees_path",
            type=str,
            nargs="?",
            default=defaults["trees_path"],
            help="Input data directory with Urban Atlas tree canopy GPKG.",
        )
        parser.add_argument(
            "hts_raster_data_dir",
            type=str,
            nargs="?",
            default=defaults["hts_raster_data_dir"],
            help="Input data directory with building height raster files.",
        )
        parser.add_argument(
            "stats_path",
            type=str,
            nargs="?",
            default=defaults["stats_path"],
            help="Input data directory with population stats GPKG.",
        )
        parser.add_argument(
            "processed_data_dir",
            type=str,
            nargs="?",
            default=defaults["processed_data_dir"],
            help="Output data directory for metrics GPKG files.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files (default: False)",
            default=False,
        )
        parser.add_argument(
            "--zip",
            action="store_true",
            help="Compress each output GeoPackage to .gpkg.zip after processing (default: False)",
            default=False,
        )
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
        )
    else:
        dd = tools.get_data_dir()
        process_metrics(
            bounds_in_path=str(dd / "datasets/boundaries.gpkg"),
            overture_data_dir=str(dd / "cities_data/overture"),
            blocks_path=str(dd / "datasets/blocks.gpkg"),
            trees_path=str(dd / "datasets/tree_canopies.gpkg"),
            hts_raster_data_dir=str(dd / "cities_data/heights"),
            stats_path=str(dd / "Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg"),
            processed_data_dir=str(dd / "cities_data/processed"),
            overwrite=False,
        )
