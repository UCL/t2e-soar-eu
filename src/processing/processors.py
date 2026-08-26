""" """

import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import rasterio
from cityseer import decay
from cityseer.network import CityNetwork
from rasterio.mask import mask
from shapely import STRtree
from shapely.geometry import LineString, Point
from tqdm import tqdm

from src import landuse_categories, tools

logger = tools.get_logger(__name__)

OVERTURE_SCHEMA = tools.generate_overture_schema()  # type: ignore
DISTANCES_LU = [200, 400, 800, 1200, 1600]
DISTANCES_CENT = [400, 800, 1200, 1600, 4800, 9600]
DISTANCES_MORPH = [200, 400]
DISTANCES_GREEN_REACH = [1600]
DISTANCES_GREEN_AGG = [200, 400, 800]

# Decay variants reproducing the published _nw / _wt column pairs:
# flat = unweighted counts/stats, exponential (steepness 4) = the historical
# distance-weighted variant (weight = exp(-4 d / d_max)).
DECAYS = {"nw": decay.flat(), "wt": decay.exponential()}

_COL_ORDER_RE = re.compile(r"^(cc_.+)_(nw|wt)_(\d+)$")


def _restore_column_order(nodes_gdf: gpd.GeoDataFrame) -> None:
    """Rename cc_*_{label}_{dist} columns to the published cc_*_{dist}_{label} order.

    cityseer 5.x suffixes decay labels before the distance; the deposited
    dataset and the S1 schema use distance-then-label, so outputs are renamed
    in place immediately after each compute call.
    """
    renames = {}
    for col in nodes_gdf.columns:
        m = _COL_ORDER_RE.match(col)
        if m:
            renames[col] = f"{m.group(1)}_{m.group(3)}_{m.group(2)}"
    if renames:
        nodes_gdf.rename(columns=renames, inplace=True)


def process_centrality(cn: CityNetwork) -> CityNetwork:
    """ """
    logger.info("Computing centrality")
    # Expression dicts reproduce the full published centrality set
    cn = cn.centrality_shortest(
        distances=DISTANCES_CENT,
        closeness={
            "density": "1",
            "farness": "c",
            "harmonic": "1/c",
            "beta": "exp(-4 * p)",
        },
        betweenness={
            "betweenness": "1",
            "betweenness_beta": "exp(-4 * p)",
        },
        cycles=True,
        postprocess={"hillier": "density**2 / farness"},
    )
    return cn


def process_places(cn: CityNetwork, places_gdf: gpd.GeoDataFrame, infrast_gdf: gpd.GeoDataFrame) -> CityNetwork:
    """ """
    logger.info("Computing places")
    # apply standardized category merging
    places_gdf = landuse_categories.merge_landuse_categories(places_gdf)
    # landuses — fixed list so every city carries an identical column schema
    landuse_keys = list(landuse_categories.COMMON_LANDUSE_CATEGORIES)
    # compute accessibilities
    cn = cn.compute_accessibilities(
        places_gdf,  # type: ignore
        landuse_column_label="merged_cats",
        accessibility_keys=landuse_keys,
        distances=DISTANCES_LU,
        decay_fn=DECAYS,
    )
    cn = cn.compute_mixed_uses(
        places_gdf,
        landuse_column_label="merged_cats",
        distances=DISTANCES_LU,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    # infrastructure
    street_furn_keys = [
        "bench",
        "drinking_water",
        "fountain",
        "picnic_table",
        "plant",
        "planter",
        "post_box",
    ]
    parking_keys = [
        # "bicycle_parking",
        "motorcycle_parking",
        "parking",
    ]
    transport_keys = [
        "aerialway_station",
        "airport",
        "bus_station",
        "bus_stop",
        "ferry_terminal",
        "helipad",
        "international_airport",
        "railway_station",
        "regional_airport",
        "seaplane_airport",
        "subway_station",
    ]
    infrast_gdf["class"] = infrast_gdf["class"].replace(street_furn_keys, "street_furn")  # type: ignore
    infrast_gdf["class"] = infrast_gdf["class"].replace(parking_keys, "parking")  # type: ignore
    infrast_gdf["class"] = infrast_gdf["class"].replace(transport_keys, "transport")  # type: ignore
    landuse_keys = ["street_furn", "parking", "transport"]
    infrast_gdf = infrast_gdf[infrast_gdf["class"].isin(landuse_keys)]  # type: ignore
    # compute accessibilities
    cn = cn.compute_accessibilities(
        infrast_gdf,  # type: ignore
        landuse_column_label="class",
        accessibility_keys=landuse_keys,
        distances=DISTANCES_LU,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    return cn


def process_blocks_buildings(
    cn: CityNetwork,
    bldgs_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    hts_path: str | Path | None,
) -> tuple[CityNetwork, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ """
    logger.info("Computing morphology")
    # placeholders
    for col_key in [
        "area",
        "perimeter",
        "compactness",
        "orientation",
        "volume",
        "form_factor",
        "corners",
        "shape_index",
        "shared_walls",
        "shared_wall_ratio",
        "fractal_dimension",
        "mean_height",
        "floor_area",
    ]:
        bldgs_gdf[col_key] = np.nan
    if not bldgs_gdf.empty:
        # explode
        bldgs_gdf = bldgs_gdf.explode(index_parts=False)  # type: ignore
        bldgs_gdf.reset_index(drop=True, inplace=True)
        bldgs_gdf.index = bldgs_gdf.index.astype(str)
        bldgs_gdf["mean_height"] = np.nan
        # sample heights when a raster is available
        if hts_path is not None:
            hts_path = Path(hts_path)
            if hts_path.exists():
                with rasterio.open(hts_path) as rast_src:
                    logger.info("Sampling building heights")
                    heights = []
                    for _idx, bldg_row in tqdm(bldgs_gdf.iterrows(), total=len(bldgs_gdf)):
                        try:
                            # raster values within building polygon
                            out_image, _ = mask(
                                rast_src,
                                [bldg_row.geometry.buffer(10)],
                                all_touched=True,
                                crop=True,
                                nodata=rast_src.nodata,
                            )
                            # Filter out nodata values before computing mean
                            raster_data = out_image[0]
                            if rast_src.nodata is not None:
                                # Mask out nodata values
                                valid_data = raster_data[raster_data != rast_src.nodata]
                            else:
                                valid_data = raster_data
                            # Compute mean, excluding NaN values as well
                            mean_height = np.nanmean(valid_data) if len(valid_data) > 0 else np.nan
                            heights.append(mean_height)
                        except ValueError:
                            heights.append(np.nan)
                    bldgs_gdf["mean_height"] = heights
            else:
                logger.warning("Building heights raster not available at %s", hts_path)
        else:
            logger.warning("No building heights raster available; leaving height metrics empty")
        # bldg metrics
        area = bldgs_gdf.area
        ht = bldgs_gdf.loc[:, "mean_height"]
        bldgs_gdf["area"] = area
        bldgs_gdf["perimeter"] = bldgs_gdf.length
        bldgs_gdf["compactness"] = momepy.circular_compactness(bldgs_gdf)
        bldgs_gdf["orientation"] = momepy.orientation(bldgs_gdf)
        # height-based metrics
        bldgs_gdf["volume"] = momepy.volume(area, ht)
        bldgs_gdf["floor_area"] = momepy.floor_area(area, ht, 3)
        bldgs_gdf["form_factor"] = momepy.form_factor(bldgs_gdf, ht)
        # complexity metrics
        bldgs_gdf["corners"] = momepy.corners(bldgs_gdf)
        bldgs_gdf["shape_index"] = momepy.shape_index(bldgs_gdf)
        bldgs_gdf["shared_walls"] = momepy.shared_walls(bldgs_gdf, strict=False, tolerance=1.0)
        bldgs_gdf["fractal_dimension"] = momepy.fractal_dimension(bldgs_gdf)
        # shared wall ratio (fraction of perimeter that is shared)
        finite_per = np.isfinite(bldgs_gdf["perimeter"]) & (bldgs_gdf["perimeter"] > 0)
        bldgs_gdf["shared_wall_ratio"] = np.nan
        bldgs_gdf.loc[finite_per, "shared_wall_ratio"] = (
            bldgs_gdf.loc[finite_per, "shared_walls"] / bldgs_gdf.loc[finite_per, "perimeter"]
        )
    # calculate
    bldgs_gdf["centroid"] = bldgs_gdf.geometry.centroid
    bldgs_gdf.set_geometry("centroid", inplace=True)
    bldg_stats_cols = [
        "area",
        "mean_height",  # already computed prior
        "perimeter",
        "compactness",
        "orientation",
        "volume",
        "form_factor",
        "corners",
        "shape_index",
        "shared_wall_ratio",
        "fractal_dimension",
    ]
    cn = cn.compute_stats(
        data_gdf=bldgs_gdf,
        stats_column_labels=bldg_stats_cols,
        distances=DISTANCES_MORPH,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    # Keep median + MAD for all; also keep sum for area and volume
    keep_sum = {"area", "volume"}
    nodes_gdf = cn.nodes_gdf
    for bldg_stats_col in bldg_stats_cols:
        trim_columns = []
        for column_name in nodes_gdf.columns:
            if column_name.startswith(f"cc_{bldg_stats_col}"):
                keep = (
                    column_name.startswith(f"cc_{bldg_stats_col}_median")
                    or column_name.startswith(f"cc_{bldg_stats_col}_mad")
                    or (bldg_stats_col in keep_sum and column_name.startswith(f"cc_{bldg_stats_col}_sum"))
                )
                if not keep:
                    trim_columns.append(column_name)
        nodes_gdf.drop(columns=trim_columns, inplace=True)
    bldgs_gdf["type"] = "building"  # for downstream use
    cn = cn.compute_accessibilities(
        bldgs_gdf,  # type: ignore
        landuse_column_label="type",
        accessibility_keys=["building"],
        distances=DISTANCES_MORPH,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    cn.nodes_gdf.drop(columns=[f"cc_building_nearest_max_{max(DISTANCES_MORPH)}"], inplace=True)
    cn, blocks_gdf = process_blocks(cn, bldgs_gdf, blocks_gdf)
    # reset geometry
    bldgs_gdf.set_geometry("geometry", inplace=True)
    bldgs_gdf.drop(columns=["centroid"], inplace=True)

    return cn, bldgs_gdf, blocks_gdf


def process_blocks(
    cn: CityNetwork,
    bldgs_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
) -> tuple[CityNetwork, gpd.GeoDataFrame]:
    """Compute block metrics and their street-level aggregations.

    ``bldgs_gdf`` must carry ``area``, ``floor_area``, and ``mean_height``
    columns and have its centroid as the active geometry: buildings are
    assigned to blocks by centroid intersection.
    """
    # placeholders
    for col_key in [
        "block_area",
        "block_perimeter",
        "block_compactness",
        "block_orientation",
        "block_covered_ratio",
        "block_far",
        "block_osr",
        "block_l",
        "block_mean_height",
    ]:
        blocks_gdf[col_key] = np.nan
    # block metrics — filter out oversized non-urban blocks (water, pasture, etc.)
    MAX_BLOCK_AREA = 2_000_000  # 2 km²
    if not blocks_gdf.empty:
        blocks_gdf = blocks_gdf[blocks_gdf.area <= MAX_BLOCK_AREA].copy()
        blocks_gdf.index = blocks_gdf.index.astype(str)
        blocks_gdf["block_area"] = blocks_gdf.area
        blocks_gdf["block_perimeter"] = blocks_gdf.length
        blocks_gdf["block_compactness"] = momepy.circular_compactness(blocks_gdf)
        blocks_gdf["block_orientation"] = momepy.orientation(blocks_gdf)
    # joint metrics require spatial join
    if not blocks_gdf.empty and not bldgs_gdf.empty:
        blocks_gdf["uID"] = blocks_gdf.index.values
        merged_gdf = gpd.sjoin(
            bldgs_gdf,
            blocks_gdf,
            how="left",
            predicate="intersects",
            lsuffix="bldg",
            rsuffix="block",
        )
        # Calculate covered ratio (GSI): sum of building areas per block / block area
        building_area_per_block = merged_gdf.groupby("uID")["area"].sum()
        blocks_gdf["block_covered_ratio"] = building_area_per_block / blocks_gdf["block_area"]
        blocks_gdf["block_covered_ratio"] = blocks_gdf["block_covered_ratio"].fillna(0).clip(upper=1.0)
        # Blocks below 1% coverage are too sparse for meaningful Spacematrix metrics
        MIN_COVERAGE = 0.01
        sparse_mask = blocks_gdf["block_covered_ratio"] < MIN_COVERAGE
        # Calculate FAR (FSI): sum of building floor areas per block / block area
        # min_count=1 ensures blocks where all buildings lack height data get NaN (not 0)
        building_floor_area_per_block = merged_gdf.groupby("uID")["floor_area"].sum(min_count=1)
        blocks_gdf["block_far"] = building_floor_area_per_block / blocks_gdf["block_area"]
        # Spacematrix derived: OSR = (1 - GSI) / FSI, L = FSI / GSI
        gsi = blocks_gdf["block_covered_ratio"]
        fsi = blocks_gdf["block_far"]
        blocks_gdf["block_osr"] = np.where(fsi > 0, (1 - gsi) / fsi, np.nan)
        blocks_gdf["block_l"] = np.where(gsi > 0, fsi / gsi, np.nan)
        # Block mean height: area-weighted mean building height
        # denominator restricted to height-valid buildings so partial raster coverage doesn't bias low
        merged_gdf["_weighted_ht"] = merged_gdf["mean_height"] * merged_gdf["area"]
        merged_gdf["_ht_area"] = merged_gdf["area"].where(merged_gdf["mean_height"].notna())
        weighted_ht_sum = merged_gdf.groupby("uID")["_weighted_ht"].sum(min_count=1)
        ht_valid_area_per_block = merged_gdf.groupby("uID")["_ht_area"].sum(min_count=1)
        blocks_gdf["block_mean_height"] = weighted_ht_sum / ht_valid_area_per_block
        # NaN out all Spacematrix metrics for sparse blocks
        spacematrix_cols = ["block_covered_ratio", "block_far", "block_osr", "block_l", "block_mean_height"]
        blocks_gdf.loc[sparse_mask, spacematrix_cols] = np.nan
    # block stats
    blocks_gdf["centroid"] = blocks_gdf.geometry.centroid
    blocks_gdf.set_geometry("centroid", inplace=True)
    block_stats_cols = [
        "block_area",
        "block_perimeter",
        "block_compactness",
        "block_orientation",
        "block_covered_ratio",
        "block_far",
        "block_osr",
        "block_l",
        "block_mean_height",
    ]
    cn = cn.compute_stats(
        data_gdf=blocks_gdf,
        stats_column_labels=block_stats_cols,
        distances=DISTANCES_MORPH,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    # Keep median + MAD for all; also keep sum for block_area
    block_keep_sum = {"block_area"}
    nodes_gdf = cn.nodes_gdf
    for block_stats_col in block_stats_cols:
        trim_columns = []
        for column_name in nodes_gdf.columns:
            if column_name.startswith(f"cc_{block_stats_col}"):
                keep = (
                    column_name.startswith(f"cc_{block_stats_col}_median")
                    or column_name.startswith(f"cc_{block_stats_col}_mad")
                    or (block_stats_col in block_keep_sum and column_name.startswith(f"cc_{block_stats_col}_sum"))
                )
                if not keep:
                    trim_columns.append(column_name)
        nodes_gdf.drop(columns=trim_columns, inplace=True)
    blocks_gdf["type"] = "block"  # for downstream use
    cn = cn.compute_accessibilities(
        blocks_gdf,  # type: ignore
        landuse_column_label="type",
        accessibility_keys=["block"],
        distances=DISTANCES_MORPH,
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    cn.nodes_gdf.drop(columns=[f"cc_block_nearest_max_{max(DISTANCES_MORPH)}"], inplace=True)
    # reset geometry
    blocks_gdf.set_geometry("geometry", inplace=True)
    blocks_gdf.drop(columns=["centroid"], inplace=True)

    return cn, blocks_gdf


def _extract_green_points(green_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract boundary points from exploded green / trees polygons."""

    # function for extracting points
    def generate_points(fid, categ, polygon, area, interval=20, simplify=20):
        if polygon.is_empty or polygon.exterior.length == 0:
            return []
        ring = polygon.exterior.simplify(simplify)
        num_points = int(ring.length // interval)
        return [
            (fid, categ, area, ring.interpolate(distance)) for distance in range(0, num_points * interval, interval)
        ]

    # extract points
    # fids are namespaced per category: cityseer deduplicates by data_id globally,
    # so a shared RangeIndex would let a nearer tree point mask a green polygon
    points = []
    # for green
    for fid, geom in zip(green_gdf.index, green_gdf.geometry, strict=True):  # type: ignore
        if geom.geom_type == "Polygon":
            points.extend(generate_points(f"green_{fid}", "green", geom, geom.area, interval=20, simplify=10))
    # for trees
    for fid, geom in zip(trees_gdf.index, trees_gdf.geometry, strict=True):  # type: ignore
        if geom.geom_type == "Polygon":
            points.extend(generate_points(f"trees_{fid}", "trees", geom, geom.area, interval=20, simplify=5))
    # create GDF
    points_gdf = gpd.GeoDataFrame(  # type: ignore
        points,
        columns=["fid", "cat", "area", "geometry"],
        geometry="geometry",
        crs=trees_gdf.crs,  # type: ignore
    )
    points_gdf.index = points_gdf.index.astype(str)
    return points_gdf


def process_green(cn: CityNetwork, green_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame) -> CityNetwork:
    """ """
    # Intentionally using points for handling extra large features like rivers
    logger.info("Computing green")
    # check Polygons
    green_gdf = green_gdf.explode(index_parts=False)  # type: ignore
    green_gdf.reset_index(drop=True, inplace=True)
    # check Polygons
    trees_gdf = trees_gdf.explode(index_parts=False)  # type: ignore
    trees_gdf.reset_index(drop=True, inplace=True)
    points_gdf = _extract_green_points(green_gdf, trees_gdf)
    # relabel area to green_area and trees_area
    green_idx = points_gdf["cat"] == "green"
    trees_idx = points_gdf["cat"] == "trees"
    points_gdf["green_area"] = np.where(green_idx, points_gdf["area"], 0.0)
    points_gdf["trees_area"] = np.where(trees_idx, points_gdf["area"], 0.0)
    points_gdf = points_gdf.drop(columns=["area"])
    # compute accessibilities
    cn = cn.compute_accessibilities(
        points_gdf,  # type: ignore
        landuse_column_label="cat",
        accessibility_keys=["green", "trees"],
        distances=DISTANCES_GREEN_REACH,
        data_id_col="fid",  # deduplicate
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    # drop - aggregation columns since these are not meaningful for interpolated aggs - only using distances
    nodes_gdf = cn.nodes_gdf
    nodes_gdf.drop(
        columns=[
            "cc_green_1600_nw",
            "cc_green_1600_wt",
            "cc_trees_1600_nw",
            "cc_trees_1600_wt",
        ],
        inplace=True,
    )
    # set contained green nodes to zero
    contained_green_idx = gpd.sjoin(nodes_gdf, green_gdf, predicate="intersects", how="inner")
    nodes_gdf.loc[contained_green_idx.index, "cc_green_nearest_max_1600"] = 0
    # same for trees
    contained_trees_idx = gpd.sjoin(nodes_gdf, trees_gdf, predicate="intersects", how="inner")
    nodes_gdf.loc[contained_trees_idx.index, "cc_trees_nearest_max_1600"] = 0
    # sum areas within buffer distances
    points_gdf["green_area"] = points_gdf["green_area"] / (1000**2)  # m2 to km2
    points_gdf["trees_area"] = points_gdf["trees_area"] / (1000**2)  # m2 to km2
    cn = cn.compute_stats(
        data_gdf=points_gdf,
        stats_column_labels=["green_area", "trees_area"],
        distances=DISTANCES_GREEN_AGG,
        data_id_col="fid",  # deduplicate
        decay_fn=DECAYS,
    )
    _restore_column_order(cn.nodes_gdf)
    # drop unnecessary columns
    for area_col in ["green_area", "trees_area"]:
        trim_columns = []
        for column_name in cn.nodes_gdf.columns:
            if column_name.startswith(f"cc_{area_col}") and not column_name.startswith(f"cc_{area_col}_sum"):
                trim_columns.append(column_name)
        cn.nodes_gdf.drop(columns=trim_columns, inplace=True)
    return cn


def _merge_intervals(intervals: list[tuple[float, float]]) -> float:
    """Merge overlapping (start, end) intervals and return total covered length."""
    if not intervals:
        return 0.0
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def compute_street_frontage(
    streets_gdf: gpd.GeoDataFrame,
    bldgs_gdf: gpd.GeoDataFrame,
    buffer_dist: float = 35.0,
    band_tol: float = 5.0,
) -> pd.DataFrame:
    """Compute bilateral street-frontage for each street segment.

    For each street LineString:
    1. Buffer by ``buffer_dist`` to define a corridor.
    2. Extract individual building edges; keep only those whose midpoint
       is inside the corridor AND whose nearest street is this one.
    3. Distance-band filter: per side, keep edges within ``band_tol``
       metres of the closest edge (excludes rear extensions/garages).
    4. Classify edges as left/right; trim 3m from ends; compute coverage.

    Parameters
    ----------
    streets_gdf : GeoDataFrame
        Street segments as LineString geometries.
    bldgs_gdf : GeoDataFrame
        Building footprints as Polygon geometries.
    buffer_dist : float
        Buffer distance (metres) around each street centreline (default 35).
    band_tol : float
        Metres beyond the closest facade edge to include (default 5).

    Returns
    -------
    DataFrame
        Columns: ``frontage_max``, ``frontage_avg``, ``frontage_left``,
        ``frontage_right`` — all (0–1), indexed to match ``streets_gdf``.
    """
    empty_result = pd.DataFrame(
        {
            "frontage_max": np.nan,
            "frontage_avg": np.nan,
            "frontage_left": np.nan,
            "frontage_right": np.nan,
            "frontage_edges_left": 0,
            "frontage_edges_right": 0,
        },
        index=streets_gdf.index,
    )
    if bldgs_gdf.empty or streets_gdf.empty:
        return empty_result

    # ── Extract building edges as arrays ─────────────────────────────
    # Pre-compute midpoints and directions to avoid per-edge object creation.
    edge_x0 = []
    edge_y0 = []
    edge_x1 = []
    edge_y1 = []
    for bldg in bldgs_gdf.geometry.values:
        rings = []
        try:
            rings.append(list(bldg.exterior.coords))
        except AttributeError:
            for part in bldg.geoms:
                if hasattr(part, "exterior"):
                    rings.append(list(part.exterior.coords))
        for ring in rings:
            for i in range(len(ring) - 1):
                edge_x0.append(ring[i][0])
                edge_y0.append(ring[i][1])
                edge_x1.append(ring[i + 1][0])
                edge_y1.append(ring[i + 1][1])

    if not edge_x0:
        return empty_result.fillna(0.0)

    ex0 = np.array(edge_x0)
    ey0 = np.array(edge_y0)
    ex1 = np.array(edge_x1)
    ey1 = np.array(edge_y1)
    emx = (ex0 + ex1) * 0.5
    emy = (ey0 + ey1) * 0.5
    n_edges = len(emx)

    # Build edge LineStrings for spatial query (unavoidable for STRtree)
    edge_geoms = np.array(
        [LineString([(ex0[i], ey0[i]), (ex1[i], ey1[i])]) for i in range(n_edges)],
        dtype=object,
    )
    edge_tree = STRtree(edge_geoms)

    # ── Batch nearest-street lookup ──────────────────────────────────
    street_geoms = streets_gdf.geometry.values
    street_tree = STRtree(street_geoms)
    mid_pts = np.array([Point(emx[i], emy[i]) for i in range(n_edges)], dtype=object)
    edge_nearest_street = street_tree.query_nearest(
        mid_pts,
        return_distance=False,
    )
    # edge_nearest_street shape: (2, n_matches) — [input_idx, tree_idx]
    # For query_nearest with single nearest, it returns one match per input.
    nearest_street_for_edge = np.full(n_edges, -1, dtype=int)
    nearest_street_for_edge[edge_nearest_street[0]] = edge_nearest_street[1]

    # ── Spatial query: which edges fall in each street's buffer ──────
    street_buffers = streets_gdf.geometry.buffer(buffer_dist)
    left_idx, right_idx = edge_tree.query(street_buffers.values, predicate="intersects")
    street_to_edges: dict[int, list[int]] = defaultdict(list)
    for s_pos, e_pos in zip(left_idx, right_idx):
        # Fast pre-filter: skip edges whose nearest street is not this one
        if nearest_street_for_edge[e_pos] != s_pos:
            continue
        street_to_edges[s_pos].append(e_pos)

    # ── Per-street processing ────────────────────────────────────────
    n_streets = len(streets_gdf)
    ratios_max = np.full(n_streets, np.nan, dtype=float)
    ratios_avg = np.full(n_streets, np.nan, dtype=float)
    ratios_left = np.full(n_streets, np.nan, dtype=float)
    ratios_right = np.full(n_streets, np.nan, dtype=float)
    edges_left = np.zeros(n_streets, dtype=int)
    edges_right = np.zeros(n_streets, dtype=int)

    for s_pos in range(len(streets_gdf)):
        line = street_geoms[s_pos]
        length = line.length
        if length < 1.0:
            continue

        edge_positions = street_to_edges.get(s_pos)
        if not edge_positions:
            ratios_max[s_pos] = 0.0
            ratios_avg[s_pos] = 0.0
            ratios_left[s_pos] = 0.0
            ratios_right[s_pos] = 0.0
            continue

        candidates_left = []
        candidates_right = []

        for e_pos in edge_positions:
            # Project edge endpoints onto street
            pr1 = line.project(Point(ex0[e_pos], ey0[e_pos]))
            pr2 = line.project(Point(ex1[e_pos], ey1[e_pos]))
            e_min, e_max = min(pr1, pr2), max(pr1, pr2)
            if e_max <= e_min:
                continue

            # Point on street nearest to edge midpoint
            mid_proj = (pr1 + pr2) * 0.5
            pt_on_line = line.interpolate(mid_proj)
            perp_dist = ((emx[e_pos] - pt_on_line.x) ** 2 + (emy[e_pos] - pt_on_line.y) ** 2) ** 0.5

            # Local street tangent for left/right classification
            t_frac = mid_proj / length
            tp1 = line.interpolate(max(0, t_frac - 0.01), normalized=True)
            tp2 = line.interpolate(min(1, t_frac + 0.01), normalized=True)
            cross = (tp2.x - tp1.x) * (emy[e_pos] - pt_on_line.y) - (tp2.y - tp1.y) * (emx[e_pos] - pt_on_line.x)
            if cross >= 0:
                candidates_left.append((e_min, e_max, perp_dist))
            else:
                candidates_right.append((e_min, e_max, perp_dist))

        # Distance-band filter per side
        raw_left = _filter_facade_band(candidates_left, band_tol)
        raw_right = _filter_facade_band(candidates_right, band_tol)
        edges_left[s_pos] = len(raw_left)
        edges_right[s_pos] = len(raw_right)

        if not raw_left and not raw_right:
            ratios_max[s_pos] = 0.0
            ratios_avg[s_pos] = 0.0
            ratios_left[s_pos] = 0.0
            ratios_right[s_pos] = 0.0
            continue

        # Trim 3m from each end to reduce junction contamination
        trim = min(3.0, length / 3)
        effective_length = length - 2 * trim
        if effective_length <= 0:
            ratios_max[s_pos] = 0.0
            ratios_avg[s_pos] = 0.0
            ratios_left[s_pos] = 0.0
            ratios_right[s_pos] = 0.0
            continue

        iv_left = [
            (max(trim, a) - trim, min(length - trim, b) - trim)
            for a, b in raw_left
            if min(length - trim, b) > max(trim, a)
        ]
        iv_right = [
            (max(trim, a) - trim, min(length - trim, b) - trim)
            for a, b in raw_right
            if min(length - trim, b) > max(trim, a)
        ]

        frac_left = min(_merge_intervals(iv_left) / effective_length, 1.0)
        frac_right = min(_merge_intervals(iv_right) / effective_length, 1.0)
        ratios_left[s_pos] = frac_left
        ratios_right[s_pos] = frac_right
        ratios_max[s_pos] = max(frac_left, frac_right)
        ratios_avg[s_pos] = (frac_left + frac_right) / 2

    return pd.DataFrame(
        {
            "frontage_max": ratios_max,
            "frontage_avg": ratios_avg,
            "frontage_left": ratios_left,
            "frontage_right": ratios_right,
            "frontage_edges_left": edges_left,
            "frontage_edges_right": edges_right,
        },
        index=streets_gdf.index,
    )


def _filter_facade_band(candidates, band_tol):
    """Keep edges within ``band_tol`` metres of the closest facade edge."""
    if not candidates:
        return []
    nearest = min(d for _, _, d in candidates)
    return [(a, b) for a, b, d in candidates if d <= nearest + band_tol]
