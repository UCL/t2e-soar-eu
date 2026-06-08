""" """

import argparse
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely import geometry
from tqdm import tqdm

from src import tools

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)

logger = tools.get_logger(__name__)


def load_urban_blocks(bounds_in_path: str, data_dir_path: str, blocks_out_path: str) -> None:
    """ """
    logger.info(f"Loading urban atlas blocks data from path: {data_dir_path}")
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_dir_path)
    tools.validate_directory(blocks_out_path, create=True)
    # load bounds
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_geom = bounds_gdf.union_all()
    # filter out unwanted block types
    filter_classes = [
        "Fast transit roads and associated land",
        "Other roads and associated land",
        "Railways and associated land",
    ]
    all_blocks = []
    # iter FGB files in per-city subdirectories
    dir_path: Path = Path(data_dir_path)
    fgb_files = sorted(dir_path.glob("**/*_LCU_*.fgb"))
    for fgb_path in tqdm(fgb_files):
        # use fiona for quick bbox check
        try:
            with fiona.open(fgb_path) as src:
                if not geometry.box(*src.bounds).intersects(bounds_geom):  # type: ignore
                    continue
        except Exception:
            continue
        # read only features within the bounds bbox when possible
        try:
            gdf = gpd.read_file(fgb_path, bbox=bounds_geom.bounds)
        except Exception:
            gdf = gpd.read_file(fgb_path)
        if gdf.empty:
            continue
        # discard rows if in filtered classes
        gdf = gdf.loc[~gdf["class_2021"].isin(filter_classes)]
        if gdf.empty:
            continue
        # filter spatially using envelope bbox for speed, then refine
        gdf["bbox"] = gdf["geometry"].envelope
        gdf_itx = gdf.set_geometry("bbox")
        gdf_itx = gdf_itx.loc[gdf_itx.intersects(bounds_geom)]
        if gdf_itx.empty:
            continue
        # rename geometry column and set it as active geometry
        gdf_itx = gdf_itx.rename(columns={"geometry": "geom", "Pop2021": "pop2021"})
        gdf_itx = gdf_itx.set_geometry("geom")
        # Clip large water/wetland polygons to the buffered boundary so
        # lagoons, rivers, and coastlines are trimmed to the relevant portion.
        # Only clip polygons > 0.5 km² to avoid expensive clipping on small blocks.
        clip_classes = {
            "Water",
            "Wetlands",
            "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)",
            "Arable land (annual crops)",
            "Forests",
            "Green urban areas (Public access)",
            "Green urban areas (Unknown access conditions)",
            "Herbaceous vegetation associations (natural grassland, moors...)",
            "Pastures",
            "Permanent crops (vineyards, fruit trees, olive groves)",
            "Orchards at the fringe of urban classes",
            "Complex and mixed cultivation patterns",
            "Port areas",
        }
        area_threshold = 500_000  # 0.5 km² in m²
        needs_clip = gdf_itx["class_2021"].isin(clip_classes) & (gdf_itx.geometry.area > area_threshold)
        if needs_clip.any():
            # Build a local bounds geometry from the UA file's bbox overlap
            # with bounds_geom for faster clipping
            local_bounds = bounds_geom.intersection(geometry.box(*gdf_itx.total_bounds))
            clipped = gdf_itx.loc[needs_clip].copy()
            clipped["geom"] = clipped.geometry.intersection(local_bounds)
            clipped = clipped[~clipped["geom"].is_empty]
            gdf_itx = pd.concat([gdf_itx.loc[~needs_clip], clipped], ignore_index=True)
            gdf_itx = gdf_itx.set_geometry("geom")
        if gdf_itx.empty:
            continue
        # explode multipolygons (clip can produce multipolygons)
        gdf_exp = gdf_itx.explode(index_parts=False)
        cols = [
            "country",
            "fua_name",
            "fua_code",
            "code_2021",
            "class_2021",
            "identifier",
            "comment",
            "pop2021",
            "geom",
        ]
        available_cols = [c for c in cols if c in gdf_exp.columns]
        if not available_cols:
            continue
        all_blocks.append(gdf_exp[available_cols])
    # save to file
    if all_blocks:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_blocks, ignore_index=True))
        final_gdf.to_file(blocks_out_path, driver="GPKG")


if __name__ == "__main__":
    # All paths default to T2E_DATA_DIR (from .env). No args needed:
    #   python -m src.data.load_urban_atlas_blocks
    dd = tools.get_data_dir()
    parser = argparse.ArgumentParser(description="Load Urban Atlas data.")
    parser.add_argument(
        "bounds_in_path",
        type=str,
        nargs="?",
        default=str(dd / "datasets" / "boundaries.gpkg"),
        help="Input data directory with boundary GPKG.",
    )
    parser.add_argument(
        "data_dir_path",
        type=str,
        nargs="?",
        default=str(dd / "UA_2021_3035_eu"),
        help="Input data directory with Urban Atlas FGB files.",
    )
    parser.add_argument(
        "blocks_out_path",
        type=str,
        nargs="?",
        default=str(dd / "datasets" / "blocks.gpkg"),
        help="Output path for urban blocks GPKG.",
    )
    args = parser.parse_args()
    load_urban_blocks(
        bounds_in_path=args.bounds_in_path,
        data_dir_path=args.data_dir_path,
        blocks_out_path=args.blocks_out_path,
    )
