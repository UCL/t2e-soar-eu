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


def load_tree_canopies(bounds_in_path: str, data_dir_path: str, trees_out_path: str) -> None:
    """ """
    logger.info(f"Loading urban atlas trees data from path: {data_dir_path}")
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_dir_path)
    tools.validate_directory(trees_out_path, create=True)
    # load bounds
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_geom = bounds_gdf.union_all()
    # gather canopies
    all_canopies = []
    # iter FGB files in per-city subdirectories (skip urban mask _UM_ files)
    dir_path: Path = Path(data_dir_path)
    fgb_files = sorted(dir_path.glob("**/*_STL_*.fgb"))
    for fgb_path in tqdm(fgb_files):
        # use fiona for quick bbox check
        try:
            with fiona.open(fgb_path) as src:  # type: ignore
                if not geometry.box(*src.bounds).intersects(bounds_geom):  # type: ignore
                    continue
        except Exception:
            continue
        # read only features within the bounds bbox when possible
        try:
            gdf = gpd.read_file(fgb_path, bbox=bounds_geom.bounds)  # type: ignore
        except Exception:
            gdf = gpd.read_file(fgb_path)  # type: ignore
        if gdf.empty:
            continue
        # filter spatially using bbox envelope for speed
        gdf = gdf.loc[gdf.geometry.notna()].copy()
        gdf["bbox"] = gdf["geometry"].envelope
        gdf_itx = gdf.set_geometry("bbox")
        gdf_itx = gdf_itx.loc[gdf_itx.intersects(bounds_geom)].copy()
        if gdf_itx.empty:
            continue
        # rename geometry column and set it as active geometry
        gdf_itx = gdf_itx.rename(columns={"geometry": "geom"})
        gdf_itx = gdf_itx.set_geometry("geom")
        # explode multipolygons
        gdf_exp = gdf_itx.explode(index_parts=False)
        # select available columns
        cols = [c for c in ["fua_name", "fua_code", "geom"] if c in gdf_exp.columns]
        if not cols:
            continue
        all_canopies.append(gdf_exp[cols])
    # save to file
    if all_canopies:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_canopies, ignore_index=True))
        final_gdf.to_file(trees_out_path, driver="GPKG")


if __name__ == "__main__":
    # All paths default to T2E_DATA_DIR (from .env). No args needed:
    #   python -m src.data.load_urban_atlas_trees
    dd = tools.get_data_dir()
    parser = argparse.ArgumentParser(description="Load tree canopy data.")
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
        default=str(dd / "STL_2021_3035_eu"),
        help="Input data directory with Urban Atlas STL FGB files.",
    )
    parser.add_argument(
        "trees_out_path",
        type=str,
        nargs="?",
        default=str(dd / "datasets" / "tree_canopies.gpkg"),
        help="Output path for urban trees GPKG.",
    )
    args = parser.parse_args()
    load_tree_canopies(
        bounds_in_path=args.bounds_in_path,
        data_dir_path=args.data_dir_path,
        trees_out_path=args.trees_out_path,
    )
