""" """

import argparse

import geopandas as gpd
from shapely import geometry

from src import tools

logger = tools.get_logger(__name__)
WORKING_CRS = 3035

# EU-27 + Norway, Liechtenstein, Switzerland (GADM names as used in GC_CNT_GAD_2025)
INCLUDED_COUNTRIES: set[str] = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
}


def extract_boundary_polys(ucdb_in_path: str, bounds_out_path: str) -> None:
    """ """
    tools.validate_filepath(ucdb_in_path)
    tools.validate_directory(bounds_out_path, create=True)
    # bounding box to filter out remote islands (e.g. Madeira, overseas territories)
    eu_bounds = [2500000, 1250000, 7000000, 5000000]  # W, S, E, N - EPSG:3035
    logger.info(f"Clipping polygons outside of hard-coded EU boundary: {eu_bounds} (WSEN / EPSG:3035)")
    eu_boundary = geometry.box(*eu_bounds)  # type: ignore
    # load GHS-UCDB vector boundaries
    logger.info(f"Loading GHS-UCDB boundaries from {ucdb_in_path}")
    ucdb_gdf = gpd.read_file(ucdb_in_path, layer="GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A")
    ucdb_gdf = ucdb_gdf.to_crs(WORKING_CRS)
    # filter to included countries within bounding box
    polys: list[geometry.Polygon | geometry.MultiPolygon] = []
    labels: list[str | None] = []
    countries: list[str | None] = []
    ucdb_ids: list[int] = []
    for _idx, row in ucdb_gdf.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue
        # skip countries not in the included set
        country = row.get("GC_CNT_GAD_2025")
        if country not in INCLUDED_COUNTRIES:
            continue
        # don't load if centroid outside bounding box (filters overseas territories / remote islands)
        if not eu_boundary.contains(poly.centroid):
            continue
        # buffer and reverse buffer to smooth edges
        poly = poly.buffer(2000).buffer(-1000)
        polys.append(poly)
        labels.append(row.get("GC_UCN_MAI_2025"))
        countries.append(row.get("GC_CNT_GAD_2025"))
        ucdb_ids.append(int(row["ID_UC_G0"]))
    logger.info(f"Filtered to {len(polys)} boundaries within EU extent")

    # generate the gdf - use ID_UC_G0 as immutable bounds_fid
    bounds_gdf = gpd.GeoDataFrame(  # type: ignore
        {"geom": polys, "label": labels, "country": countries, "bounds_fid": ucdb_ids},
        geometry="geom",
        crs=WORKING_CRS,
    )
    bounds_gdf = bounds_gdf.set_index("bounds_fid")
    bounds_gdf.to_file(bounds_out_path, driver="GPKG", layer="bounds")
    # save validation subset for France and Netherlands
    validation_countries = ["France", "Netherlands"]
    validation_gdf = bounds_gdf[bounds_gdf["country"].isin(validation_countries)]
    validation_path = bounds_out_path.replace(".gpkg", "_validation.gpkg")
    logger.info(f"Saving {len(validation_gdf)} validation boundaries (FR/NL) to {validation_path}")
    validation_gdf.to_file(validation_path, driver="GPKG", layer="bounds")


if __name__ == "__main__":
    # All paths default to T2E_DATA_DIR (from .env). No args needed:
    #   python -m src.data.generate_boundary_polys
    logger.info("Extracting boundary polygons from GHS-UCDB.")
    dd = tools.get_data_dir()
    parser = argparse.ArgumentParser(description="Extract boundary polygons from GHS-UCDB.")
    parser.add_argument(
        "ucdb_data_path",
        type=str,
        nargs="?",
        default=str(dd / "GHS_UCDB_REGION_EUROPE_R2024A_V1_1" / "GHS_UCDB_REGION_EUROPE_R2024A.gpkg"),
        help="Path to the GHS-UCDB GeoPackage.",
    )
    parser.add_argument(
        "bounds_output_path",
        type=str,
        nargs="?",
        default=str(dd / "datasets" / "boundaries.gpkg"),
        help="Path to the data output.",
    )
    args = parser.parse_args()
    extract_boundary_polys(args.ucdb_data_path, args.bounds_output_path)
