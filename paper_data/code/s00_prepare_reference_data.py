#!/usr/bin/env python3
"""Prepare SIRENE (France) and BAG (Netherlands) registry extracts.

Processes official government datasets for comparison with Overture Maps POI data:

1. Loads country boundaries from boundaries_validation.gpkg
2. Loads SIRENE Parquet and filters to active establishments with valid coordinates
3. Loads BAG data and filters to relevant usage purposes
4. Converts all data to EPSG:3035 (ETRS89 / LAEA Europe) CRS
5. Spatially filters to city boundaries + 2 km buffer
6. Saves to GeoPackage format in validation directory

Usage:
    python paper_data/code/s00_prepare_reference_data.py
"""

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import STRtree


def _bbox_filter_lambert93(df: pd.DataFrame, bbox_2154: tuple[float, float, float, float]) -> pd.DataFrame:
    """Fast bbox pre-filter on raw Lambert 93 coordinates (no geometry needed)."""
    xmin, ymin, xmax, ymax = bbox_2154
    x = df["coordonneeLambertAbscisseEtablissement"]
    y = df["coordonneeLambertOrdonneeEtablissement"]
    return df[(x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)]


def _spatial_filter_strtree(gdf: gpd.GeoDataFrame, boundary_geom) -> gpd.GeoDataFrame:
    """Filter points using STRtree for memory-efficient spatial lookup."""
    tree = STRtree(gdf.geometry.values)
    hits = tree.query(boundary_geom, predicate="intersects")
    return gdf.iloc[hits].copy()


def run_prepare_reference_data(
    bounds_path: Path,
    validation_dir: Path,
    force_reload: bool = False,
    verbose: bool = True,
) -> dict:
    """Prepare SIRENE and BAG reference datasets for validation.

    Parameters
    ----------
    bounds_path
        Path to boundaries.gpkg containing city boundaries (EPSG:3035)
    validation_dir
        Directory containing source data and for output files:
        - Input: StockEtablissement_utf8.parquet, lvbag-extract-nl/*.zip
        - Output: sirene_france.gpkg, bag_netherlands.gpkg
    force_reload
        If True, reprocess even if output files exist
    verbose
        If True, print progress messages

    Returns
    -------
    dict with keys:
        - sirene_path: Path to SIRENE output file (or None if not processed)
        - bag_path: Path to BAG output file (or None if not processed)
        - sirene_count: Number of SIRENE establishments
        - bag_count: Number of BAG buildings
    """
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    sirene_output = validation_dir / "sirene_france.gpkg"
    bag_output = validation_dir / "bag_netherlands.gpkg"

    result = {
        "sirene_path": None,
        "bag_path": None,
        "sirene_count": 0,
        "bag_count": 0,
    }

    if verbose:
        print("=" * 80)
        print("DATA QUALITY COMPARISON - DATA PREPARATION")
        print("=" * 80)

    # Load country boundaries
    if verbose:
        print(f"\nLoading country boundaries from: {bounds_path}")
    boundaries_gdf = gpd.read_file(bounds_path)
    if boundaries_gdf.crs and boundaries_gdf.crs.to_epsg() != 3035:
        boundaries_gdf = boundaries_gdf.to_crs("EPSG:3035")
    if verbose:
        print(f"  Loaded {len(boundaries_gdf)} boundary polygons")
        print(f"  Countries available: {sorted(boundaries_gdf['country'].dropna().unique())}")

    # ========================================================================
    # PART 1: FRANCE SIRENE
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("PART 1: FRANCE SIRENE")
        print("=" * 80)

    if sirene_output.exists() and not force_reload:
        if verbose:
            print(f"\n✓ SIRENE data already prepared: {sirene_output}")
            print(f"  File size: {sirene_output.stat().st_size / 1e9:.2f} GB")
            print("  (Set force_reload=True to reprocess)")
        result["sirene_path"] = sirene_output
        result["sirene_count"] = len(gpd.read_file(sirene_output))
    else:
        sirene_parquet_path = validation_dir / "StockEtablissement_utf8.parquet"

        if not sirene_parquet_path.exists():
            if verbose:
                print(f"\n⚠ SIRENE source file not found: {sirene_parquet_path}")
        else:
            if force_reload and sirene_output.exists() and verbose:
                print("\n⚠ force_reload=True, reprocessing SIRENE data...")

            # Build the buffered clip boundary in EPSG:3035
            france_bounds_3035 = boundaries_gdf[boundaries_gdf["country"] == "France"]
            france_clip = france_bounds_3035.geometry.union_all().buffer(2000)

            # Convert clip boundary to Lambert 93 bbox for fast pre-filter
            france_clip_2154 = gpd.GeoSeries([france_clip], crs="EPSG:3035").to_crs("EPSG:2154").iloc[0]
            bbox_2154 = france_clip_2154.bounds  # (minx, miny, maxx, maxy)

            if verbose:
                print(f"\nLoading SIRENE data from Parquet: {sirene_parquet_path}")

            sirene_df = pd.read_parquet(
                sirene_parquet_path,
                columns=[
                    "siret",
                    "activitePrincipaleEtablissement",
                    "etatAdministratifEtablissement",
                    "coordonneeLambertAbscisseEtablissement",
                    "coordonneeLambertOrdonneeEtablissement",
                ],
            )
            if verbose:
                print(f"  Loaded {len(sirene_df):,} establishments")

            # Filter to active establishments
            sirene_df = sirene_df[sirene_df["etatAdministratifEtablissement"] == "A"]
            sirene_df = sirene_df.drop(columns=["etatAdministratifEtablissement"])
            if verbose:
                print(f"  Active establishments: {len(sirene_df):,}")

            # Clean coordinates
            sirene_df = sirene_df.dropna(
                subset=["coordonneeLambertAbscisseEtablissement", "coordonneeLambertOrdonneeEtablissement"]
            )
            sirene_df = sirene_df[
                (sirene_df["coordonneeLambertAbscisseEtablissement"] != "[ND]")
                & (sirene_df["coordonneeLambertOrdonneeEtablissement"] != "[ND]")
            ]
            sirene_df["coordonneeLambertAbscisseEtablissement"] = pd.to_numeric(
                sirene_df["coordonneeLambertAbscisseEtablissement"], errors="coerce"
            )
            sirene_df["coordonneeLambertOrdonneeEtablissement"] = pd.to_numeric(
                sirene_df["coordonneeLambertOrdonneeEtablissement"], errors="coerce"
            )
            sirene_df = sirene_df.dropna(
                subset=["coordonneeLambertAbscisseEtablissement", "coordonneeLambertOrdonneeEtablissement"]
            )
            if verbose:
                print(f"  With valid coordinates: {len(sirene_df):,}")

            # Fast bbox pre-filter in Lambert 93 (no geometry objects yet)
            sirene_df = _bbox_filter_lambert93(sirene_df, bbox_2154)
            if verbose:
                print(f"  After bbox pre-filter: {len(sirene_df):,}")

            # Now build geometry only for the bbox-filtered subset
            sirene_df = sirene_df.rename(columns={"activitePrincipaleEtablissement": "APE_code"})
            sirene_gdf = gpd.GeoDataFrame(
                sirene_df[["siret", "APE_code"]],
                geometry=gpd.points_from_xy(
                    sirene_df["coordonneeLambertAbscisseEtablissement"],
                    sirene_df["coordonneeLambertOrdonneeEtablissement"],
                ),
                crs="EPSG:2154",
            )
            del sirene_df  # free the raw DataFrame

            if verbose:
                print("  Converting Lambert 93 to EPSG:3035...")
            sirene_gdf = sirene_gdf.to_crs("EPSG:3035")

            # Precise spatial filter using STRtree
            if verbose:
                print("  Spatial filtering to French city boundaries (+ 2 km buffer)...")
            sirene_gdf = _spatial_filter_strtree(sirene_gdf, france_clip)
            if verbose:
                print(f"  Establishments within boundaries: {len(sirene_gdf):,}")

            # Save
            if verbose:
                print(f"\n  Saving to: {sirene_output}")
            sirene_gdf.to_file(sirene_output, driver="GPKG", layer="establishments")

            result["sirene_path"] = sirene_output
            result["sirene_count"] = len(sirene_gdf)

            if verbose:
                print(f"  ✓ SIRENE data prepared: {len(sirene_gdf):,} establishments")
                print(f"  File size: {sirene_output.stat().st_size / 1e9:.2f} GB")

            del sirene_gdf  # free before BAG processing

    # ========================================================================
    # PART 2: NETHERLANDS BAG
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("PART 2: NETHERLANDS BAG")
        print("=" * 80)

    if bag_output.exists() and not force_reload:
        if verbose:
            print(f"\n✓ BAG data already prepared: {bag_output}")
            print(f"  File size: {bag_output.stat().st_size / 1e9:.2f} GB")
            print("  (Set force_reload=True to reprocess)")
        result["bag_path"] = bag_output
        result["bag_count"] = len(gpd.read_file(bag_output))
    else:
        bag_zip_dir = validation_dir / "lvbag-extract-nl"
        # BAG extract filename is 9999VBO<DDMMYYYY>.zip — date varies by extract,
        # so glob for whichever VBO archive is present.
        vbo_matches = sorted(bag_zip_dir.glob("9999VBO*.zip"))
        bag_vbo_zip = vbo_matches[-1] if vbo_matches else bag_zip_dir / "9999VBO*.zip"

        if not vbo_matches:
            if verbose:
                print(f"\n⚠ BAG source file not found: no 9999VBO*.zip in {bag_zip_dir}")
        else:
            if force_reload and bag_output.exists() and verbose:
                print("\n⚠ force_reload=True, reprocessing BAG data...")

            # Build the buffered clip boundary for NL
            nl_bounds_3035 = boundaries_gdf[boundaries_gdf["country"] == "Netherlands"]
            nl_clip = nl_bounds_3035.geometry.union_all().buffer(2000)

            active_statuses = {
                "Verblijfsobject in gebruik",
                "Verblijfsobject in gebruik (niet ingemeten)",
                "Verbouwing verblijfsobject",
            }

            if verbose:
                print(f"\nLoading BAG data from ZIP: {bag_vbo_zip}")

            # Process each XML file individually to avoid holding all in memory.
            # Filter and keep only the columns we need per chunk.
            kept_chunks: list[pd.DataFrame] = []
            n_raw = 0
            with zipfile.ZipFile(bag_vbo_zip, "r") as zip_ref:
                vbo_files = [f for f in zip_ref.namelist() if f.endswith((".xml", ".gml"))]
                if verbose:
                    print(f"  Found {len(vbo_files)} XML files")

                for i, vbo_file in enumerate(vbo_files, 1):
                    temp_vbo = validation_dir / f"temp_{vbo_file}"

                    with zip_ref.open(vbo_file) as source, open(temp_vbo, "wb") as target:
                        target.write(source.read())

                    chunk = gpd.read_file(temp_vbo)
                    temp_vbo.unlink()
                    n_raw += len(chunk)

                    # Filter immediately: active status + has usage purpose
                    chunk = chunk[chunk["gebruiksdoel"].notna() & chunk["status"].isin(active_statuses)]
                    if not chunk.empty:
                        kept_chunks.append(chunk[["identificatie", "gebruiksdoel", "geometry"]])

                    if verbose and i % 500 == 0:
                        n_kept = sum(len(c) for c in kept_chunks)
                        print(f"    ...processed {i}/{len(vbo_files)} files ({n_kept:,} kept so far)")

            if verbose:
                n_kept = sum(len(c) for c in kept_chunks)
                print(f"  Loaded {n_raw:,} records total, kept {n_kept:,} active with usage")

            bag_gdf = gpd.GeoDataFrame(pd.concat(kept_chunks, ignore_index=True))
            del kept_chunks

            # Ensure points
            if not bag_gdf.empty and bag_gdf.geometry.geom_type.iloc[0] != "Point":
                bag_gdf["geometry"] = bag_gdf.geometry.representative_point()
                bag_gdf = bag_gdf.set_geometry("geometry")

            # Convert to EPSG:3035
            if verbose:
                print(f"  Converting from {bag_gdf.crs} to EPSG:3035...")
            bag_gdf = bag_gdf.to_crs("EPSG:3035")

            # Spatial filter using STRtree
            if verbose:
                print("  Spatial filtering to Dutch city boundaries (+ 2 km buffer)...")
            bag_gdf = _spatial_filter_strtree(bag_gdf, nl_clip)
            if verbose:
                print(f"  Buildings within boundaries: {len(bag_gdf):,}")

            # Save
            if verbose:
                print(f"\n  Saving to: {bag_output}")
            bag_gdf.to_file(bag_output, driver="GPKG", layer="buildings")

            result["bag_path"] = bag_output
            result["bag_count"] = len(bag_gdf)

            if verbose:
                print(f"  ✓ BAG data prepared: {len(bag_gdf):,} buildings")
                print(f"  File size: {bag_output.stat().st_size / 1e9:.2f} GB")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("PREPARATION SUMMARY")
        print("=" * 80)

        if result["sirene_path"] or result["bag_path"]:
            print("\n✓ Comparison datasets prepared:")
            if result["sirene_path"]:
                size_gb = result["sirene_path"].stat().st_size / 1e9
                print(f"  - sirene_france.gpkg ({size_gb:.2f} GB, {result['sirene_count']:,} establishments)")
            if result["bag_path"]:
                size_gb = result["bag_path"].stat().st_size / 1e9
                print(f"  - bag_netherlands.gpkg ({size_gb:.2f} GB, {result['bag_count']:,} buildings)")
        else:
            print("\n⚠ No comparison datasets prepared")
            print("  Place source data in validation directory and re-run")

    return result


if __name__ == "__main__":
    from config import BOUNDS_VALIDATION_PATH, VALIDATION_DIR

    run_prepare_reference_data(
        bounds_path=BOUNDS_VALIDATION_PATH,
        validation_dir=VALIDATION_DIR,
        force_reload=False,
        verbose=True,
    )
