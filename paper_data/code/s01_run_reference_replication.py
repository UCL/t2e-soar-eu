#!/usr/bin/env python3
"""Point A: Compute node-level accessibility from Overture vs official registries.

This script is intended for *testing the workflow* on a small number of cities.
It mirrors SOAR's network construction (no decomposition, natural street
segments via CityNetwork.from_geopandas) and computes accessibilities
using the CityNetwork class-based API:
- Overture Places (treatment A)
- Registry POIs from SIRENE (FR) or BAG (NL) mapped into `merged_cats` (treatment B)

Outputs are written to `paper_data/outputs/replication/` by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import numpy as np
import pandas as pd
from cityseer.network import CityNetwork
from config import (
    BAG_USAGE_MAPPING,
    BOUNDS_VALIDATION_PATH,
    OUTPUT_DIR,
    OVERTURE_DIR,
    SIRENE_APE_MAPPING,
    VALIDATION_DIR,
)
from pointa_constants import DEFAULT_CATEGORIES
from pointa_utils import (
    DISTANCES_LU,
    WORKING_CRS,
    extract_accessibility_columns,
    map_bag_to_registry_places,
    map_sirene_to_registry_places,
)

from src.landuse_categories import merge_landuse_categories


def main() -> int:
    categories = list(DEFAULT_CATEGORIES)
    out_dir = OUTPUT_DIR / "replication"

    bounds_gdf = gpd.read_file(BOUNDS_VALIDATION_PATH, layer="bounds").set_index("bounds_fid").to_crs(WORKING_CRS)
    bounds_fids = [str(x) for x in bounds_gdf.index.tolist()]
    if not bounds_fids:
        raise RuntimeError("No bounds_fids found in boundaries file.")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_is_int = hasattr(bounds_gdf.index, "dtype") and np.issubdtype(bounds_gdf.index.dtype, np.integer)

    n_total = len(bounds_fids)
    n_missing_overture = 0
    n_done = 0
    poi_count_rows: list[dict] = []

    for bounds_fid in bounds_fids:
        resolved_fid: int | str
        if index_is_int:
            try:
                resolved_fid = int(bounds_fid)
            except Exception as exc:
                raise KeyError(f"bounds_fid must be an integer for this boundaries file: {bounds_fid}") from exc
        else:
            resolved_fid = str(bounds_fid)

        if resolved_fid not in bounds_gdf.index:
            raise KeyError(f"bounds_fid not found in bounds layer: {bounds_fid}")

        bounds_row = bounds_gdf.loc[resolved_fid]

        bounds_geom = bounds_row.geometry
        fid_str = str(resolved_fid)
        overture_gpkg = Path(OVERTURE_DIR) / f"overture_{fid_str}.gpkg"
        overture_gpkg_zip = Path(OVERTURE_DIR) / f"overture_{fid_str}.gpkg.zip"
        if overture_gpkg_zip.exists():
            overture_gpkg = overture_gpkg_zip
        elif not overture_gpkg.exists():
            n_missing_overture += 1
            print(f"⚠ Skipping bounds_fid={fid_str}: missing Overture city file: {overture_gpkg}")
            continue

        out_ovt = out_dir / f"pointa_nodes_overture_{fid_str}.parquet"
        out_reg = out_dir / f"pointa_nodes_registry_{fid_str}.parquet"

        if out_ovt.exists() and out_reg.exists():
            n_done += 1
            continue

        # Build network — mirrors production pipeline (no decomposition).
        clean_edges_gdf = gpd.read_file(overture_gpkg, layer="clean_edges").to_crs(WORKING_CRS)
        cn = CityNetwork.from_geopandas(clean_edges_gdf, crs=WORKING_CRS, boundary=bounds_geom)

        # Treatment A: Overture places.
        overture_places = gpd.read_file(overture_gpkg, layer="places").to_crs(WORKING_CRS)
        overture_places = overture_places[overture_places.intersects(bounds_geom.buffer(2000))].copy()
        overture_places = merge_landuse_categories(overture_places)
        overture_places = overture_places[overture_places["merged_cats"].isin(categories)].copy()

        if not overture_places.empty:
            keys = sorted(overture_places["merged_cats"].dropna().unique().tolist())
            if keys:
                cn.compute_accessibilities(
                    overture_places,
                    landuse_column_label="merged_cats",
                    accessibility_keys=keys,
                    distances=DISTANCES_LU,
                )
        nodes_ovt = cn.to_geopandas().copy(deep=True)
        nodes_ovt["bounds_fid"] = fid_str
        nodes_ovt["node_id"] = nodes_ovt.index.astype(str)
        centroids = nodes_ovt.geometry.centroid
        nodes_ovt["x"] = centroids.x
        nodes_ovt["y"] = centroids.y

        # Not strictly necessary as different col names and overwrite behaviour, but playing it safe.
        cc_cols = [c for c in cn.nodes_gdf.columns if c.startswith("cc_")]
        if cc_cols:
            cn.nodes_gdf.drop(columns=cc_cols, inplace=True)

        # Treatment B: Registry places.
        country = str(bounds_row.get("country"))
        if country == "France":
            registry_places = map_sirene_to_registry_places(
                sirene_gpkg=Path(VALIDATION_DIR) / "sirene_france.gpkg",
                bounds_geom=bounds_geom,
                sirene_ape_mapping=SIRENE_APE_MAPPING,
                categories=categories,
            )
        elif country == "Netherlands":
            registry_places = map_bag_to_registry_places(
                bag_gpkg=Path(VALIDATION_DIR) / "bag_netherlands.gpkg",
                bounds_geom=bounds_geom,
                bag_usage_mapping=BAG_USAGE_MAPPING,
                categories=categories,
            )
        else:
            raise RuntimeError(f"Unsupported country for Point A registry replication: {country}")

        # Log POI counts per category for both sources.
        ovt_counts = (
            overture_places["merged_cats"].value_counts() if not overture_places.empty else pd.Series(dtype=int)
        )
        reg_counts = (
            registry_places["merged_cats"].value_counts() if not registry_places.empty else pd.Series(dtype=int)
        )
        for cat in categories:
            poi_count_rows.append(
                {
                    "bounds_fid": fid_str,
                    "country": country,
                    "category": cat,
                    "n_overture": int(ovt_counts.get(cat, 0)),
                    "n_registry": int(reg_counts.get(cat, 0)),
                }
            )

        # Registry treatment: recompute accessibilities on same network.
        if not registry_places.empty:
            keys = sorted(registry_places["merged_cats"].dropna().unique().tolist())
            if keys:
                cn.compute_accessibilities(
                    registry_places,
                    landuse_column_label="merged_cats",
                    accessibility_keys=keys,
                    distances=DISTANCES_LU,
                )
        nodes_reg = cn.to_geopandas()
        nodes_reg["bounds_fid"] = fid_str
        nodes_reg["node_id"] = nodes_reg.index.astype(str)
        centroids_reg = nodes_reg.geometry.centroid
        nodes_reg["x"] = centroids_reg.x
        nodes_reg["y"] = centroids_reg.y

        # Persist minimal columns for comparisons (intersection of both).
        keep_cols_ovt = extract_accessibility_columns(nodes_ovt, categories)
        keep_cols_reg = extract_accessibility_columns(nodes_reg, categories)
        keep_cols = sorted(set(keep_cols_ovt) & set(keep_cols_reg))
        # Always keep base ID columns at front.
        base = [c for c in ["bounds_fid", "node_id", "live"] if c in nodes_ovt.columns]
        keep_cols = base + [c for c in keep_cols if c not in base]
        nodes_ovt = nodes_ovt[keep_cols].copy()
        nodes_reg = nodes_reg[keep_cols].copy()

        nodes_ovt.to_parquet(out_ovt, index=False)
        nodes_reg.to_parquet(out_reg, index=False)

        print(f"✓ Wrote: {out_ovt}")
        print(f"✓ Wrote: {out_reg}")

        n_done += 1
        if n_done % 25 == 0:
            print(f"…progress: wrote {n_done}/{n_total} cities (missing_overture={n_missing_overture})")

    print(f"Done. wrote={n_done} / requested={n_total}; missing_overture={n_missing_overture}")

    # Write POI count summary.
    if poi_count_rows:
        poi_counts_csv = out_dir / "pointa_poi_counts.csv"
        poi_df = pd.DataFrame(poi_count_rows)
        poi_df.to_csv(poi_counts_csv, index=False)
        print(f"✓ Wrote POI counts: {poi_counts_csv} ({len(poi_df)} rows)")
        # Flag suspicious cities.
        zero_reg = poi_df[poi_df["n_registry"] == 0]
        low_reg = poi_df[(poi_df["n_registry"] > 0) & (poi_df["n_registry"] < 10)]
        if not zero_reg.empty:
            print(f"⚠ {len(zero_reg)} city/category pairs with ZERO registry POIs:")
            for _, r in zero_reg.iterrows():
                print(f"    fid={r['bounds_fid']} {r['category']}: overture={r['n_overture']}, registry=0")
        if not low_reg.empty:
            print(f"⚠ {len(low_reg)} city/category pairs with <10 registry POIs:")
            for _, r in low_reg.iterrows():
                print(
                    f"    fid={r['bounds_fid']} {r['category']}: overture={r['n_overture']}, registry={r['n_registry']}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
