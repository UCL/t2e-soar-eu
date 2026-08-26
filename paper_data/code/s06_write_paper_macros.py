"""Generate LaTeX macros for the data paper.

Produces `paper_macros.tex` with city-count, validation, and data-coverage
macros used in the manuscript. City totals are taken from the processed archive
set in `PROCESSED_DIR`. Coverage statistics are derived from the per-column
`completeness_coverage.csv` produced by the audit step.
"""

import datetime as _dt
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from config import BOUNDS_PATH, BOUNDS_VALIDATION_PATH, OUTPUT_DIR, PAPER_DIR, PROCESSED_DIR

# Heights raster directory (per-city .tif files from Stage 4).
DATA_ROOT = Path(os.environ.get("T2E_DATA_DIR", ""))
HEIGHTS_DIR = DATA_ROOT / "cities_data" / "heights"


def update_stats():
    """Compute statistics directly from boundary files and analysis outputs."""
    stats = {}

    # Total active city archives in the release set (main processed dir only).
    active_city_archives = sorted(PROCESSED_DIR.glob("metrics_*.gpkg.zip"))
    if active_city_archives:
        stats["n_cities_total"] = len(active_city_archives)
        # Dataset size figures (compressed). Round to integer GB / single
        # decimal MB so prose remains stable across rebuilds.
        sizes_bytes = np.array([p.stat().st_size for p in active_city_archives], dtype=float)
        stats["dataset_total_compressed_gb"] = int(round(sizes_bytes.sum() / 1e9))
        stats["dataset_min_mb"] = round(float(sizes_bytes.min() / 1e6), 1)
        stats["dataset_max_gb"] = round(float(sizes_bytes.max() / 1e9), 1)
        stats["dataset_mean_mb"] = int(round(sizes_bytes.mean() / 1e6))
    elif BOUNDS_PATH.exists():
        bounds_gdf = gpd.read_file(BOUNDS_PATH)
        stats["n_cities_total"] = len(bounds_gdf)

    # Country roster from the main boundaries file.
    if BOUNDS_PATH.exists():
        bounds_gdf = gpd.read_file(BOUNDS_PATH)
        stats["n_countries"] = int(bounds_gdf["country"].nunique())

    # Per-city building-height rasters actually present on disk
    # (distinguishes "raster missing" from "raster present but no usable
    # pixels inside the city extent", which is the n_cities_no_height
    # figure derived from completeness_coverage).
    if HEIGHTS_DIR.exists():
        stats["n_cities_with_heights_raster"] = len(list(HEIGHTS_DIR.glob("*.tif")))

    # Age of the Copernicus building-height raster, derived from current
    # year so the manuscript stays accurate across rebuilds. Reference
    # year for the DHM is 2012.
    stats["bldg_heights_age_years"] = _dt.date.today().year - 2012

    # Reference city counts from validation boundaries
    if BOUNDS_VALIDATION_PATH.exists():
        val_gdf = gpd.read_file(BOUNDS_VALIDATION_PATH)
        stats["n_ref_cities"] = len(val_gdf)
        country_counts = val_gdf["country"].value_counts()
        stats["n_ref_cities_fr"] = int(country_counts.get("France", 0))
        stats["n_ref_cities_nl"] = int(country_counts.get("Netherlands", 0))

    # Derived count
    if "n_cities_total" in stats and "n_ref_cities" in stats:
        stats["n_non_ref_cities"] = stats["n_cities_total"] - stats["n_ref_cities"]

    # Coverage stats from per-column audit
    coverage_csv = PAPER_DIR / "completeness_coverage.csv"
    if coverage_csv.exists():
        cov = pd.read_csv(coverage_csv)

        # Total street segments across all cities (millions)
        streets_n = cov[cov["layer"] == "streets"].groupby("bounds_fid")["n_features"].max()
        stats["n_segments_total_m"] = round(float(streets_n.sum() / 1e6), 1)

        # Building height coverage
        bh = cov[(cov["layer"] == "buildings") & (cov["column"] == "mean_height")]
        if not bh.empty:
            stats["n_cities_low_height"] = int((bh["coverage"] < 0.50).sum())
            stats["n_cities_no_height"] = int((bh["coverage"] == 0.0).sum())
            stats["median_height_coverage"] = round(float(bh["coverage"].median() * 100))

        # Block contextual morphology metrics at 200m. Coverage differs by
        # metric group (geometry > FAR/GSI/L > height-dependent), so report
        # the range of per-column median coverage. Count columns
        # (cc_block_200_nw) and area sums (always complete) are excluded.
        block200 = cov[
            (cov["layer"] == "streets")
            & cov["column"].str.startswith("cc_block_")
            & cov["column"].str.contains("_200_")
            & ~cov["column"].str.match(r"^cc_block_\d+_")
            & ~cov["column"].str.startswith("cc_block_area_sum")
        ]
        if not block200.empty:
            per_col = block200.groupby("column")["coverage"].median() * 100
            stats["median_block200_coverage_min"] = round(float(per_col.min()))
            stats["median_block200_coverage_max"] = round(float(per_col.max()))
            # Cities with no qualifying blocks at all: zero coverage on the
            # best-covered (geometry) column, so missing heights don't inflate it.
            best = block200[block200["column"] == per_col.idxmax()]
            stats["n_cities_no_blocks"] = int((best["coverage"] == 0.0).sum())

        # Employment demographics coverage
        emp = cov[(cov["layer"] == "streets") & (cov["column"] == "emp")]
        if not emp.empty:
            stats["n_cities_no_employment"] = int((emp["coverage"] == 0.0).sum())

    # POI nearest-distance F1 thresholds (within-city scope) at the 0.80
    # support threshold. The "good" cluster is the three commercial
    # categories that reach F1 >= 0.80 fastest; the "other" cluster is
    # education + health. Accommodation is reported separately because it
    # lags. These macros are referenced from both papers, so we emit them
    # here as the data paper is the canonical source.
    f1_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_nearest_tolerance_agreement.csv"
    if f1_csv.exists():
        f1_df = pd.read_csv(f1_csv)

        def _min_tol_for_f1(category, threshold=0.80):
            sub = f1_df[f1_df["category"] == category]
            if sub.empty:
                return None
            med = sub.groupby("tolerance_m")["f1"].median().sort_index()
            ok = med[med >= threshold]
            return int(ok.index.min()) if len(ok) else None

        good = [_min_tol_for_f1(c) for c in ("retail", "eat_and_drink", "business_and_services")]
        other = [_min_tol_for_f1(c) for c in ("education", "health_and_medical")]
        accom = _min_tol_for_f1("accommodation")
        if all(t is not None for t in good):
            stats["poi_near_dist_good_m"] = int(max(good))
        if all(t is not None for t in other):
            stats["poi_near_dist_other_m"] = int(max(other))
        if accom is not None:
            stats["poi_near_dist_accom_m"] = int(accom)

    return stats


def generate_macros(stats: dict) -> str:
    """Generate LaTeX macro definitions."""
    lines = [
        "% Auto-generated LaTeX macros",
        "% Generated by: python paper_data/code/s06_write_paper_macros.py",
        "% DO NOT EDIT MANUALLY - regenerate from source data",
    ]

    macro_map = {
        "n_cities_total": ("NCitiesTotal", "City Counts"),
        "n_countries": ("NCountries", None),
        "n_ref_cities": ("NRefCities", None),
        "n_ref_cities_fr": ("NRefCitiesFr", None),
        "n_ref_cities_nl": ("NRefCitiesNl", None),
        "n_non_ref_cities": ("NNonRefCities", None),
        "n_segments_total_m": ("NSegmentsTotalM", None),
        "n_cities_low_height": ("NCitiesLowHeight", "Data Coverage"),
        "n_cities_no_height": ("NCitiesNoHeight", None),
        "n_cities_with_heights_raster": ("NCitiesWithHeightsRaster", None),
        "bldg_heights_age_years": ("BldgHeightsAgeYears", None),
        "median_height_coverage": ("MedianHeightCoverage", None),
        "median_block200_coverage_min": ("MedianBlockCoverageMin", None),
        "median_block200_coverage_max": ("MedianBlockCoverageMax", None),
        "n_cities_no_blocks": ("NCitiesNoBlocks", None),
        "n_cities_no_employment": ("NCitiesNoEmployment", None),
        "dataset_total_compressed_gb": ("DatasetTotalCompressedGB", "Dataset Size"),
        "dataset_min_mb": ("DatasetMinMB", None),
        "dataset_max_gb": ("DatasetMaxGB", None),
        "dataset_mean_mb": ("DatasetMeanMB", None),
        "poi_near_dist_good_m": ("PoiNearDistGoodM", "POI Validation Thresholds"),
        "poi_near_dist_other_m": ("PoiNearDistOtherM", None),
        "poi_near_dist_accom_m": ("PoiNearDistAccomM", None),
    }

    current_section = None
    for key, (macro_name, section) in macro_map.items():
        if section and section != current_section:
            if current_section is not None:
                lines.append("")
            lines.append(f"% === {section} ===")
            current_section = section
        if key in stats:
            lines.append(f"\\newcommand{{\\{macro_name}}}{{{stats[key]}}}")

    lines.append("")
    return "\n".join(lines)


def main():
    print("Generating LaTeX macros ...")

    stats = update_stats()

    # Save stats to JSON
    stats_path = PAPER_DIR / "paper_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Generate and save macros
    macros = generate_macros(stats)
    output_path = PAPER_DIR / "paper_macros.tex"
    with open(output_path, "w") as f:
        f.write(macros)
    print(f"  Saved: {output_path}")
    print(f"  {len(stats)} macros generated")


if __name__ == "__main__":
    main()
