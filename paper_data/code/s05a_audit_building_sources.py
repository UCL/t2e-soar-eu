#!/usr/bin/env python3
"""Audit building footprint provenance and test for source-driven metric bias.

For each city, parses the Overture ``sources`` JSON on the buildings layer to
classify each footprint as community-contributed (OpenStreetMap, Esri Community
Maps, national cadastres) or ML-derived (Microsoft ML Buildings, Google Open
Buildings).  Then tests whether the proportion of ML-derived buildings correlates
with morphometric values — and whether octant classification is confounded by
source composition.

Outputs (paper_data/outputs/csv/):
    building_source_counts.csv   — per-city counts by dataset
    building_source_metrics.csv  — per-city ML fraction + median morphometrics
    building_source_by_country.csv — country-level summary
    building_source_correlations.csv — correlations between ML fraction and metrics
    building_source_octant_test.csv — octant × ML-fraction independence test

Examples:
    uv run python paper_data/code/s05a_audit_building_sources.py
    uv run python paper_data/code/s05a_audit_building_sources.py --workers 4
    uv run python paper_data/code/s05a_audit_building_sources.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from config import BOUNDS_PATH, CSV_DIR, PROCESSED_DIR
from scipy import stats

# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

# Datasets that are community-contributed (human-digitised or cadastral imports)
COMMUNITY_DATASETS = {
    "OpenStreetMap",
    "Esri Community Maps",
    "Instituto Geografico Nacional",  # Spain cadastre
    "City of Vancouver",
}

# Everything else is ML-derived (Microsoft ML Buildings, Google Open Buildings,
# Buildings in East Asian Countries, etc.)


def _classify_source(dataset_name: str) -> str:
    """Classify a source dataset as 'community' or 'ml'."""
    for prefix in COMMUNITY_DATASETS:
        if dataset_name.startswith(prefix):
            return "community"
    return "ml"


# Morphometric columns present on the buildings layer
MORPH_COLS = [
    "area",
    "perimeter",
    "compactness",
    "orientation",
    "corners",
    "shape_index",
    "fractal_dimension",
    "shared_walls",
    "shared_wall_ratio",
    "mean_height",
    "volume",
    "form_factor",
]


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------


def _audit_city(task: dict) -> dict | None:
    """Extract source composition and per-source morphometrics for one city."""
    bounds_fid = str(task["bounds_fid"])
    path = Path(str(task["metrics_path"]))
    city_label = str(task.get("city_label", ""))
    country = str(task.get("country", ""))

    if not path.exists():
        return None

    try:
        available_layers = {name for name, _ in pyogrio.list_layers(path)}
    except Exception:
        return None

    if "buildings" not in available_layers:
        return None

    # Read buildings with sources + morphometrics
    try:
        read_cols = ["sources"] + MORPH_COLS
        df = pyogrio.read_dataframe(path, layer="buildings", columns=read_cols, read_geometry=False)
    except Exception as exc:
        print(f"  Warning: {bounds_fid} buildings: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return None

    if df.empty:
        return None

    # Parse sources JSON → extract primary dataset name and classification
    def _extract(sources_str):
        try:
            parsed = json.loads(sources_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0].get("dataset", "unknown")
        except (json.JSONDecodeError, TypeError):
            pass
        return "unknown"

    df["dataset"] = df["sources"].apply(_extract)
    df["source_class"] = df["dataset"].apply(_classify_source)

    # --- Per-dataset counts ---
    dataset_counts = df["dataset"].value_counts().to_dict()

    # --- Per-source-class morphometrics ---
    n_total = len(df)
    n_ml = int((df["source_class"] == "ml").sum())
    n_community = n_total - n_ml
    ml_frac = n_ml / n_total if n_total > 0 else np.nan

    # City-wide median morphometrics
    city_medians = {}
    for col in MORPH_COLS:
        if col in df.columns:
            city_medians[f"median_{col}"] = df[col].median()

    # Per-source-class medians (for within-city comparison)
    for src_class in ("community", "ml"):
        subset = df[df["source_class"] == src_class]
        for col in MORPH_COLS:
            if col in df.columns:
                city_medians[f"median_{col}_{src_class}"] = subset[col].median() if len(subset) > 0 else np.nan

    # Mann-Whitney U tests: community vs ML for shape-sensitive metrics
    mw_results = {}
    for col in ["compactness", "fractal_dimension", "shared_wall_ratio", "corners", "shape_index"]:
        if col not in df.columns:
            continue
        comm = df.loc[df["source_class"] == "community", col].dropna()
        ml = df.loc[df["source_class"] == "ml", col].dropna()
        if len(comm) >= 20 and len(ml) >= 20:
            stat, pval = stats.mannwhitneyu(comm, ml, alternative="two-sided")
            # rank-biserial correlation as effect size
            n1, n2 = len(comm), len(ml)
            r_rb = 1 - (2 * stat) / (n1 * n2)
            mw_results[f"mw_U_{col}"] = stat
            mw_results[f"mw_p_{col}"] = pval
            mw_results[f"mw_r_{col}"] = r_rb  # rank-biserial correlation
        else:
            mw_results[f"mw_U_{col}"] = np.nan
            mw_results[f"mw_p_{col}"] = np.nan
            mw_results[f"mw_r_{col}"] = np.nan

    return {
        "bounds_fid": bounds_fid,
        "city_label": city_label,
        "country": country,
        "n_buildings": n_total,
        "n_community": n_community,
        "n_ml": n_ml,
        "ml_fraction": ml_frac,
        "dataset_counts": dataset_counts,
        **city_medians,
        **mw_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for test runs.")
    args = parser.parse_args(argv)

    counts_path = CSV_DIR / "building_source_counts.csv"
    metrics_path = CSV_DIR / "building_source_metrics.csv"
    country_path = CSV_DIR / "building_source_by_country.csv"
    # country_path is the last unconditional write, so its presence signals a
    # completed run; require all three (downstream s05c reads by_country + metrics).
    if counts_path.exists() and metrics_path.exists() and country_path.exists() and args.limit is None:
        print(f"✓ Building-source audit outputs already exist, skipping: {counts_path.name}, {metrics_path.name}, {country_path.name}")
        return 0

    # Load city metadata
    bounds = pyogrio.read_dataframe(BOUNDS_PATH, columns=["bounds_fid", "label", "country"], read_geometry=False)
    bounds["bounds_fid"] = bounds["bounds_fid"].astype(str)

    tasks = [
        {
            "bounds_fid": row.bounds_fid,
            "city_label": row.label,
            "country": row.country,
            "metrics_path": str(PROCESSED_DIR / f"metrics_{row.bounds_fid}.gpkg.zip"),
        }
        for row in bounds.itertuples(index=False)
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    total = len(tasks)
    print(f"Auditing building sources for {total} cities (workers={args.workers})")

    results: list[dict] = []
    completed = 0

    if args.workers <= 1:
        for task in tasks:
            r = _audit_city(task)
            if r is not None:
                results.append(r)
            completed += 1
            if completed % 25 == 0:
                print(f"  {completed}/{total}")
    else:
        max_workers = min(args.workers, max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_audit_city, t): t for t in tasks}
            for future in as_completed(future_map):
                try:
                    r = future.result()
                    if r is not None:
                        results.append(r)
                except Exception as exc:
                    task = future_map[future]
                    print(f"  Error: {task['bounds_fid']}: {exc}")
                completed += 1
                if completed % 25 == 0:
                    print(f"  {completed}/{total}")

    if not results:
        print("No results collected.")
        return 1

    # -----------------------------------------------------------------------
    # 1. Per-city source counts (long format)
    # -----------------------------------------------------------------------
    count_rows = []
    for r in results:
        for dataset, count in r["dataset_counts"].items():
            count_rows.append(
                {
                    "bounds_fid": r["bounds_fid"],
                    "city_label": r["city_label"],
                    "country": r["country"],
                    "dataset": dataset,
                    "n_buildings": count,
                }
            )
    counts_df = pd.DataFrame(count_rows).sort_values(["country", "city_label", "dataset"])
    counts_df.to_csv(counts_path, index=False)
    print(f"\nWrote per-city source counts: {counts_path}")

    # -----------------------------------------------------------------------
    # 2. Per-city ML fraction + morphometrics (wide format)
    # -----------------------------------------------------------------------
    metrics_rows = [{k: v for k, v in r.items() if k != "dataset_counts"} for r in results]
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["country", "city_label"])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote per-city metrics:       {metrics_path}")

    # -----------------------------------------------------------------------
    # 3. Country-level summary
    # -----------------------------------------------------------------------
    country_agg = (
        metrics_df.groupby("country")
        .agg(
            n_cities=("bounds_fid", "count"),
            total_buildings=("n_buildings", "sum"),
            total_ml=("n_ml", "sum"),
            total_community=("n_community", "sum"),
            mean_ml_fraction=("ml_fraction", "mean"),
            median_ml_fraction=("ml_fraction", "median"),
            min_ml_fraction=("ml_fraction", "min"),
            max_ml_fraction=("ml_fraction", "max"),
        )
        .reset_index()
    )
    country_agg["overall_ml_fraction"] = country_agg["total_ml"] / country_agg["total_buildings"]
    country_agg = country_agg.sort_values("overall_ml_fraction", ascending=False)
    country_agg.to_csv(country_path, index=False)
    print(f"Wrote country summary:        {country_path}")

    # Print country summary
    print("\n--- Country summary (sorted by ML fraction, descending) ---")
    for _, row in country_agg.iterrows():
        print(
            f"  {row['country']:20s}  "
            f"cities={int(row['n_cities']):3d}  "
            f"buildings={int(row['total_buildings']):>9,d}  "
            f"ML={row['overall_ml_fraction']:.1%}  "
            f"(range {row['min_ml_fraction']:.1%}–{row['max_ml_fraction']:.1%})"
        )

    # -----------------------------------------------------------------------
    # 4. Cross-city correlations: ML fraction vs morphometrics
    # -----------------------------------------------------------------------
    # Use Spearman (non-parametric) to test whether cities with more ML
    # buildings systematically differ in morphometric summaries.
    corr_rows = []
    target_cols = [c for c in metrics_df.columns if c.startswith("median_") and not c.endswith(("_community", "_ml"))]
    for col in target_cols:
        valid = metrics_df[["ml_fraction", col]].dropna()
        if len(valid) < 30:
            continue
        rho, p = stats.spearmanr(valid["ml_fraction"], valid[col])
        corr_rows.append(
            {"metric": col.replace("median_", ""), "spearman_rho": rho, "p_value": p, "n_cities": len(valid)}
        )
    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        print("\nToo few cities for cross-city correlations (need ≥30).")
    else:
        corr_df = corr_df.sort_values("p_value")
    corr_path = CSV_DIR / "building_source_correlations.csv"
    if not corr_df.empty:
        corr_df.to_csv(corr_path, index=False)
        print(f"\nWrote cross-city correlations: {corr_path}")
        print("\n--- Cross-city Spearman correlations: ML fraction vs city median metric ---")
        for _, row in corr_df.iterrows():
            sig = (
                "***"
                if row["p_value"] < 0.001
                else "**"
                if row["p_value"] < 0.01
                else "*"
                if row["p_value"] < 0.05
                else ""
            )
            print(f"  {row['metric']:25s}  rho={row['spearman_rho']:+.3f}  p={row['p_value']:.4f} {sig}")

    # -----------------------------------------------------------------------
    # 5. Within-city effect sizes: community vs ML buildings
    # -----------------------------------------------------------------------
    # Summarise the Mann-Whitney rank-biserial correlations across cities
    mw_cols = [c for c in metrics_df.columns if c.startswith("mw_r_")]
    if mw_cols:
        print("\n--- Within-city effect sizes (community vs ML buildings) ---")
        print("  Metric                    median r_rb   mean |r_rb|  cities with p<0.05")
        for col in mw_cols:
            metric_name = col.replace("mw_r_", "")
            p_col = col.replace("mw_r_", "mw_p_")
            valid_r = metrics_df[col].dropna()
            valid_p = metrics_df[p_col].dropna()
            n_sig = (valid_p < 0.05).sum()
            print(
                f"  {metric_name:25s}  "
                f"{valid_r.median():+.3f}         "
                f"{valid_r.abs().mean():.3f}        "
                f"{n_sig}/{len(valid_p)}"
            )

    # -----------------------------------------------------------------------
    # 6. Octant × ML-fraction test
    # -----------------------------------------------------------------------
    # Load the octant classification (requires atlas_common from paper_research)
    # We import lazily to keep this script runnable from paper_data/code/
    octant_test_results = _run_octant_test(metrics_df)
    if octant_test_results is not None:
        octant_path = CSV_DIR / "building_source_octant_test.csv"
        octant_test_results.to_csv(octant_path, index=False)
        print(f"\nWrote octant test:            {octant_path}")

    n_cities = len(results)
    n_datasets = counts_df["dataset"].nunique()
    print(f"\nDone: {n_cities} cities, {n_datasets} unique source datasets")
    return 0


def _run_octant_test(metrics_df: pd.DataFrame) -> pd.DataFrame | None:
    """Test whether octant assignment correlates with ML building fraction.

    Uses the atlas paper's classify_octants() to assign each street segment to
    an octant, then computes per-city dominant octant.  Tests:
      (a) Kruskal-Wallis: do octants differ in ML fraction?
      (b) Per-octant summary statistics.
      (c) Partial correlation: ML fraction vs axis values controlling for each other.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper_research" / "code"))
        from atlas_common import (
            AXIS_COLS,
            BOUNDARIES_PATH,
            OCTANT_ORDER,
            classify_octants,
            load_all_cached,
        )
    except ImportError as exc:
        print(f"  Skipping octant test (import failed: {exc})")
        return None

    # Load cached street-level data and classify octants
    print("\nLoading cached street-level data for octant classification...")
    df = load_all_cached(columns=list(AXIS_COLS.values()) + ["bounds_fid"])
    if df.empty:
        print("  No cached data available; skipping octant test.")
        return None

    classified, thresholds = classify_octants(df)
    print(f"  Classified {len(classified):,d} street segments into octants")
    print(f"  Thresholds: {thresholds}")

    # Dominant octant per city (plurality vote)
    city_octant = classified.groupby(["bounds_fid", "octant"]).size().unstack(fill_value=0)
    city_dom = city_octant.idxmax(axis=1).rename("dom_octant")
    # Also compute octant proportions per city
    city_octant_frac = city_octant.div(city_octant.sum(axis=1), axis=0)

    # Merge with ML fraction
    metrics_df = metrics_df.copy()
    metrics_df["bounds_fid"] = metrics_df["bounds_fid"].astype(int)
    merged = metrics_df.set_index("bounds_fid").join(city_dom).dropna(subset=["dom_octant"])

    if len(merged) < 30:
        print(f"  Only {len(merged)} cities with both sources and octant data; skipping test.")
        return None

    # (a) Kruskal-Wallis: ML fraction differs across octants?
    groups = [g["ml_fraction"].dropna().values for _, g in merged.groupby("dom_octant")]
    groups = [g for g in groups if len(g) >= 5]  # need at least 5 per group
    if len(groups) >= 3:
        h_stat, h_p = stats.kruskal(*groups)
        # Epsilon-squared effect size: H / (n-1)
        n = sum(len(g) for g in groups)
        eps_sq = h_stat / (n - 1) if n > 1 else np.nan
        print(f"\n  Kruskal-Wallis: H={h_stat:.2f}, p={h_p:.4f}, eps²={eps_sq:.4f}, k={len(groups)} octants")
    else:
        h_stat, h_p, eps_sq = np.nan, np.nan, np.nan
        print("  Too few octant groups for Kruskal-Wallis test")

    # (b) Per-octant summary
    octant_summary_rows = []
    for octant in OCTANT_ORDER:
        subset = merged[merged["dom_octant"] == octant]
        if len(subset) == 0:
            continue
        octant_summary_rows.append(
            {
                "octant": octant,
                "n_cities": len(subset),
                "mean_ml_fraction": subset["ml_fraction"].mean(),
                "median_ml_fraction": subset["ml_fraction"].median(),
                "std_ml_fraction": subset["ml_fraction"].std(),
                "mean_n_buildings": subset["n_buildings"].mean(),
            }
        )
    octant_summary = pd.DataFrame(octant_summary_rows)

    print("\n  --- ML fraction by dominant octant ---")
    for _, row in octant_summary.iterrows():
        print(
            f"    {row['octant']}  "
            f"n={int(row['n_cities']):3d}  "
            f"ML={row['mean_ml_fraction']:.1%} ± {row['std_ml_fraction']:.1%}  "
            f"(median {row['median_ml_fraction']:.1%})"
        )

    # (c) Partial correlations: does ML fraction predict each axis value
    #     after controlling for the other two axes?
    # Use city-level median axis values
    axis_cols_map = {
        "intensity": "median_" + AXIS_COLS["intensity"].replace("cc_", "").replace("_median_400_wt", ""),
        "continuity": "median_shared_wall_ratio",
        "irregularity": "median_" + AXIS_COLS["irregularity"].replace("cc_", "").replace("_median_400_wt", ""),
    }
    # Map to actual column names in metrics_df
    # The columns are named median_{metric} from the morphometrics
    # intensity = cc_block_far_median_400_wt → not directly available as building metric
    # continuity = cc_shared_wall_ratio_median_400_wt → building-level: shared_wall_ratio
    # irregularity = cc_orientation_mad_400_wt → building-level: orientation (but MAD is different)
    # We'll use building-level medians where available and note the limitation

    print("\n  --- Spearman correlations: ML fraction vs axis-proxy building metrics ---")
    for axis_name, col_name in [
        ("continuity (shared_wall_ratio)", "median_shared_wall_ratio"),
        ("complexity (fractal_dimension)", "median_fractal_dimension"),
        ("shape (compactness)", "median_compactness"),
        ("shape (corners)", "median_corners"),
        ("shape (shape_index)", "median_shape_index"),
    ]:
        if col_name not in merged.columns:
            continue
        valid = merged[["ml_fraction", col_name]].dropna()
        if len(valid) < 30:
            continue
        rho, p = stats.spearmanr(valid["ml_fraction"], valid[col_name])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {axis_name:40s}  rho={rho:+.3f}  p={p:.4f} {sig}")

    # (d) Key diagnostic: within cities that have BOTH community and ML buildings,
    #     do shared wall ratios differ by source? Aggregate the per-city MW tests.
    has_both = merged[(merged["n_community"] > 0) & (merged["n_ml"] > 0) & merged["mw_p_shared_wall_ratio"].notna()]
    if len(has_both) >= 10:
        # Combine p-values via Fisher's method
        valid_p = has_both["mw_p_shared_wall_ratio"].replace(0, 1e-300)  # avoid log(0)
        chi2_stat = -2 * np.sum(np.log(valid_p))
        combined_p = stats.chi2.sf(chi2_stat, df=2 * len(valid_p))
        median_effect = has_both["mw_r_shared_wall_ratio"].median()
        print("\n  --- Within-city shared_wall_ratio: community vs ML ---")
        print(f"    Cities with both sources: {len(has_both)}")
        print(f"    Median rank-biserial r:   {median_effect:+.3f}")
        print(f"    Fisher combined p-value:  {combined_p:.2e}")
        direction = "community > ML" if median_effect > 0 else "ML > community"
        print(f"    Direction:                {direction}")

    # Add global test stats to the summary
    octant_summary["kruskal_H"] = h_stat
    octant_summary["kruskal_p"] = h_p
    octant_summary["kruskal_eps_sq"] = eps_sq

    return octant_summary


if __name__ == "__main__":
    raise SystemExit(main())
