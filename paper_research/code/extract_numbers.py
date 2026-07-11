"""Extract key numbers for the atlas manuscript prose.

Computes all statistics needed for plates 1-10 text: axis correlations,
octant segment shares, service distances by octant, desert fractions,
P25/P75 inequality gaps, density quintile form gaps, and paired
comparison raw values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import AXIS_COLS, BOUNDARIES_PATH, OCTANT_ORDER, classify_octants, load_all_cached

# ── Service distance columns ──
SVC_COLS = {
    "trees": "cc_trees_nearest_max_1600",
    "green": "cc_green_nearest_max_1600",
    "retail": "cc_retail_nearest_max_1600",
    "business": "cc_business_and_services_nearest_max_1600",
    "eat_drink": "cc_eat_and_drink_nearest_max_1600",
    "health": "cc_health_and_medical_nearest_max_1600",
    "education": "cc_education_nearest_max_1600",
    "transport": "cc_transport_nearest_max_1600",
    "accommodation": "cc_accommodation_nearest_max_1600",
    "arts": "cc_arts_and_entertainment_nearest_max_1600",
    "attractions": "cc_attractions_and_activities_nearest_max_1600",
    "religious": "cc_religious_nearest_max_1600",
}

# ── Morphometric columns ──
MORPH_COLS = {
    "height": "cc_mean_height_median_400_wt",
    "volume": "cc_mean_volume_median_200_wt",
    "area": "cc_mean_area_median_200_wt",
    "perimeter": "cc_mean_perimeter_median_200_wt",
    "compactness": "cc_mean_compactness_median_200_wt",
    "corners": "cc_mean_corners_median_200_wt",
    "shared_wall": "cc_shared_wall_length_median_400_wt",
    "swr": "cc_shared_wall_ratio_median_400_wt",
    "frontage": "frontage_max",
    "gsi": "cc_block_covered_ratio_median_400_wt",
    "fsi": "cc_block_far_median_400_wt",
    "osr": "cc_block_osr_median_400_wt",
    "mean_floors": "cc_block_mean_height_median_400_wt",
}

DEMO_COLS = {
    "density": "density",
    "employment": "emp_%",
    "youth": "y_lt15_%",
    "working_age": "y_1564_%",
    "elderly": "y_ge65_%",
}


def main():
    print("=" * 70)
    print("ATLAS MANUSCRIPT — NUMBER EXTRACTION")
    print("=" * 70)

    # Load all needed columns
    all_cols = list(AXIS_COLS.values()) + list(SVC_COLS.values()) + list(MORPH_COLS.values()) + list(DEMO_COLS.values())
    df = load_all_cached(columns=all_cols)
    n_cities = df["bounds_fid"].nunique()
    print(f"\nLoaded {len(df):,} segments across {n_cities} cities")

    # Classify octants
    classified, thresholds = classify_octants(df)
    print(f"Classified {len(classified):,} segments")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 1: Axis correlations and segment shares
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 1 — GRAMMAR")
    print("=" * 70)

    # Axis correlations (Spearman)
    from scipy.stats import spearmanr

    axes = classified[list(AXIS_COLS.values())].dropna()
    for a, b in [("intensity", "continuity"), ("intensity", "irregularity"), ("continuity", "irregularity")]:
        rho, p = spearmanr(axes[AXIS_COLS[a]], axes[AXIS_COLS[b]])
        print(f"  {a} vs {b}: rho={rho:.3f} (p={p:.2e})")

    # Segment shares per octant
    oct_counts = classified["octant"].value_counts()
    total = oct_counts.sum()
    print("\n  Segment shares:")
    for o in OCTANT_ORDER:
        n = oct_counts.get(o, 0)
        print(f"    {o}: {n:>8,} ({100 * n / total:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 3: Building morphometrics by octant
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 3 — BUILDINGS")
    print("=" * 70)

    for name, col in MORPH_COLS.items():
        if col not in classified.columns:
            continue
        overall_med = classified[col].median()
        by_oct = classified.groupby("octant")[col].median()
        print(f"\n  {name} (European median: {overall_med:.2f}):")
        for o in OCTANT_ORDER:
            v = by_oct.get(o, np.nan)
            print(f"    {o}: {v:.2f}")

    # Compactness CV
    if "cc_mean_compactness_median_200_wt" in classified.columns:
        city_compact = classified.groupby("bounds_fid")["cc_mean_compactness_median_200_wt"].median()
        cv = city_compact.std() / city_compact.mean()
        print(f"\n  Compactness CV across cities: {cv:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 4: Service distances by octant + overall
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 4 — ACCESS")
    print("=" * 70)

    # City-level medians
    city_meds = classified.groupby("bounds_fid")[list(SVC_COLS.values())].median()
    print("\n  City-level median distances (median of city medians):")
    for name, col in SVC_COLS.items():
        if col in city_meds.columns:
            v = city_meds[col].median()
            print(f"    {name}: {v:.0f}m")

    # Octant-level medians
    print("\n  Octant-level segment median distances:")
    for name, col in SVC_COLS.items():
        if col not in classified.columns:
            continue
        by_oct = classified.groupby("octant")[col].median()
        vals = [f"{by_oct.get(o, np.nan):.0f}" for o in OCTANT_ORDER]
        ratio = by_oct.get("LLL", np.nan) / by_oct.get("HHH", np.nan) if by_oct.get("HHH", 0) > 0 else np.nan
        print(f"    {name}: {' / '.join(vals)}  (LLL/HHH ratio: {ratio:.1f}x)")

    # Restaurant desert fraction (% cities with median eat_drink > 400m)
    eat_col = SVC_COLS["eat_drink"]
    if eat_col in city_meds.columns:
        pct_desert = (city_meds[eat_col] >= 400).mean() * 100
        print(f"\n  Cities with median eat&drink >= 400m: {pct_desert:.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 5: Demographics by octant
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 5 — DEMOGRAPHICS")
    print("=" * 70)

    for name, col in DEMO_COLS.items():
        if col not in classified.columns:
            continue
        by_oct = classified.groupby("octant")[col].median()
        overall = classified[col].median()
        print(f"\n  {name} (European median: {overall:.2f}):")
        for o in OCTANT_ORDER:
            v = by_oct.get(o, np.nan)
            print(f"    {o}: {v:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 6: Null correlation (green vs retail)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 6 / DISCUSSION — NULL CORRELATION")
    print("=" * 70)

    retail_col = SVC_COLS["retail"]
    green_col = SVC_COLS["green"]
    trees_col = SVC_COLS["trees"]
    if retail_col in city_meds.columns and green_col in city_meds.columns:
        from scipy.stats import pearsonr

        r_green, p_green = pearsonr(city_meds[retail_col].dropna(), city_meds[green_col].dropna())
        print(f"  Retail vs Green (city medians): r={r_green:.3f} (p={p_green:.3f})")
    if retail_col in city_meds.columns and trees_col in city_meds.columns:
        valid = city_meds[[retail_col, trees_col]].dropna()
        r_trees, p_trees = pearsonr(valid[retail_col], valid[trees_col])
        print(f"  Retail vs Trees (city medians): r={r_trees:.3f} (p={p_trees:.3f})")

    # Cities achieving both: close services AND close green
    if retail_col in city_meds.columns and green_col in city_meds.columns:
        med_retail = city_meds[retail_col].median()
        med_green = city_meds[green_col].median()
        both = city_meds[(city_meds[retail_col] <= med_retail) & (city_meds[green_col] <= med_green)]
        print(f"\n  Cities with below-median retail AND green distance: {len(both)}")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 8: Density quintile form gaps
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 8 — SAME DENSITY, DIFFERENT CITY")
    print("=" * 70)

    dens_col = DEMO_COLS["density"]
    if dens_col in classified.columns:
        valid = classified[classified[dens_col] > 0].copy()
        valid["dq"] = pd.qcut(valid[dens_col], 5, labels=False, duplicates="drop") + 1

        dq_stats = valid.groupby("dq")[dens_col].agg(["median", "min", "max"])
        print("\n  Density quintiles:")
        for q in dq_stats.index:
            row = dq_stats.loc[q]
            print(f"    Q{q}: {row['min']:,.0f} – {row['max']:,.0f}  (med {row['median']:,.0f}/km²)")

        for svc_name, svc_col in [
            ("retail", SVC_COLS["retail"]),
            ("education", SVC_COLS["education"]),
            ("transport", SVC_COLS["transport"]),
            ("green", SVC_COLS["green"]),
        ]:
            if svc_col not in valid.columns:
                continue
            print(f"\n  {svc_name} — form gap by quintile:")
            qo = valid.groupby(["dq", "octant"])[svc_col].median().unstack("octant")
            for q in sorted(valid["dq"].unique()):
                if q not in qo.index:
                    continue
                row = qo.loc[q]
                lo, hi = row.min(), row.max()
                gap = hi - lo
                best = row.idxmin()
                worst = row.idxmax()
                print(f"    Q{q}: {lo:.0f}m ({best}) – {hi:.0f}m ({worst}), Δ={gap:.0f}m")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 9: Desert fractions
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 9 — SERVICE DESERT")
    print("=" * 70)

    DESERT_THRESHOLD = 400
    for name, col in SVC_COLS.items():
        if col not in classified.columns:
            continue
        vals = classified[["octant", col]].dropna(subset=[col])
        vals["is_desert"] = (vals[col] >= DESERT_THRESHOLD).astype(float)
        by_oct = vals.groupby("octant")["is_desert"].mean() * 100
        overall = vals["is_desert"].mean() * 100
        lo, hi = by_oct.min(), by_oct.max()
        best = by_oct.idxmin()
        worst = by_oct.idxmax()
        print(f"  {name}: {lo:.0f}% ({best}) – {hi:.0f}% ({worst})  [overall: {overall:.0f}%]")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 10: P25-P75 inequality gaps
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 10 — INEQUALITY")
    print("=" * 70)

    for name, col in SVC_COLS.items():
        if col not in classified.columns:
            continue
        print(f"\n  {name}:")
        for o in OCTANT_ORDER:
            ov = classified.loc[classified["octant"] == o, col].dropna()
            if len(ov) < 50:
                continue
            p25 = ov.quantile(0.25)
            p75 = ov.quantile(0.75)
            gap = p75 - p25
            print(f"    {o}: P25={p25:.0f}m, P75={p75:.0f}m, gap={gap:.0f}m")

    # ═══════════════════════════════════════════════════════════════════
    # PLATE 7: Paired comparison raw values
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PLATE 7 — COMPARISONS (raw city medians)")
    print("=" * 70)

    import geopandas as gpd

    bounds = gpd.read_file(BOUNDARIES_PATH).to_crs(3035)
    all_metric_cols = list(
        dict.fromkeys(
            list(SVC_COLS.values()) + list(MORPH_COLS.values()) + list(DEMO_COLS.values()) + list(AXIS_COLS.values())
        )
    )
    available = [c for c in all_metric_cols if c in classified.columns]
    city_df = classified.groupby("bounds_fid")[available].median()
    city_meta = classified.merge(bounds[["bounds_fid", "label", "country"]], on="bounds_fid")
    city_country = city_meta.groupby("bounds_fid")[["country", "label"]].first()
    city_df = city_df.join(city_country)

    # Key comparisons
    comparisons = [
        ("Nordic", {"country": ["Norway", "Finland"]}, "Mediterranean", {"country": ["Spain", "Greece"]}),
        ("Netherlands", {"cities": ["Amsterdam", "Utrecht"]}, "Belgium", {"cities": ["Brussels", "Ghent"]}),
        ("Poland", {"country": ["Poland"]}, "Romania", {"country": ["Romania"]}),
        (
            "Northern Italy",
            {"cities": ["Milan", "Turin", "Bologna"]},
            "Southern Italy",
            {"cities": ["Naples", "Palermo", "Bari"]},
        ),
        (
            "Alpine",
            {"cities": ["Innsbruck", "Salzburg", "Bern", "Freiburg im Breisgau", "St. Gallen"]},
            "Host Countries",
            {"country": ["Austria", "Switzerland", "Germany"]},
        ),
    ]

    report_metrics = [
        ("FSI", AXIS_COLS["intensity"]),
        ("Frontage", AXIS_COLS["continuity"]),
        ("MAD", AXIS_COLS["irregularity"]),
        ("Height", MORPH_COLS["height"]),
        ("Retail dist", SVC_COLS["retail"]),
        ("Eat&Drink dist", SVC_COLS["eat_drink"]),
        ("Green dist", SVC_COLS["green"]),
        ("Tree dist", SVC_COLS["trees"]),
        ("Pop density", DEMO_COLS["density"]),
        ("Elderly %", DEMO_COLS["elderly"]),
        ("Street density", "cc_density_800"),
    ]

    for a_label, a_filter, b_label, b_filter in comparisons:
        print(f"\n  {a_label} vs {b_label}:")
        if "country" in a_filter:
            a_df = city_df[city_df["country"].isin(a_filter["country"])]
        else:
            a_fids = bounds[bounds["label"].isin(a_filter["cities"])]["bounds_fid"].values
            a_df = city_df[city_df.index.isin(a_fids)]
        if "country" in b_filter:
            b_df = city_df[city_df["country"].isin(b_filter["country"])]
        else:
            b_fids = bounds[bounds["label"].isin(b_filter["cities"])]["bounds_fid"].values
            b_df = city_df[city_df.index.isin(b_fids)]

        # For Alpine vs Host, exclude alpine cities from host
        if a_label == "Alpine":
            b_df = b_df[~b_df.index.isin(a_df.index)]

        print(f"    ({len(a_df)} vs {len(b_df)} cities)")
        for mname, mcol in report_metrics:
            if mcol not in city_df.columns:
                continue
            a_vals = pd.to_numeric(a_df[mcol], errors="coerce")
            b_vals = pd.to_numeric(b_df[mcol], errors="coerce")
            a_med = float(a_vals.median())
            b_med = float(b_vals.median())
            print(f"    {mname}: {a_med:.1f} vs {b_med:.1f}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
