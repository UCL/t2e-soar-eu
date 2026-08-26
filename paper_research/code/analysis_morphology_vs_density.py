"""
Comprehensive statistical analysis: morphology vs population density.

Core question: How much of the variation in service access is explained by
morphological configuration (FSI, frontage, MAD — all three are morphological)
vs. population density (a demographic variable) vs. network structure?

The decomposition is:
  - Population density ALONE
  - Each morphological axis alone (FSI = intensity, frontage = continuity, MAD = irregularity)
  - All three morphological axes together
  - Network structure (closeness, network density)
  - Everything together

For green space: identify cities that achieve short distances on BOTH
commercial services and green space — proving the trade-off is a design
choice, not an inevitable constraint.

All designs use spatial subsampling (~10% thinning) to address 400m kernel
spatial autocorrelation, with city-clustered standard errors.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import AXIS_COLS, classify_octants, load_all_cached

# ============================================================================
# CONFIG
# ============================================================================
OUTCOME_COLS = {
    "retail": "cc_retail_nearest_max_1600",
    "eat_drink": "cc_eat_and_drink_nearest_max_1600",
    "education": "cc_education_nearest_max_1600",
    "health": "cc_health_and_medical_nearest_max_1600",
    "green": "cc_green_nearest_max_1600",
    "trees": "cc_trees_nearest_max_1600",
}

# --- Variable families ---
# MORPHOLOGICAL (all describe physical built form)
MORPH_INTENSITY = {
    "FSI": "cc_block_far_median_400_wt",
    "GSI": "cc_block_covered_ratio_median_400_wt",
    "OSR": "cc_block_osr_median_400_wt",
    "height": "cc_mean_height_median_400_wt",
    "bldg_count": "cc_building_400_nw",
    "volume": "cc_volume_median_400_wt",
}
MORPH_CONTINUITY = {
    "frontage": "frontage_max",
    "SWR": "cc_shared_wall_ratio_median_400_wt",
}
MORPH_IRREGULARITY = {
    "MAD": "cc_orientation_mad_400_wt",
}

# THREE AXES (one representative per axis)
THREE_AXES = {
    "FSI": "cc_block_far_median_400_wt",
    "frontage": "frontage_max",
    "MAD": "cc_orientation_mad_400_wt",
}

# POPULATION DENSITY (demographic, not morphological)
POP_DENSITY = {
    "pop_density": "density",
}

# NETWORK STRUCTURE (graph properties)
NETWORK = {
    "closeness_800": "cc_beta_800",
    "net_density_400": "cc_density_400",
    "betweenness_800": "cc_betweenness_beta_800",
}

# EXTENDED MORPHOLOGY (all morphological variables)
ALL_MORPH = {**MORPH_INTENSITY, **MORPH_CONTINUITY, **MORPH_IRREGULARITY}

ALL_VARS = {**ALL_MORPH, **POP_DENSITY, **NETWORK}

DESERT_THRESHOLD = 400

# ============================================================================
# LOAD AND PREPARE
# ============================================================================
print("=" * 80)
print("MORPHOLOGY VS POPULATION DENSITY: DECOMPOSITION ANALYSIS")
print("=" * 80)

needed = (
    list(AXIS_COLS.values())
    + list(OUTCOME_COLS.values())
    + list(ALL_VARS.values())
    + ["bounds_fid"]
)
needed = sorted(set(needed))

print("\n[1] Loading data...")
df = load_all_cached(columns=needed)
n_raw = len(df)
n_cities_raw = df["bounds_fid"].nunique()
print(f"  Raw: {n_raw:,} streets, {n_cities_raw} cities")

classified, thresholds = classify_octants(df)
print(f"  Classified: {len(classified):,} streets")

# Core columns that must be present
core_cols = (list(THREE_AXES.values()) + list(POP_DENSITY.values())
             + list(NETWORK.values()) + list(OUTCOME_COLS.values()))
classified = classified.dropna(subset=core_cols)
n_analysis = len(classified)
n_cities = classified["bounds_fid"].nunique()
print(f"  Analysis set: {n_analysis:,} streets, {n_cities} cities")

# Subsample
print("\n[2] Spatial subsampling (10% per city)...")
np.random.seed(42)
parts = []
for fid, grp in classified.groupby("bounds_fid"):
    parts.append(grp.sample(frac=0.1, random_state=42) if len(grp) >= 10 else grp)
sub = pd.concat(parts, ignore_index=True)
print(f"  Subsampled: {len(sub):,} streets, {sub['bounds_fid'].nunique()} cities")


# ============================================================================
# HELPER: within-city regression with cluster-robust SEs
# ============================================================================
def within_city_ols(data, outcome_col, predictor_cols, predictor_names):
    """OLS on city-demeaned data with cluster-robust SEs.

    Returns dict with beta, se, t for each predictor + R²_within.
    """
    cols_needed = ["bounds_fid", outcome_col] + list(predictor_cols)
    df_r = data[cols_needed].dropna().copy()
    df_r["log_y"] = np.log(df_r[outcome_col].clip(lower=1))

    # Demean within city
    for col in list(predictor_cols) + ["log_y"]:
        df_r[f"{col}_dm"] = df_r[col] - df_r.groupby("bounds_fid")[col].transform("mean")

    dm_pred = [f"{c}_dm" for c in predictor_cols]

    # Standardise demeaned predictors
    scaler = StandardScaler()
    X = scaler.fit_transform(df_r[dm_pred].values)
    y = df_r["log_y_dm"].values
    n = len(y)
    X_c = np.column_stack([np.ones(n), X])
    k = X_c.shape[1]

    beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
    resid = y - X_c @ beta

    # R² within
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Cluster-robust SEs
    city_ids = df_r["bounds_fid"].values
    unique_cities = np.unique(city_ids)
    G = len(unique_cities)
    XtX_inv = np.linalg.inv(X_c.T @ X_c)
    meat = np.zeros((k, k))
    for cid in unique_cities:
        mask = city_ids == cid
        score = X_c[mask].T @ resid[mask]
        meat += np.outer(score, score)
    correction = G / (G - 1) * (n - 1) / (n - k)
    V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))

    result = {"n": n, "n_cities": G, "R2_within": r2}
    for i, name in enumerate(predictor_names):
        result[f"beta_{name}"] = beta[i + 1]
        result[f"se_{name}"] = se[i + 1]
        result[f"t_{name}"] = beta[i + 1] / se[i + 1] if se[i + 1] > 0 else 0
    return result


def within_r2(data, outcome_col, predictor_cols):
    """Quick within-city R² (no SEs needed)."""
    cols_needed = ["bounds_fid", outcome_col] + list(predictor_cols)
    df_r = data[cols_needed].dropna().copy()
    df_r["log_y"] = np.log(df_r[outcome_col].clip(lower=1))
    for col in list(predictor_cols) + ["log_y"]:
        df_r[f"{col}_dm"] = df_r[col] - df_r.groupby("bounds_fid")[col].transform("mean")
    dm_pred = [f"{c}_dm" for c in predictor_cols]
    X = df_r[dm_pred].values
    y = df_r["log_y_dm"].values
    X_c = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
    ss_res = np.sum((y - X_c @ beta) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


# ============================================================================
# ANALYSIS 1: VARIANCE DECOMPOSITION — WHICH FAMILY EXPLAINS MOST?
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: VARIANCE DECOMPOSITION BY VARIABLE FAMILY")
print("=" * 80)
print("Which family of variables explains the most within-city variance?")
print("All R² values are within-city (city means absorbed).\n")

# Define model families
families = {
    "Pop density only":       list(POP_DENSITY.values()),
    "FSI only":               [THREE_AXES["FSI"]],
    "Frontage only":          [THREE_AXES["frontage"]],
    "MAD only":               [THREE_AXES["MAD"]],
    "3 axes (FSI+FR+MAD)":    list(THREE_AXES.values()),
    "All morphology (9 vars)": [v for v in ALL_MORPH.values() if v in sub.columns],
    "Network only (3 vars)":  list(NETWORK.values()),
    "Pop density + network":  list(POP_DENSITY.values()) + list(NETWORK.values()),
    "3 axes + pop density":   list(THREE_AXES.values()) + list(POP_DENSITY.values()),
    "3 axes + network":       list(THREE_AXES.values()) + list(NETWORK.values()),
    "All morph + network":    [v for v in ALL_MORPH.values() if v in sub.columns] + list(NETWORK.values()),
    "Everything":             [v for v in ALL_VARS.values() if v in sub.columns],
}

for outcome_name, outcome_col in OUTCOME_COLS.items():
    print(f"  --- {outcome_name.upper()} ---")
    print(f"  {'Model':<28} {'R²_within':>10}")
    print(f"  {'-'*40}")
    for fam_name, fam_cols in families.items():
        # filter to columns that exist
        valid_cols = [c for c in fam_cols if c in sub.columns]
        if not valid_cols:
            continue
        r2 = within_r2(sub, outcome_col, valid_cols)
        print(f"  {fam_name:<28} {r2:>10.4f}")
    print()


# ============================================================================
# ANALYSIS 2: MARGINAL CONTRIBUTIONS (SHAPLEY-STYLE)
# ============================================================================
print("=" * 80)
print("ANALYSIS 2: MARGINAL CONTRIBUTIONS OF EACH FAMILY")
print("=" * 80)
print("How much R² does each family ADD to a model already containing the others?\n")

# Four families: morphology (3 axes), pop density, network
fam_A = list(THREE_AXES.values())       # morphology
fam_B = list(POP_DENSITY.values())      # pop density
fam_C = list(NETWORK.values())          # network

for outcome_name, outcome_col in OUTCOME_COLS.items():
    r2_all = within_r2(sub, outcome_col, fam_A + fam_B + fam_C)
    r2_no_morph = within_r2(sub, outcome_col, fam_B + fam_C)
    r2_no_popd = within_r2(sub, outcome_col, fam_A + fam_C)
    r2_no_net = within_r2(sub, outcome_col, fam_A + fam_B)

    # Also: within morphology, marginal of each axis
    r2_morph_all = within_r2(sub, outcome_col, fam_A)
    r2_morph_no_fsi = within_r2(sub, outcome_col, [THREE_AXES["frontage"], THREE_AXES["MAD"]])
    r2_morph_no_fr = within_r2(sub, outcome_col, [THREE_AXES["FSI"], THREE_AXES["MAD"]])
    r2_morph_no_mad = within_r2(sub, outcome_col, [THREE_AXES["FSI"], THREE_AXES["frontage"]])

    print(f"  --- {outcome_name.upper()} (full R² = {r2_all:.4f}) ---")
    print(f"  {'Family':<25} {'Marginal ΔR²':>12} {'% of full':>10}")
    print(f"  {'-'*50}")
    delta_morph = r2_all - r2_no_morph
    delta_popd = r2_all - r2_no_popd
    delta_net = r2_all - r2_no_net
    print(f"  {'Morphology (3 axes)':<25} {delta_morph:>12.4f} {delta_morph/r2_all*100:>9.1f}%")
    print(f"  {'Population density':<25} {delta_popd:>12.4f} {delta_popd/r2_all*100:>9.1f}%")
    print(f"  {'Network structure':<25} {delta_net:>12.4f} {delta_net/r2_all*100:>9.1f}%")
    print(f"    Within morphology:")
    delta_fsi = r2_morph_all - r2_morph_no_fsi
    delta_fr = r2_morph_all - r2_morph_no_fr
    delta_mad = r2_morph_all - r2_morph_no_mad
    print(f"    {'+ FSI (intensity)':<23} {delta_fsi:>12.4f} {delta_fsi/r2_morph_all*100 if r2_morph_all > 0 else 0:>9.1f}%")
    print(f"    {'+ Frontage (continuity)':<23} {delta_fr:>12.4f} {delta_fr/r2_morph_all*100 if r2_morph_all > 0 else 0:>9.1f}%")
    print(f"    {'+ MAD (irregularity)':<23} {delta_mad:>12.4f} {delta_mad/r2_morph_all*100 if r2_morph_all > 0 else 0:>9.1f}%")
    print()


# ============================================================================
# ANALYSIS 3: FULL REGRESSION — ALL VARIABLES, STANDARDISED COEFFICIENTS
# ============================================================================
print("=" * 80)
print("ANALYSIS 3: FULL REGRESSION (standardised betas, city FE, cluster-robust SEs)")
print("=" * 80)
print("All predictors entered simultaneously. Coefficients are per 1 SD.\n")

all_pred_cols = list(THREE_AXES.values()) + list(POP_DENSITY.values()) + list(NETWORK.values())
all_pred_names = list(THREE_AXES.keys()) + list(POP_DENSITY.keys()) + list(NETWORK.keys())

header = f"  {'Outcome':<12} {'R²w':>5} " + " ".join([f"{n:>10}" for n in all_pred_names])
print(header)
print("  " + "-" * len(header))

for outcome_name, outcome_col in OUTCOME_COLS.items():
    r = within_city_ols(sub, outcome_col, all_pred_cols, all_pred_names)
    vals = []
    for name in all_pred_names:
        b = r[f"beta_{name}"]
        sig = "*" if abs(r[f"t_{name}"]) > 1.96 else " "
        vals.append(f"{b:>+9.4f}{sig}")
    print(f"  {outcome_name:<12} {r['R2_within']:>5.3f} " + " ".join(vals))

print("\n  Negative = higher value → shorter distance (better access)")
print("  * = significant at p<0.05 with city-clustered SEs")
print("  Note: FSI, frontage, MAD are ALL morphological. Pop density is demographic.")


# ============================================================================
# ANALYSIS 4: WITHIN-CITY MATCHED PAIRS — MORPHOLOGY EFFECT
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: WITHIN-CITY MATCHED PAIRS")
print("=" * 80)
print("CEM: within each city × pop_density_quintile, compare Attached vs Freestanding.")
print("This isolates the Continuity effect while holding city AND pop density constant.\n")

def within_city_cem(data, outcome_cols):
    """CEM: within city × pop density quintile, compare Attached vs Freestanding."""
    pred_cols = [POP_DENSITY["pop_density"], THREE_AXES["frontage"]]
    df_m = data[["bounds_fid"] + pred_cols + list(outcome_cols.values())].dropna().copy()

    df_m["attached"] = (df_m[THREE_AXES["frontage"]] >= 0.75).astype(int)

    # Pop density quintiles within each city
    df_m["dq"] = df_m.groupby("bounds_fid")[POP_DENSITY["pop_density"]].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else 0
    )

    results = {}
    for name, col in outcome_cols.items():
        cell = df_m.groupby(["bounds_fid", "dq", "attached"])[col].agg(["mean", "count"]).reset_index()
        cell.columns = ["bounds_fid", "dq", "attached", "mean_dist", "count"]
        piv = cell.pivot_table(index=["bounds_fid", "dq"], columns="attached",
                               values=["mean_dist", "count"])
        if ("mean_dist", 0) not in piv.columns or ("mean_dist", 1) not in piv.columns:
            results[name] = None
            continue
        both = piv.dropna(subset=[("mean_dist", 0), ("mean_dist", 1)])
        if len(both) == 0:
            results[name] = None
            continue

        # Attached(1) - Freestanding(0): negative means Attached is closer
        diffs = both[("mean_dist", 1)].values - both[("mean_dist", 0)].values
        weights = both[("count", 0)].clip(upper=both[("count", 1)]).values
        weighted_effect = np.average(diffs, weights=weights)

        # Per-city effects
        city_effects = []
        for cid in both.index.get_level_values("bounds_fid").unique():
            try:
                cc = both.loc[cid]
                if isinstance(cc, pd.Series):
                    city_effects.append(cc[("mean_dist", 1)] - cc[("mean_dist", 0)])
                else:
                    w = cc[("count", 0)].clip(upper=cc[("count", 1)]).values
                    d = cc[("mean_dist", 1)].values - cc[("mean_dist", 0)].values
                    city_effects.append(np.average(d, weights=w))
            except Exception:
                pass
        city_effects = np.array(city_effects)

        results[name] = {
            "effect_m": weighted_effect,
            "median_city": np.median(city_effects),
            "pct_negative": np.mean(city_effects < 0) * 100,
            "n_cells": len(both),
            "n_cities": len(city_effects),
            "p25": np.percentile(city_effects, 25),
            "p75": np.percentile(city_effects, 75),
        }
    return results

cem = within_city_cem(classified, OUTCOME_COLS)

print(f"  {'Outcome':<12} {'Effect(m)':>10} {'Median':>8} {'P25':>8} {'P75':>8} {'%neg':>7} {'Cities':>7}")
print(f"  {'-'*65}")
for name, r in cem.items():
    if r:
        print(f"  {name:<12} {r['effect_m']:>+10.1f} {r['median_city']:>+8.1f} "
              f"{r['p25']:>+8.1f} {r['p75']:>+8.1f} {r['pct_negative']:>6.1f}% {r['n_cities']:>7}")

print("\n  Negative = Attached streets are closer (better commercial access)")
print("  Positive = Attached streets are farther (worse green access)")
print("  Matched on: same city, same pop density quintile")


# ============================================================================
# ANALYSIS 5: DOSE-RESPONSE — FRONTAGE GRADIENT WITHIN FSI QUINTILES
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 5: DOSE-RESPONSE (frontage × FSI quintiles)")
print("=" * 80)
print("Within each FSI quintile, how does retail distance vary with frontage?\n")

dr = sub[[OUTCOME_COLS["retail"], THREE_AXES["FSI"], THREE_AXES["frontage"]]].dropna().copy()
dr["fsi_q"] = pd.qcut(dr[THREE_AXES["FSI"]], 5, labels=False, duplicates="drop") + 1

print(f"  {'FSI Q':<8} {'FR<0.1':>10} {'0.1-0.3':>10} {'0.3-0.5':>10} {'0.5-0.7':>10} "
      f"{'0.7-0.9':>10} {'FR>0.9':>10} {'Ratio':>8}")
print(f"  {'-'*75}")

for q in range(1, 6):
    qd = dr[dr["fsi_q"] == q]
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    vals = []
    for lo, hi in bins:
        m = qd[(qd[THREE_AXES["frontage"]] >= lo) & (qd[THREE_AXES["frontage"]] < hi)][OUTCOME_COLS["retail"]].median()
        vals.append(m)
    ratio = vals[0] / vals[-1] if vals[-1] > 0 and pd.notna(vals[0]) and pd.notna(vals[-1]) else np.nan
    vstr = " ".join([f"{v:>10.0f}" if pd.notna(v) else f"{'--':>10}" for v in vals])
    print(f"  Q{q:<7} {vstr} {ratio:>8.2f}x")

# Same for green space
print(f"\n  Green-space distance:")
drg = sub[[OUTCOME_COLS["green"], THREE_AXES["FSI"], THREE_AXES["frontage"]]].dropna().copy()
drg["fsi_q"] = pd.qcut(drg[THREE_AXES["FSI"]], 5, labels=False, duplicates="drop") + 1

print(f"  {'FSI Q':<8} {'FR<0.1':>10} {'0.1-0.3':>10} {'0.3-0.5':>10} {'0.5-0.7':>10} "
      f"{'0.7-0.9':>10} {'FR>0.9':>10} {'Ratio':>8}")
print(f"  {'-'*75}")
for q in range(1, 6):
    qd = drg[drg["fsi_q"] == q]
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    vals = []
    for lo, hi in bins:
        m = qd[(qd[THREE_AXES["frontage"]] >= lo) & (qd[THREE_AXES["frontage"]] < hi)][OUTCOME_COLS["green"]].median()
        vals.append(m)
    ratio = vals[0] / vals[-1] if vals[-1] > 0 and pd.notna(vals[0]) and pd.notna(vals[-1]) else np.nan
    vstr = " ".join([f"{v:>10.0f}" if pd.notna(v) else f"{'--':>10}" for v in vals])
    print(f"  Q{q:<7} {vstr} {ratio:>8.2f}x")


# ============================================================================
# ANALYSIS 6: DOSE-RESPONSE — POP DENSITY GRADIENT WITHIN MORPHOLOGY BINS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 6: POP DENSITY GRADIENT WITHIN MORPHOLOGICAL TYPE")
print("=" * 80)
print("Within each octant, how much does pop density improve access?\n")

oct_data = classified[["octant", POP_DENSITY["pop_density"], OUTCOME_COLS["retail"]]].dropna()
print(f"  {'Octant':<8} {'Pop Q1':>10} {'Pop Q3':>10} {'Pop Q5':>10} {'Q1/Q5':>8} {'N':>10}")
print(f"  {'-'*55}")

for octant in ["HHH", "HHL", "HLH", "HLL", "LHH", "LHL", "LLH", "LLL"]:
    od = oct_data[oct_data["octant"] == octant].copy()
    if len(od) < 100:
        continue
    od["pq"] = pd.qcut(od[POP_DENSITY["pop_density"]], 5, labels=False, duplicates="drop") + 1
    q1 = od[od["pq"] == 1][OUTCOME_COLS["retail"]].median()
    q3 = od[od["pq"] == 3][OUTCOME_COLS["retail"]].median()
    q5 = od[od["pq"] == 5][OUTCOME_COLS["retail"]].median()
    ratio = q1 / q5 if q5 > 0 else np.nan
    print(f"  {octant:<8} {q1:>10.0f} {q3:>10.0f} {q5:>10.0f} {ratio:>8.2f}x {len(od):>10,}")

print("\n  This shows how much pop density matters WITHIN a fixed morphological type.")
print("  If ratio is close to 1, pop density adds little once form is known.")


# ============================================================================
# ANALYSIS 7: GREEN SPACE AS DESIGN CHOICE
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 7: GREEN SPACE AS DESIGN CHOICE")
print("=" * 80)
print("Which cities achieve short distances to BOTH services and green space?")
print("This demonstrates the trade-off is a policy choice, not inevitable.\n")

city_med = classified.groupby("bounds_fid").agg(
    retail_med=(OUTCOME_COLS["retail"], "median"),
    green_med=(OUTCOME_COLS["green"], "median"),
    trees_med=(OUTCOME_COLS["trees"], "median"),
    pop_dens=(POP_DENSITY["pop_density"], "median"),
    fsi_med=(THREE_AXES["FSI"], "median"),
    frontage_med=(THREE_AXES["frontage"], "median"),
    n_streets=(THREE_AXES["FSI"], "count"),
).dropna()

# Continental medians
med_retail = city_med["retail_med"].median()
med_green = city_med["green_med"].median()
med_trees = city_med["trees_med"].median()

print(f"  Continental medians: retail={med_retail:.0f}m, green={med_green:.0f}m, trees={med_trees:.0f}m")
print(f"  N cities: {len(city_med)}")

# Quadrant analysis
q_good_both = city_med[(city_med["retail_med"] < med_retail) & (city_med["green_med"] < med_green)]
q_good_retail = city_med[(city_med["retail_med"] < med_retail) & (city_med["green_med"] >= med_green)]
q_good_green = city_med[(city_med["retail_med"] >= med_retail) & (city_med["green_med"] < med_green)]
q_poor_both = city_med[(city_med["retail_med"] >= med_retail) & (city_med["green_med"] >= med_green)]

print(f"\n  Quadrant distribution (below-median = better access):")
print(f"    Good retail + Good green:  {len(q_good_both):>4} cities ({len(q_good_both)/len(city_med)*100:.1f}%)")
print(f"    Good retail + Poor green:  {len(q_good_retail):>4} cities ({len(q_good_retail)/len(city_med)*100:.1f}%)")
print(f"    Poor retail + Good green:  {len(q_good_green):>4} cities ({len(q_good_green)/len(city_med)*100:.1f}%)")
print(f"    Poor retail + Poor green:  {len(q_poor_both):>4} cities ({len(q_poor_both)/len(city_med)*100:.1f}%)")

# Correlation
r_corr, p_corr = stats.spearmanr(city_med["retail_med"], city_med["green_med"])
print(f"\n  Spearman correlation (retail × green): rho={r_corr:.3f}, p={p_corr:.4f}")
print(f"  R² = {r_corr**2:.4f}")
print(f"  → The two are {'not ' if abs(r_corr) < 0.2 else ''}strongly coupled.")

# Profile of "good both" cities
print(f"\n  Profile of cities achieving BOTH short retail AND short green distances:")
print(f"  {'Metric':<20} {'Good both':>12} {'Rest':>12} {'Difference':>12}")
print(f"  {'-'*60}")
rest = city_med[~city_med.index.isin(q_good_both.index)]
for metric, label in [("pop_dens", "Pop density"), ("fsi_med", "Median FSI"),
                       ("frontage_med", "Median frontage"), ("retail_med", "Retail dist (m)"),
                       ("green_med", "Green dist (m)"), ("trees_med", "Trees dist (m)")]:
    gb = q_good_both[metric].median()
    rt = rest[metric].median()
    print(f"  {label:<20} {gb:>12.1f} {rt:>12.1f} {gb - rt:>+12.1f}")

# Top 20 "good both" cities by composite score
q_good_both = q_good_both.copy()
q_good_both["composite"] = (
    q_good_both["retail_med"] / med_retail + q_good_both["green_med"] / med_green
) / 2
top20 = q_good_both.nsmallest(20, "composite")

# Try to load city names
try:
    import geopandas as gpd
    from atlas_common import BOUNDARIES_PATH
    bounds = gpd.read_file(BOUNDARIES_PATH, columns=["bounds_fid", "UC_NM_MN", "CNTR_ID"])
    top20 = top20.merge(bounds[["bounds_fid", "UC_NM_MN", "CNTR_ID"]], left_index=True, right_on="bounds_fid", how="left")
    print(f"\n  Top 20 cities achieving both short retail AND short green distances:")
    print(f"  {'City':<25} {'Country':>8} {'Retail':>8} {'Green':>8} {'Trees':>8} {'FSI':>6} {'Front':>6} {'PopD':>8}")
    print(f"  {'-'*85}")
    for _, row in top20.iterrows():
        print(f"  {str(row.get('UC_NM_MN', '?')):<25} {str(row.get('CNTR_ID', '?')):>8} "
              f"{row['retail_med']:>8.0f} {row['green_med']:>8.0f} {row['trees_med']:>8.0f} "
              f"{row['fsi_med']:>6.2f} {row['frontage_med']:>6.2f} {row['pop_dens']:>8.0f}")
except Exception as e:
    print(f"\n  (Could not load city names: {e})")
    print(f"  Top 20 bounds_fid values: {list(top20.index[:20])}")

# Also: which countries dominate the "good both" quadrant?
try:
    gb_countries = q_good_both.merge(bounds[["bounds_fid", "CNTR_ID"]], left_index=True, right_on="bounds_fid")
    rest_countries = rest.merge(bounds[["bounds_fid", "CNTR_ID"]], left_index=True, right_on="bounds_fid")
    print(f"\n  Country representation in 'good both' quadrant:")
    country_counts = gb_countries["CNTR_ID"].value_counts().head(15)
    total_per_country = bounds["CNTR_ID"].value_counts()
    print(f"  {'Country':>8} {'N good':>7} {'N total':>8} {'%':>6}")
    print(f"  {'-'*32}")
    for country, count in country_counts.items():
        tot = total_per_country.get(country, 0)
        print(f"  {country:>8} {count:>7} {tot:>8} {count/tot*100 if tot > 0 else 0:>5.0f}%")
except Exception:
    pass


# ============================================================================
# ANALYSIS 8: ICC — HOW MUCH IS BETWEEN-CITY VS WITHIN-CITY?
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 8: ICC (between-city vs within-city variance)")
print("=" * 80)

for outcome_name, outcome_col in OUTCOME_COLS.items():
    vals = sub[["bounds_fid", outcome_col]].dropna()
    vals["log_y"] = np.log(vals[outcome_col].clip(lower=1))
    city_means = vals.groupby("bounds_fid")["log_y"].transform("mean")
    var_total = vals["log_y"].var()
    var_between = city_means.var()
    icc = var_between / var_total if var_total > 0 else 0
    print(f"  {outcome_name:<12} ICC = {icc:.3f}  ({icc*100:.1f}% of variance is between cities)")

print("\n  Low ICC = most variation is WITHIN cities → morphological variables can explain it")
print("  High ICC = much variation between cities → city-level factors dominate")


# ============================================================================
# SYNTHESIS
# ============================================================================
print("\n" + "=" * 80)
print("SYNTHESIS")
print("=" * 80)
print("""
KEY COMPARISONS TO READ:

1. ANALYSIS 1: Does pop density alone explain less than morphology alone?
   → Compare "Pop density only" R² vs "3 axes" R².

2. ANALYSIS 2: What's the marginal contribution of each family?
   → Morphology ΔR² vs pop density ΔR² vs network ΔR².
   → Within morphology: FSI vs frontage vs MAD.

3. ANALYSIS 3: Per SD, which variables predict access most strongly?
   → Are morphological betas larger than pop density beta?
   → FSI, frontage, and MAD are ALL morphological — sum their contributions.

4. ANALYSIS 4: At same pop density in same city, does Continuity matter?
   → % of cities showing consistent direction.

5. ANALYSIS 5: Does the frontage gradient hold across all FSI quintiles?
   → If yes, Continuity isn't just a proxy for Intensity.

6. ANALYSIS 6: Does pop density add much within a morphological type?
   → Low ratios = morphology already captured what pop density would predict.

7. ANALYSIS 7: Which cities achieve BOTH good retail AND good green?
   → These prove it's a design choice, not a trade-off.

8. ANALYSIS 8: How much variance is within-city (where morphology operates)?
""")

print("Analysis complete.")
