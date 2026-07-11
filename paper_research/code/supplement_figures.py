"""
Supplementary figures and tables for the analytical appendix.

Generates:
  S1 — Variance decomposition table (LaTeX)
  S2 — Matched-pair table (LaTeX)
  S3 — Dose-response figure (retail and green, by FSI quintile × frontage bin)

Uses the same data pipeline as analysis_morphology_vs_density.py.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
    GREY,
    OUTPUT_DIR,
    apply_atlas_style,
    classify_octants,
    load_all_cached,
)

apply_atlas_style()

# ============================================================================
# CONFIG (mirrors analysis_morphology_vs_density.py)
# ============================================================================
OUTCOME_COLS = {
    "Retail": "cc_retail_nearest_max_1600",
    "Eat \\& drink": "cc_eat_and_drink_nearest_max_1600",
    "Education": "cc_education_nearest_max_1600",
    "Health": "cc_health_and_medical_nearest_max_1600",
    "Green space": "cc_green_nearest_max_1600",
    "Trees": "cc_trees_nearest_max_1600",
}

THREE_AXES = {
    "FSI": "cc_block_far_median_400_wt",
    "frontage": "frontage_max",
    "MAD": "cc_orientation_mad_400_wt",
}

POP_DENSITY = {"pop_density": "density"}

NETWORK = {
    "closeness_800": "cc_beta_800",
    "net_density_400": "cc_density_400",
    "betweenness_800": "cc_betweenness_beta_800",
}

SUPP_DIR = OUTPUT_DIR / "supplement"
SUPP_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
needed = (
    list(AXIS_COLS.values())
    + list(OUTCOME_COLS.values())
    + list(THREE_AXES.values())
    + list(POP_DENSITY.values())
    + list(NETWORK.values())
    + ["bounds_fid"]
)
needed = sorted(set(needed))

df = load_all_cached(columns=needed)
classified, _ = classify_octants(df)

core_cols = (
    list(THREE_AXES.values())
    + list(POP_DENSITY.values())
    + list(NETWORK.values())
    + list(OUTCOME_COLS.values())
)
classified = classified.dropna(subset=core_cols)
n_analysis = len(classified)
n_cities = classified["bounds_fid"].nunique()
print(f"  Analysis set: {n_analysis:,} streets, {n_cities} cities")

# Subsample for regression
np.random.seed(42)
parts = []
for fid, grp in classified.groupby("bounds_fid"):
    parts.append(grp.sample(frac=0.1, random_state=42) if len(grp) >= 10 else grp)
sub = pd.concat(parts, ignore_index=True)
print(f"  Subsampled: {len(sub):,} streets")


# ============================================================================
# HELPER: within-city R²
# ============================================================================
def within_r2(data, outcome_col, predictor_cols):
    """Quick within-city R²."""
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
# S1: VARIANCE DECOMPOSITION TABLE
# ============================================================================
print("\nGenerating S1: Variance decomposition table...")

families = {
    "Pop.~density alone": list(POP_DENSITY.values()),
    "3 morph.~axes alone": list(THREE_AXES.values()),
    "Network alone": list(NETWORK.values()),
    "3 axes + network": list(THREE_AXES.values()) + list(NETWORK.values()),
    "Everything": list(THREE_AXES.values()) + list(POP_DENSITY.values()) + list(NETWORK.values()),
}

rows = []
for fam_name, fam_cols in families.items():
    valid_cols = [c for c in fam_cols if c in sub.columns]
    row = {"Model": fam_name}
    for out_name, out_col in OUTCOME_COLS.items():
        row[out_name] = within_r2(sub, out_col, valid_cols)
    rows.append(row)

decomp_df = pd.DataFrame(rows)

# Write LaTeX table
with open(SUPP_DIR / "table_variance_decomposition.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("  \\centering\n")
    f.write("  \\caption{Within-city $R^2$ by predictor family. All models use city-demeaned\n")
    f.write("    (within-city) estimation with 10\\% spatial subsampling. Population density's\n")
    f.write("    marginal $\\Delta R^2$ when added to the full model is 0.000 for every outcome.}\n")
    f.write("  \\label{tab:decomposition}\n")
    f.write("  \\small\n")
    out_names = list(OUTCOME_COLS.keys())
    ncols = len(out_names) + 1
    f.write("  \\begin{tabular}{l" + "r" * len(out_names) + "}\n")
    f.write("    \\toprule\n")
    f.write("    Model & " + " & ".join(out_names) + " \\\\\n")
    f.write("    \\midrule\n")
    for _, row in decomp_df.iterrows():
        vals = " & ".join([f"{row[n]:.3f}" for n in out_names])
        f.write(f"    {row['Model']} & {vals} \\\\\n")
    f.write("    \\bottomrule\n")
    f.write("  \\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"  Saved {SUPP_DIR / 'table_variance_decomposition.tex'}")

# Also write marginal contributions table
print("\nGenerating S1b: Marginal contributions table...")
fam_A = list(THREE_AXES.values())
fam_B = list(POP_DENSITY.values())
fam_C = list(NETWORK.values())

marginal_rows = []
for out_name, out_col in OUTCOME_COLS.items():
    r2_all = within_r2(sub, out_col, fam_A + fam_B + fam_C)
    r2_no_morph = within_r2(sub, out_col, fam_B + fam_C)
    r2_no_popd = within_r2(sub, out_col, fam_A + fam_C)
    r2_no_net = within_r2(sub, out_col, fam_A + fam_B)

    # Within morphology
    r2_morph = within_r2(sub, out_col, fam_A)
    r2_no_fsi = within_r2(sub, out_col, [THREE_AXES["frontage"], THREE_AXES["MAD"]])
    r2_no_fr = within_r2(sub, out_col, [THREE_AXES["FSI"], THREE_AXES["MAD"]])
    r2_no_mad = within_r2(sub, out_col, [THREE_AXES["FSI"], THREE_AXES["frontage"]])

    marginal_rows.append({
        "Outcome": out_name,
        "Full R²": r2_all,
        "ΔR² morph": r2_all - r2_no_morph,
        "ΔR² pop": r2_all - r2_no_popd,
        "ΔR² net": r2_all - r2_no_net,
        "ΔR² FSI": r2_morph - r2_no_fsi,
        "ΔR² front": r2_morph - r2_no_fr,
        "ΔR² MAD": r2_morph - r2_no_mad,
    })

marg_df = pd.DataFrame(marginal_rows)

with open(SUPP_DIR / "table_marginal_contributions.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("  \\centering\n")
    f.write("  \\caption{Marginal $\\Delta R^2$ of each predictor family and each morphological axis.\n")
    f.write("    Within-morphology marginals show the unique contribution of each axis\n")
    f.write("    when added to the other two.}\n")
    f.write("  \\label{tab:marginal}\n")
    f.write("  \\small\n")
    f.write("  \\begin{tabular}{lrrrrrrr}\n")
    f.write("    \\toprule\n")
    f.write("    Outcome & Full $R^2$ & $\\Delta$ Morph & $\\Delta$ Pop.~dens & $\\Delta$ Network & $\\Delta$ FSI & $\\Delta$ Frontage & $\\Delta$ MAD \\\\\n")
    f.write("    \\midrule\n")
    for _, row in marg_df.iterrows():
        f.write(f"    {row['Outcome']} & {row['Full R²']:.3f} & {row['ΔR² morph']:.3f} & {row['ΔR² pop']:.4f} & {row['ΔR² net']:.3f} & {row['ΔR² FSI']:.3f} & {row['ΔR² front']:.3f} & {row['ΔR² MAD']:.3f} \\\\\n")
    f.write("    \\bottomrule\n")
    f.write("  \\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"  Saved {SUPP_DIR / 'table_marginal_contributions.tex'}")

# ============================================================================
# S2: MATCHED-PAIR TABLE
# ============================================================================
print("\nGenerating S2: Matched-pair table...")


def within_city_cem(data, outcome_cols):
    """CEM: within city × pop density quintile, compare Attached vs Freestanding."""
    pred_cols = [POP_DENSITY["pop_density"], THREE_AXES["frontage"]]
    df_m = data[["bounds_fid"] + pred_cols + list(outcome_cols.values())].dropna().copy()
    df_m["attached"] = (df_m[THREE_AXES["frontage"]] >= 0.75).astype(int)
    df_m["dq"] = df_m.groupby("bounds_fid")[POP_DENSITY["pop_density"]].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else 0
    )
    results = {}
    for name, col in outcome_cols.items():
        cell = df_m.groupby(["bounds_fid", "dq", "attached"])[col].agg(["mean", "count"]).reset_index()
        cell.columns = ["bounds_fid", "dq", "attached", "mean_dist", "count"]
        piv = cell.pivot_table(
            index=["bounds_fid", "dq"], columns="attached", values=["mean_dist", "count"]
        )
        if ("mean_dist", 0) not in piv.columns or ("mean_dist", 1) not in piv.columns:
            results[name] = None
            continue
        both = piv.dropna(subset=[("mean_dist", 0), ("mean_dist", 1)])
        if len(both) == 0:
            results[name] = None
            continue
        diffs = both[("mean_dist", 1)].values - both[("mean_dist", 0)].values
        weights = both[("count", 0)].clip(upper=both[("count", 1)]).values
        weighted_effect = np.average(diffs, weights=weights)
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
            "n_cities": len(city_effects),
            "p25": np.percentile(city_effects, 25),
            "p75": np.percentile(city_effects, 75),
        }
    return results


cem = within_city_cem(classified, OUTCOME_COLS)

with open(SUPP_DIR / "table_matched_pairs.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("  \\centering\n")
    f.write("  \\caption{Within-city matched-pair comparison: Attached vs Freestanding streets,\n")
    f.write("    matched on city and population-density quintile. Effect is the difference\n")
    f.write("    in mean distance (Attached minus Freestanding); negative values indicate\n")
    f.write("    Attached streets are closer. \\%~cities shows the proportion of cities\n")
    f.write("    where the effect is in the indicated direction.}\n")
    f.write("  \\label{tab:matched}\n")
    f.write("  \\small\n")
    f.write("  \\begin{tabular}{lrrrrrr}\n")
    f.write("    \\toprule\n")
    f.write("    Outcome & Effect (m) & Median & P25 & P75 & \\% cities & $N$ cities \\\\\n")
    f.write("    \\midrule\n")
    for name, r in cem.items():
        if r is None:
            continue
        direction = "closer" if r["effect_m"] < 0 else "farther"
        pct = r["pct_negative"] if r["effect_m"] < 0 else (100 - r["pct_negative"])
        f.write(
            f"    {name} & {r['effect_m']:+.1f} & {r['median_city']:+.1f} & "
            f"{r['p25']:+.1f} & {r['p75']:+.1f} & {pct:.1f}\\% {direction} & {r['n_cities']} \\\\\n"
        )
    f.write("    \\bottomrule\n")
    f.write("  \\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"  Saved {SUPP_DIR / 'table_matched_pairs.tex'}")

# ============================================================================
# MACRO WRITING — emit key analytical results as LaTeX macros
# ============================================================================
print("\nWriting analytical result macros...")

MACRO_SUPP = Path(__file__).resolve().parent.parent / "atlas_macros_supplement.tex"

def _mac(name, value, fmt=".0f"):
    return f"\\newcommand{{\\{name}}}{{{value:{fmt}}}}\n"

COMMERCIAL = ["Retail", "Eat \\& drink", "Education", "Health"]

with open(MACRO_SUPP, "w") as mf:
    mf.write("% Auto-generated by supplement_figures.py — do not edit by hand.\n")
    mf.write("% Re-run: cd paper_research/code && python supplement_figures.py\n\n")

    mf.write("% ── Variance decomposition ───────────────────────────────────────\n")

    # 3-axis morphology R² range across commercial outcomes
    morph_r2s = [
        decomp_df.loc[decomp_df["Model"] == "3 morph.~axes alone", o].values[0] * 100
        for o in COMMERCIAL if o in decomp_df.columns
    ]
    mf.write(_mac("rSqMorphMin", min(morph_r2s)))
    mf.write(_mac("rSqMorphMax", max(morph_r2s)))

    # 3 axes + network R² range across commercial outcomes
    morphnet_r2s = [
        decomp_df.loc[decomp_df["Model"] == "3 axes + network", o].values[0] * 100
        for o in COMMERCIAL if o in decomp_df.columns
    ]
    mf.write(_mac("rSqMorphNetMin", min(morphnet_r2s)))
    mf.write(_mac("rSqMorphNetMax", max(morphnet_r2s)))

    # Population density alone (max across all outcomes)
    popd_r2s = [
        decomp_df.loc[decomp_df["Model"] == "Pop.~density alone", o].values[0] * 100
        for o in OUTCOME_COLS if o in decomp_df.columns
    ]
    mf.write(_mac("rSqPopDensMax", max(popd_r2s), ".2f"))

    # Network alone R² range across commercial outcomes (cited as 10–14%)
    net_r2s = [
        decomp_df.loc[decomp_df["Model"] == "Network alone", o].values[0] * 100
        for o in COMMERCIAL if o in decomp_df.columns
    ]
    mf.write(_mac("mNetworkRsqMin", min(net_r2s)))
    mf.write(_mac("mNetworkRsqMax", max(net_r2s)))

    mf.write("\n% ── Marginal contributions within morphology ─────────────────────\n")

    comm_rows = marg_df[marg_df["Outcome"].isin(COMMERCIAL)]
    fsi_shares, cont_shares, ratios = [], [], []
    for _, row in comm_rows.iterrows():
        r2_morph = decomp_df.loc[decomp_df["Model"] == "3 morph.~axes alone",
                                  row["Outcome"]].values[0]
        if r2_morph <= 0:
            continue
        fsi_shares.append(row["ΔR² FSI"] / r2_morph * 100)
        cont_shares.append(row["ΔR² front"] / r2_morph * 100)
        if row["ΔR² front"] > 0:
            ratios.append(row["ΔR² FSI"] / row["ΔR² front"])

    mf.write(_mac("mFSIshareMin", min(fsi_shares)))
    mf.write(_mac("mFSIshareMax", max(fsi_shares)))
    mf.write(_mac("mContshareMin", min(cont_shares)))
    mf.write(_mac("mContshareMax", max(cont_shares)))
    if ratios:
        mf.write(_mac("mFSIContRatioMin", min(ratios)))
        mf.write(_mac("mFSIContRatioMax", max(ratios)))

    green_row = marg_df[marg_df["Outcome"] == "Green space"]
    if len(green_row):
        gr = green_row.iloc[0]
        r2_morph_g = decomp_df.loc[decomp_df["Model"] == "3 morph.~axes alone",
                                    "Green space"].values[0]
        if r2_morph_g > 0:
            mf.write(_mac("mContGreenShare", gr["ΔR² front"] / r2_morph_g * 100))
            mf.write(_mac("mFSIGreenShare",  gr["ΔR² FSI"]   / r2_morph_g * 100))

    mf.write("\n% ── Matched-pair comparisons ──────────────────────────────────────\n")

    comm_cem = {k: v for k, v in cem.items() if k in COMMERCIAL and v is not None}
    if comm_cem:
        effects  = [abs(r["effect_m"])   for r in comm_cem.values()]
        dir_pcts = [r["pct_negative"]    for r in comm_cem.values()]
        mf.write(_mac("matchedGainMin", min(effects)))
        mf.write(_mac("matchedGainMax", max(effects)))
        mf.write(_mac("matchedDirMin",  min(dir_pcts)))
        mf.write(_mac("matchedDirMax",  max(dir_pcts)))

    green_cem = cem.get("Green space")
    if green_cem is not None:
        mf.write(_mac("matchedGreenLoss", abs(green_cem["effect_m"])))

    mf.write("\n% ── Analysis set size (streets/cities with complete predictors) ──\n")
    mf.write(_mac("nAnalysisCities", n_cities))
    mf.write(_mac("nAnalysisMStreets", n_analysis / 1e6, ".1f"))

print(f"  Saved {MACRO_SUPP}")

# ============================================================================
# S3: DOSE-RESPONSE FIGURE
# ============================================================================
print("\nGenerating S3: Dose-response figure...")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), facecolor=BG)

fr_bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
fr_labels = ["<0.1", "0.1–0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", ">0.9"]
fr_mids = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]

q_colors = ["#d73027", "#fc8d59", "#fee08b", "#91bfdb", "#4575b4"]

for panel_idx, (outcome_label, outcome_col, ylabel) in enumerate([
    ("Retail", "cc_retail_nearest_max_1600", "Median retail distance (m)"),
    ("Green space", "cc_green_nearest_max_1600", "Median green-space distance (m)"),
]):
    ax = axes[panel_idx]
    ax.set_facecolor(BG)

    dr = sub[[outcome_col, THREE_AXES["FSI"], THREE_AXES["frontage"]]].dropna().copy()
    dr["fsi_q"] = pd.qcut(dr[THREE_AXES["FSI"]], 5, labels=False, duplicates="drop") + 1

    for q in range(1, 6):
        qd = dr[dr["fsi_q"] == q]
        vals = []
        for lo, hi in fr_bins:
            m = qd[
                (qd[THREE_AXES["frontage"]] >= lo) & (qd[THREE_AXES["frontage"]] < hi)
            ][outcome_col].median()
            vals.append(m)

        valid = [(x, y) for x, y in zip(fr_mids, vals) if pd.notna(y)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "-o", color=q_colors[q - 1], markersize=4, linewidth=1.5,
                    label=f"FSI Q{q}", zorder=3)

    ax.set_xlabel("Frontage ratio", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(outcome_label, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.set_xticks(fr_mids)
    ax.set_xticklabels(fr_labels, fontsize=6, rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if panel_idx == 0:
        ax.legend(fontsize=6, frameon=False, loc="upper right")

fig.tight_layout(pad=1.0)
out_path = SUPP_DIR / "fig_dose_response.pdf"
fig.savefig(out_path, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_path}")

print("\nDone.")
