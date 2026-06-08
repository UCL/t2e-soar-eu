#!/usr/bin/env python3
"""Generate publication figures for the POI source-substitution analysis.

Reads cached CSV outputs (produced by `s02`–`s04`) and emits:

Main paper:
  - fig_pointa_agreement_panels.pdf
  - fig_pointa_support_heatmap.pdf
  - table_pointa_minima_clean.tex

Supplementary:
  - fig_pointa_city_variability.pdf
  - fig_pointa_agreement_panels_wt.pdf  (distance-weighted counts)
  - fig_pointa_support_heatmap_wt.pdf   (distance-weighted counts)
  - fig_pointa_city_variability_wt.pdf  (distance-weighted counts)
  - fig_pointa_cross_city_scatter_wt.pdf (distance-weighted counts)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CATEGORY_NAMES, FIG_DIR, OUTPUT_DIR, TABLE_DIR, apply_plot_style

# Category display order (strongest → weakest agreement, roughly)
CAT_ORDER = [
    "retail",
    "eat_and_drink",
    "business_and_services",
    "education",
    "health_and_medical",
    "accommodation",
]

# Wong (2011) colorblind-safe palette — Nature/Science standard.
# Ordered to alternate warm/cool for maximum separation in line plots.
PUB_CATEGORY_COLORS = {
    "retail": "#0072B2",  # blue
    "eat_and_drink": "#E69F00",  # amber
    "business_and_services": "#009E73",  # teal
    "education": "#CC79A7",  # mauve
    "health_and_medical": "#D55E00",  # vermillion
    "accommodation": "#999999",  # grey
}


def _cat_label(cat: str) -> str:
    return CATEGORY_NAMES.get(cat, cat.replace("_", " ").title())


def _cat_label_tex(cat: str) -> str:
    """LaTeX-safe category label (escapes &)."""
    return _cat_label(cat).replace("&", "\\&")


def _load_support(src: Path | pd.DataFrame) -> pd.DataFrame:
    """Load support matrix from a path or return a copy of an existing DataFrame."""
    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    if "estimate_type" in df.columns:
        df = df[df["estimate_type"] == "ci"].copy()
    return df


def _compute_between_city_correlations(
    city_summaries_csv: Path, n_boot: int = 2000, metric_family: str = "count_nw"
) -> pd.DataFrame:
    """Compute between-city Spearman ρ and Kendall τ (with bootstrap CI) from city-level summaries."""
    from scipy import stats as scipy_stats

    df = pd.read_csv(city_summaries_csv)
    rng = np.random.default_rng(42)

    # Spearman and Kendall variants for each between-city metric
    DT_METRIC = {
        "between_spearman_counts": (metric_family, f"{metric_family}_median_log1p"),
        "between_spearman_hotspot": (metric_family, f"{metric_family}_top10_median_log1p"),
    }
    DT_KENDALL = {
        "between_kendall_counts": (metric_family, f"{metric_family}_median_log1p"),
        "between_kendall_hotspot": (metric_family, f"{metric_family}_top10_median_log1p"),
    }

    rows = []
    for dt, (mf, city_metric) in DT_METRIC.items():
        for dist in [200, 400, 800, 1200, 1600]:
            for cat in CAT_ORDER:
                sub = df[(df["category"] == cat) & (df["distance_m"] == dist) & (df["metric"] == city_metric)]
                wide = sub.pivot(index="bounds_fid", columns="source", values="value").dropna()
                if len(wide) < 4:
                    continue
                r, o = wide["reg"].values, wide["ovt"].values
                rho = scipy_stats.spearmanr(r, o).statistic
                n = len(r)
                boot = [scipy_stats.spearmanr(r[idx := rng.integers(0, n, n)], o[idx]).statistic for _ in range(n_boot)]
                rows.append(
                    {
                        "category": cat,
                        "dimension": "distance_m",
                        "value": float(dist),
                        "metric_family": mf,
                        "decision_type": dt,
                        "estimate_type": "ci",
                        "score": rho,
                        "ci_low": float(np.percentile(boot, 2.5)),
                        "ci_high": float(np.percentile(boot, 97.5)),
                        "label": "between_spearman",
                        "n_cities": n,
                    }
                )

    for dt, (mf, city_metric) in DT_KENDALL.items():
        for dist in [200, 400, 800, 1200, 1600]:
            for cat in CAT_ORDER:
                sub = df[(df["category"] == cat) & (df["distance_m"] == dist) & (df["metric"] == city_metric)]
                wide = sub.pivot(index="bounds_fid", columns="source", values="value").dropna()
                if len(wide) < 4:
                    continue
                r, o = wide["reg"].values, wide["ovt"].values
                tau = scipy_stats.kendalltau(r, o).statistic
                n = len(r)
                boot = [
                    scipy_stats.kendalltau(r[idx := rng.integers(0, n, n)], o[idx]).statistic for _ in range(n_boot)
                ]
                rows.append(
                    {
                        "category": cat,
                        "dimension": "distance_m",
                        "value": float(dist),
                        "metric_family": mf,
                        "decision_type": dt,
                        "estimate_type": "ci",
                        "score": tau,
                        "ci_low": float(np.percentile(boot, 2.5)),
                        "ci_high": float(np.percentile(boot, 97.5)),
                        "label": "between_kendall",
                        "n_cities": n,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Canonical 2x3 claim structure — used in figure panels and heatmap blocks.
# The table uses its own cell specs (Kendall τ for rank-correlation columns).
# Rows: within-city, between-city.  Columns: counts, hotspots, nearest.
CELL_SPEC = [
    # (label, decision_type, metric_family, x_label)
    # --- Within-city row ---
    ("Within: counts", "distribution_spearman", "count_nw", "Catchment distance (m)"),
    ("Within: hotspots", "hotspot_top10", "count_nw", "Catchment distance (m)"),
    ("Within: nearest", "nearest_within_f1", "nearest", "Tolerance (m)"),
    # --- Between-city row ---
    ("Between: counts", "between_spearman_counts", "count_nw", "Catchment distance (m)"),
    ("Between: hotspots", "between_spearman_hotspot", "count_nw", "Catchment distance (m)"),
    ("Between: nearest", "benchmark_nearest", "nearest", "Tolerance (m)"),
]
# Weighted variant for supplementary figures
CELL_SPEC_WT = [
    ("Within: counts", "distribution_spearman", "count_wt", "Catchment distance (m)"),
    ("Within: hotspots", "hotspot_top10", "count_wt", "Catchment distance (m)"),
    ("Within: nearest", "nearest_within_f1", "nearest", "Tolerance (m)"),
    ("Between: counts", "between_spearman_counts", "count_wt", "Catchment distance (m)"),
    ("Between: hotspots", "between_spearman_hotspot", "count_wt", "Catchment distance (m)"),
    ("Between: nearest", "benchmark_nearest", "nearest", "Tolerance (m)"),
]
# Short column headers for the minima table (3 families; scope is a row group)
CELL_TABLE_HEADERS = ["Counts", "Hotspots", "Nearest"]


# ---------------------------------------------------------------------------
# Figure 1: 2x3 agreement curves (one per claim)
# ---------------------------------------------------------------------------


def fig_agreement_panels(
    support: Path | pd.DataFrame, out_path: Path, *, cell_spec: list | None = None
) -> None:
    """2x3 figure: rows = within/between city, columns = counts/hotspots/nearest."""
    apply_plot_style()
    df = _load_support(support)
    cell_spec = cell_spec or CELL_SPEC

    panel_letters = "ABCDEF"
    panel_titles = [
        "Within-city: rank correlation (Spearman \u03c1)",
        "Within-city: hotspot overlap (top 10\u0025)",
        "Within-city: nearest-distance F1",
        "Between-city: Spearman \u03c1 on city medians",
        "Between-city: Spearman \u03c1 on hotspot intensity",
        "Between-city: nearest-share agreement",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for idx, (_label, dt, mf, xlabel) in enumerate(cell_spec):
        ax = axes.flatten()[idx]
        data = df[(df["decision_type"] == dt) & (df["metric_family"] == mf)].copy()
        data = data.dropna(subset=["ci_low", "ci_high"])
        values = sorted(data["value"].dropna().unique())

        for cat in CAT_ORDER:
            sub = data[data["category"] == cat].sort_values("value")
            if sub.empty:
                continue
            color = PUB_CATEGORY_COLORS.get(cat, "#333333")
            ax.plot(
                sub["value"],
                sub["score"],
                marker="o",
                markersize=4,
                color=color,
                label=_cat_label(cat),
                linewidth=1.6,
                zorder=3,
            )
            ax.fill_between(sub["value"], sub["ci_low"], sub["ci_high"], alpha=0.12, color=color, zorder=2)

        ax.set_ylim(-0.02, 1.05)
        ax.grid(color="#ECECEC", linewidth=0.5, zorder=0)
        ax.set_title(f"({panel_letters[idx]}) {panel_titles[idx]}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8.5)
        if values:
            ax.set_xticks(values)
            left_pad = 30 if mf == "nearest" else 40
            right_pad = 30 if mf == "nearest" else 40
            ax.set_xlim(min(values) - left_pad, max(values) + right_pad)
            if mf == "nearest":
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    _PANEL_YLABELS = [
        ["Spearman \u03c1", "Top-10% overlap", "F1"],
        ["Spearman \u03c1", "Spearman \u03c1", "City F1 (shares)"],
    ]
    for row_idx in range(2):
        for col_idx in range(3):
            axes[row_idx, col_idx].set_ylabel(_PANEL_YLABELS[row_idx][col_idx], fontsize=8.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(CAT_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.5,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Support heatmap
# ---------------------------------------------------------------------------


def fig_support_heatmap(
    support: Path | pd.DataFrame, out_path: Path, *, cell_spec: list | None = None
) -> None:
    """Bubble chart: 2×3 subplot grid, category x distance/tolerance per cell."""
    apply_plot_style()
    df = _load_support(support)
    cell_spec = cell_spec or CELL_SPEC

    import matplotlib.colors as mcolors

    # One colour ramp per row: blue for within-city, green for between-city.
    SCOPE_CMAPS = {
        "within": plt.get_cmap("Blues"),
        "between": plt.get_cmap("Greens"),
    }
    SCOPE_LABELS = {
        "within": "Within-city",
        "between": "Between-city",
    }
    DT_TO_SCOPE = {
        "distribution_spearman": "within",
        "hotspot_top10": "within",
        "nearest_within_f1": "within",
        "between_spearman_counts": "between",
        "between_spearman_hotspot": "between",
        "benchmark_nearest": "between",
    }
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Six claim blocks matching the 2×3 cell spec
    blocks = []
    for label, dt, mf, _xl in cell_spec:
        bdf = df[(df["decision_type"] == dt) & (df["metric_family"] == mf)].copy()
        bdf = bdf.dropna(subset=["ci_low", "ci_high"])
        blocks.append((label, bdf))

    n_cats = len(CAT_ORDER)

    # Build cell data lookup: (cat_idx, block_label, value) → (label, score)
    cell_data = {}
    for block_label, bdf in blocks:
        for _, r in bdf.iterrows():
            cat = r["category"]
            if cat not in CAT_ORDER:
                continue
            cat_idx = CAT_ORDER.index(cat)
            cell_data[(cat_idx, block_label, int(r["value"]))] = (r["label"], r["score"])

    S_MIN, S_MAX = 120, 900

    # Height scaled per category row, with extra space for labels and colourbar.
    fig, axes = plt.subplots(2, 3, figsize=(15, n_cats * 1.6 + 2.2), sharey=True)

    for idx, (label, dt, _mf, xlabel) in enumerate(cell_spec):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        block_label, bdf = blocks[idx]
        values = sorted(bdf["value"].unique())
        cmap = SCOPE_CMAPS[DT_TO_SCOPE[dt]]

        # Collect bubble data for this subplot
        xs, ys, sizes, face_cols, annotations = [], [], [], [], []
        for i in range(n_cats):
            for x_pos, val in enumerate(values):
                data = cell_data.get((i, block_label, int(val)))
                if data is None:
                    annotations.append((x_pos, i, "nd", "#777777"))
                    continue
                lbl, score = data
                if pd.isna(score):
                    annotations.append((x_pos, i, "nan", "#777777"))
                    continue
                xs.append(x_pos)
                ys.append(i)
                clamped = max(0.0, min(1.0, score))
                sizes.append(S_MIN + (S_MAX - S_MIN) * clamped)
                face_cols.append(cmap(norm(clamped)))

                txt = f"{score:.2f}".lstrip("0") if score < 1.0 else "1.0"
                if score < 0:
                    txt = f"\u2212{abs(score):.2f}".lstrip("0") if abs(score) < 1.0 else f"\u2212{abs(score):.2f}"
                tc = "white" if clamped >= 0.55 else "#333333"
                annotations.append((x_pos, i, txt, tc))

        if xs:
            ax.scatter(
                xs,
                ys,
                s=sizes,
                facecolors=face_cols,
                edgecolors="#CCCCCC",
                linewidths=0.5,
                zorder=2,
                clip_on=False,
            )
        for x_pos, y_pos, txt, tc in annotations:
            ax.text(x_pos, y_pos, txt, ha="center", va="center", fontsize=11, color=tc, fontweight="bold", zorder=3)

        # Axis config
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(int(v)) for v in values], fontsize=15, rotation=45, ha="right")
        ax.tick_params(axis="x", pad=5)
        ax.set_xlabel(xlabel, fontsize=16, labelpad=6)
        # Slightly padded x-limits so the panels don't feel cramped while
        # still keeping the gutters between the 3 column blocks minimal.
        ax.set_xlim(-0.6, len(values) - 0.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax.grid(axis="y", color="#F0F0F0", linewidth=0.5, zorder=0)

        # Strip "Within: " / "Between: " prefix for column titles (scope is a row label)
        short_label = label.split(": ", 1)[1] if ": " in label else label
        ax.set_title(short_label.capitalize(), fontsize=17, fontweight="bold", pad=10)

    # Invert y-axis once (sharey shares the state across subplots)
    # Invert + add a bit of vertical padding so titles/xticklabels don't crowd the grid.
    axes[0, 0].set_ylim(n_cats - 0.6, -0.6)

    # Y-axis labels (left column only, via sharey)
    for row_idx in range(2):
        axes[row_idx, 0].set_yticks(range(n_cats))
        axes[row_idx, 0].set_yticklabels([_cat_label(c) for c in CAT_ORDER], fontsize=16)

    # Row scope labels — keep close to the panels to avoid extra inter-row whitespace.
    for row_idx, scope in enumerate(["Within-city", "Between-city"]):
        ax0 = axes[row_idx, 0]
        ax0.annotate(
            scope,
            xy=(-0.02, 1.03),
            xycoords="axes fraction",
            fontsize=19,
            fontweight="bold",
            fontstyle="italic",
            ha="right",
            va="bottom",
        )

    # Keep gutters small but non-zero to avoid panels visually merging.
    fig.subplots_adjust(left=0.10, right=0.995, wspace=0.05, hspace=0.52, bottom=0.16)

    # Colour-family legend: round dots, bold colours, centred below the figure.
    import matplotlib.lines as mlines

    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=11, color=cmap(0.80), label=SCOPE_LABELS[scope])
        for scope, cmap in SCOPE_CMAPS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(SCOPE_CMAPS),
        bbox_to_anchor=(0.5, 0.01),
        fontsize=17,
        frameon=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Table: Clean minima
# ---------------------------------------------------------------------------


def table_minima_clean(
    support: Path | pd.DataFrame, out_path: Path, *, metric_family: str = "count_nw"
) -> None:
    """Concordance-threshold minima table.

    For each category × claim, reports the minimum distance/tolerance at which
    the median agreement score reaches ≥70%, ≥75%, ≥80%, ≥85% concordance.
    Rank-correlation cells use Kendall τ directly (concordance = (1+τ)/2,
    so τ threshold = 2×concordance − 1).  Overlap and F1 cells use the
    concordance fraction as the score threshold.
    """
    sm = _load_support(support)
    sm_ci = sm.dropna(subset=["ci_low", "ci_high"])

    # Concordance thresholds (fractions)
    _CONCORDANCE = [0.70, 0.75, 0.80, 0.85]

    def _min_exceeds(cat: str, decision_type: str, mf: str, concordance: float) -> str:
        """Minimum distance/tolerance where score >= concordance-equivalent threshold."""
        # Kendall τ decision types: threshold = 2×concordance − 1
        is_kendall = decision_type.startswith("distribution_kendall") or decision_type.startswith("between_kendall")
        # Overlap / F1: concordance fraction is the score directly
        threshold = 2 * concordance - 1 if is_kendall else concordance
        # Kendall τ rows have no bootstrap CIs — use the full matrix for those.
        source = sm if is_kendall else sm_ci
        base = source[
            (source["category"] == cat)
            & (source["decision_type"] == decision_type)
            & (source["metric_family"] == mf)
        ]
        above = base[base["score"] >= threshold]
        if above.empty:
            return "---"
        return str(int(above["value"].min()))

    # Table cell specs: use Kendall τ for rank-correlation columns,
    # keep overlap/F1 columns unchanged.
    within_cells = [
        ("Within: counts", "distribution_kendall", metric_family, "Catchment distance (m)"),
        ("Within: hotspots", "hotspot_top10", metric_family, "Catchment distance (m)"),
        ("Within: nearest", "nearest_within_f1", "nearest", "Tolerance (m)"),
    ]
    between_cells = [
        ("Between: counts", "between_kendall_counts", metric_family, "Catchment distance (m)"),
        ("Between: hotspots", "between_kendall_hotspot", metric_family, "Catchment distance (m)"),
        ("Between: nearest", "benchmark_nearest", "nearest", "Tolerance (m)"),
    ]

    def _build_scope_rows(cell_spec_subset):
        scope_rows = []
        for cat in CAT_ORDER:
            cells = [_cat_label_tex(cat)]
            for _label, dt, mf, _xl in cell_spec_subset:
                for conc in _CONCORDANCE:
                    cells.append(_min_exceeds(cat, dt, mf, conc))
            scope_rows.append(cells)
        return scope_rows

    within_rows = _build_scope_rows(within_cells)
    between_rows = _build_scope_rows(between_cells)

    # LaTeX columns = category + 3 families × 4 thresholds
    n_total = 1 + 3 * len(_CONCORDANCE)  # 13
    col_spec = "l" + "c" * (n_total - 1)
    thr_labels = " & ".join(f"$\\geq${int(c * 100)}\\%" for c in _CONCORDANCE)

    lines = [
        f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        "\\toprule",
    ]
    # Top header: family names spanning 3 sub-columns each
    family_header = "\\textbf{Category}"
    for h in CELL_TABLE_HEADERS:
        family_header += f" & \\multicolumn{{{len(_CONCORDANCE)}}}{{c}}{{\\textbf{{{h}}}}}"
    lines.append(family_header + " \\\\")
    # Cmidrules under each family group
    for i, _h in enumerate(CELL_TABLE_HEADERS):
        start = 2 + i * len(_CONCORDANCE)
        end = start + len(_CONCORDANCE) - 1
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")
    # Sub-header: threshold values
    sub_header = " & " + " & ".join([thr_labels] * len(CELL_TABLE_HEADERS))
    lines.append(sub_header + " \\\\")
    within_metric_labels = ["rank concordance", "top-10\\% overlap", "nearest F1"]
    between_metric_labels = [
        "rank concordance (city medians)",
        "rank concordance (hotspot intensity)",
        "harmonic-mean share",
    ]
    n_thr = len(_CONCORDANCE)

    def _scope_header(scope_label, metric_labels):
        parts = [f"\\textit{{{scope_label}}}"]
        for lbl in metric_labels:
            parts.append(f"\\multicolumn{{{n_thr}}}{{c}}{{\\footnotesize {lbl}}}")
        return " & ".join(parts) + " \\\\"

    lines.append("\\midrule")
    lines.append(_scope_header("Within-city", within_metric_labels))
    for cells in within_rows:
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append(_scope_header("Between-city", between_metric_labels))
    for cells in between_rows:
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\u2713 Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Supplementary: City-level variability
# ---------------------------------------------------------------------------


def fig_city_variability(
    node_agreement_csv: Path, out_path: Path, *, metric_family: str = "count_nw"
) -> None:
    """Horizontal strip plot showing per-city Spearman rho at key distances."""
    apply_plot_style()
    df = pd.read_csv(node_agreement_csv)

    # Filter to metric family and two representative distances
    sub = df[(df["metric"] == metric_family) & (df["distance_m"].isin([400, 800]))].copy()
    if sub.empty:
        print("No data for city variability figure.")
        return

    sub["cat_label"] = sub["category"].map(_cat_label)
    cat_label_order = [_cat_label(c) for c in CAT_ORDER]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

    for ax, dist in zip(axes, [400, 800], strict=False):
        dsub = sub[sub["distance_m"] == dist].copy()
        cats_present = [c for c in cat_label_order if c in dsub["cat_label"].values]
        n_cats = len(cats_present)

        for i, cat_label in enumerate(cats_present):
            cat_data = dsub[dsub["cat_label"] == cat_label]["spearman_rho"].dropna()
            if cat_data.empty:
                continue
            cat_key = [k for k, v in CATEGORY_NAMES.items() if v == cat_label]
            if not cat_key:
                cat_key = [cat_label.lower().replace(" & ", "_and_").replace(" ", "_")]
            color = PUB_CATEGORY_COLORS.get(cat_key[0], "#333333")
            y = n_cats - 1 - i  # top-to-bottom order

            # Horizontal median line (thin, behind points)
            median = cat_data.median()
            ax.plot([cat_data.min(), cat_data.max()], [y, y], color=color, linewidth=0.6, alpha=0.3, zorder=1)

            # Strip points (jittered vertically)
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.18, 0.18, size=len(cat_data))
            ax.scatter(
                cat_data.values, y + jitter, color=color, alpha=0.55, s=18, edgecolors="white", linewidths=0.3, zorder=3
            )

            # Median diamond
            ax.scatter([median], [y], color=color, marker="D", s=45, edgecolors="white", linewidths=0.8, zorder=4)

        ax.grid(axis="x", color="#ECECEC", linewidth=0.5, zorder=0)
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(list(reversed(cats_present)), fontsize=8.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(f"Catchment distance: {dist} m", fontsize=10)

    axes[1].set_xlabel("Spearman \u03c1 (per city)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\u2713 Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Supplementary: Cross-city scatter (Overture vs registry city medians)
# ---------------------------------------------------------------------------


def fig_cross_city_scatter(
    city_summaries_csv: Path, out_path: Path, *, metric_family: str = "count_nw"
) -> None:
    """Scatter plot: city-level median accessibility (Overture vs registry) at 800 m."""
    apply_plot_style()
    df = pd.read_csv(city_summaries_csv)

    dist = 800
    metric = f"{metric_family}_median_log1p"
    sub = df[(df["distance_m"] == dist) & (df["metric"] == metric)].copy()
    if sub.empty:
        print("No data for cross-city scatter.")
        return

    # Pivot to get ovt vs reg per city×category
    wide = sub.pivot_table(index=["bounds_fid", "category"], columns="source", values="value").reset_index()
    if "ovt" not in wide.columns or "reg" not in wide.columns:
        print("Missing ovt/reg columns for cross-city scatter.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, cat in zip(axes_flat, CAT_ORDER, strict=False):
        cat_data = wide[wide["category"] == cat].dropna(subset=["ovt", "reg"])
        if cat_data.empty:
            ax.set_visible(False)
            continue
        color = PUB_CATEGORY_COLORS.get(cat, "#333333")
        ax.scatter(
            cat_data["reg"],
            cat_data["ovt"],
            color=color,
            alpha=0.55,
            s=25,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

        # Identity line
        lo = min(cat_data["reg"].min(), cat_data["ovt"].min())
        hi = max(cat_data["reg"].max(), cat_data["ovt"].max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            color="#CCCCCC",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(_cat_label(cat), fontsize=10)
        ax.grid(color="#F0F0F0", linewidth=0.5, zorder=0)

    # Shared axis labels
    fig.supxlabel("Registry median accessibility (log count)", fontsize=10)
    fig.supylabel("Overture median accessibility (log count)", fontsize=10)

    fig.suptitle(f"Cross-city comparison at {dist} m catchment", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    csv_dir = OUTPUT_DIR / "csv"
    support_csv = csv_dir / "pointa_support_matrix.csv"
    city_summaries_csv = csv_dir / "pointa_accessibility_city_summaries.csv"
    node_agreement_csv = csv_dir / "pointa_accessibility_node_agreement.csv"

    if not support_csv.exists():
        print(f"Missing: {support_csv} — run s04_build_support_matrix.py first.")
        return 1
    if not city_summaries_csv.exists():
        print(f"Missing: {city_summaries_csv} — run s03_summarise_city_agreement.py first.")
        return 1

    # Skip the (bootstrap-heavy) regeneration if all manuscript-input artifacts
    # already exist. Delete an output to force a rebuild.
    expected_outputs = [
        FIG_DIR / "fig_pointa_agreement_panels.pdf",
        FIG_DIR / "fig_pointa_support_heatmap.pdf",
        TABLE_DIR / "table_pointa_minima_clean.tex",
        FIG_DIR / "fig_pointa_agreement_panels_wt.pdf",
        FIG_DIR / "fig_pointa_support_heatmap_wt.pdf",
        TABLE_DIR / "table_pointa_minima_clean_wt.tex",
        FIG_DIR / "fig_pointa_cross_city_scatter.pdf",
    ]
    if all(p.exists() for p in expected_outputs):
        print("✓ Point A figures and tables already exist, skipping.")
        return 0

    base = pd.read_csv(support_csv)
    between_types = {
        "between_spearman_counts",
        "between_spearman_hotspot",
        "between_kendall_counts",
        "between_kendall_hotspot",
    }

    # Unweighted (main paper)
    between_city_nw = _compute_between_city_correlations(city_summaries_csv, metric_family="count_nw")
    support = pd.concat(
        [base[~base["decision_type"].isin(between_types)], between_city_nw],
        ignore_index=True,
    )

    # Main paper figures
    fig_agreement_panels(support, FIG_DIR / "fig_pointa_agreement_panels.pdf")
    fig_support_heatmap(support, FIG_DIR / "fig_pointa_support_heatmap.pdf")

    # Main paper table
    table_minima_clean(support, TABLE_DIR / "table_pointa_minima_clean.tex")

    # Weighted (supplementary)
    between_city_wt = _compute_between_city_correlations(city_summaries_csv, metric_family="count_wt")
    support_wt = pd.concat(
        [base[~base["decision_type"].isin(between_types)], between_city_wt],
        ignore_index=True,
    )
    fig_agreement_panels(support_wt, FIG_DIR / "fig_pointa_agreement_panels_wt.pdf", cell_spec=CELL_SPEC_WT)
    fig_support_heatmap(support_wt, FIG_DIR / "fig_pointa_support_heatmap_wt.pdf", cell_spec=CELL_SPEC_WT)
    table_minima_clean(support_wt, TABLE_DIR / "table_pointa_minima_clean_wt.tex", metric_family="count_wt")

    # Supplementary (unweighted)
    if node_agreement_csv.exists():
        fig_city_variability(node_agreement_csv, FIG_DIR / "fig_pointa_city_variability.pdf")
    fig_cross_city_scatter(city_summaries_csv, FIG_DIR / "fig_pointa_cross_city_scatter.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
