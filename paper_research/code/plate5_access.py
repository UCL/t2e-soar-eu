"""
Plate 5 — Access from the Typical European Street.

Skyline ridgeline (city-level medians) + 8 octant value columns (node-level medians).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
    GREY,
    GRID_COLOR,
    METRIC_LAYERS,
    OCTANT_ORDER,
    OUTPUT_DIR,
    apply_atlas_style,
    classify_octants,
    fmt_value,
    load_all_cached,
)

apply_atlas_style()

# ============================================================================
# DATA LOADING
# ============================================================================


def load_data():
    """Load cached data, classify octants, compute summaries."""
    print("Loading cached data...")
    needed = [m["col"] for g in METRIC_LAYERS for m in g["metrics"]] + list(AXIS_COLS.values())
    df = load_all_cached(columns=needed)
    print(f"  {len(df):,} nodes, {df['bounds_fid'].nunique()} cities")

    # Filter METRIC_LAYERS to columns present in cache
    available_cols = set(df.columns)
    for group in METRIC_LAYERS:
        group["metrics"] = [m for m in group["metrics"] if m["col"] in available_cols]
    metric_cols = [m["col"] for g in METRIC_LAYERS for m in g["metrics"]]
    print(f"  Available metrics: {len(metric_cols)}")

    # City-level medians (for skyline)
    city_medians = df.groupby("bounds_fid")[metric_cols].median()
    print(f"  City medians: {len(city_medians)} cities × {len(metric_cols)} metrics")

    # Octant classification (for dots)
    classified, thresholds = classify_octants(df)
    print(f"  Classified: {len(classified):,} nodes into octants")
    print(f"  Thresholds: {thresholds}")

    # Octant medians
    octant_medians = classified.groupby("octant")[metric_cols].median()
    continental_medians = classified[metric_cols].median()

    return city_medians, classified, octant_medians, continental_medians, thresholds


# ============================================================================
# FIGURE 1A — SKYLINE RIDGELINE
# ============================================================================

# Layout constants
UNIFORM_PEAK = 0.20
VISUAL_X_MAX = 1000
LABEL_X = VISUAL_X_MAX + 50
DISTANCE_RANGE = (0, 1200)
KDE_BW = 0.25
GRID_MARKS = [200, 400, 600, 800, 1000, 1200]


# ============================================================================
# PLATE 3 — ACCESS: SKYLINE + OCTANT VALUES
# ============================================================================

COL_SPACING = 75  # visual-x spacing between octant columns
GLOBAL_X = VISUAL_X_MAX + 80  # x position for the global median number
OCTANT_START = GLOBAL_X + 85  # tighter gap between global and octant columns


def build_combined(city_medians, classified, octant_medians, continental_medians, n_cities=0):
    """Skyline (left, street-level distribution) + 8 octant value columns (right, node medians)."""
    # ── Row positions (shared by both panels) ────────────────────────
    # Build rows in display order: nearest at TOP (highest y), farthest at bottom.
    # Reverse both group order and metric order within groups so that
    # y_offset (increasing upward) puts tree canopy at the top.
    reversed_layers = []
    for group in reversed(METRIC_LAYERS):
        reversed_layers.append(
            {
                "label": group["label"],
                "metrics": list(reversed(group["metrics"])),
            }
        )

    rows = []
    y_offset = 0
    row_spacing = 0.13  # tighter rows
    group_gap = 0.14  # small gap between groups
    for gi, group in enumerate(reversed_layers):
        if gi > 0:
            y_offset += group_gap
        group_y_start = y_offset
        for m in group["metrics"]:
            rows.append({**m, "y": y_offset, "group": group["label"]})
            y_offset += row_spacing
        group["_y_start"] = group_y_start
        group["_y_end"] = y_offset - row_spacing

    total_height = y_offset
    right_end = OCTANT_START + 8 * COL_SPACING
    LABEL_LEFT = -50  # metric names go left of the ridgeline
    fig_w = 7.5
    fig_h = min(10.0, max(4, total_height * 1.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    from atlas_common import standard_margins

    fig.subplots_adjust(**standard_margins(fig))

    # Compute aspect ratio for symbol rendering
    x_range = right_end + 20 - (LABEL_LEFT - 380)
    y_range = total_height + 0.25 - (-0.18)
    (fig_h / fig_w) * (x_range / y_range)

    # ── RIDGELINE SKYLINE (centre) ───────────────────────────────────
    dist_y_max = total_height + 0.02
    grid_label_y = total_height + 0.04  # distance labels at TOP, with gap below title
    for gm in GRID_MARKS:
        vx = gm / DISTANCE_RANGE[1] * VISUAL_X_MAX
        ax.plot([vx, vx], [-0.02, dist_y_max], color=GRID_COLOR, alpha=0.5, lw=0.4, zorder=0)
        ax.text(vx, grid_label_y, f"{gm}", ha="center", va="bottom", fontsize=6, fontweight="bold", color=GREY)

    # Gradient encoding constants (shared for global + octant values)

    for draw_idx, row in enumerate(reversed(rows)):
        col = row["col"]
        color = row["color"]
        yo = row["y"]
        data = city_medians[col].dropna().values
        if len(data) < 5:
            continue

        kde = gaussian_kde(data, bw_method=KDE_BW)
        xs = np.linspace(0, DISTANCE_RANGE[1], 300)
        density = kde(xs)
        peak = density.max()
        if peak > 0:
            density = density / peak * UNIFORM_PEAK
        vxs = xs / DISTANCE_RANGE[1] * VISUAL_X_MAX

        alpha = 0.35 + draw_idx * 0.015
        ax.fill_between(vxs, yo, yo + density, color=color, alpha=alpha, edgecolor="none", zorder=2 + draw_idx)
        ax.plot(vxs, yo + density, color=color, lw=0.6, alpha=0.85, zorder=3 + draw_idx)
        ax.plot([vxs[0], vxs[-1]], [yo, yo], color=color, lw=0.3, alpha=0.2, zorder=1)

        med = np.nanmedian(data)
        med_vx = med / DISTANCE_RANGE[1] * VISUAL_X_MAX
        med_density = kde(med)[0] / peak * UNIFORM_PEAK if peak > 0 else 0
        ax.plot([med_vx, med_vx], [yo, yo + med_density], color=color, ls=":", lw=0.6, alpha=0.55, zorder=4 + draw_idx)

        # Metric name — LEFT of the ridgeline
        ax.text(
            LABEL_LEFT, yo + 0.015, row["name"], fontsize=6, fontweight="bold", color=color, ha="right", va="bottom"
        )

        # EU median (street-level) — RIGHT of the ridgeline (no units)
        cont_med = continental_medians.get(col, med)
        ax.text(
            GLOBAL_X,
            yo + 0.01,
            fmt_value(cont_med, ""),
            fontsize=6,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
        )

    # Domain group labels (far left, rotated)
    for group in reversed_layers:
        y_mid = (group["_y_start"] + group["_y_end"]) / 2
        ax.text(
            LABEL_LEFT - 350,
            y_mid,
            group["label"],
            fontsize=6,
            fontweight="bold",
            fontstyle="italic",
            color=GREY,
            rotation=90,
            ha="center",
            va="center",
        )

    # Set axes limits before placing symbols (needed for coord conversion)
    ax.set_xlim(LABEL_LEFT - 380, right_end + 20)
    ax.set_ylim(-0.18, total_height + 0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()  # force layout so get_position is accurate

    # ── OCTANT VALUE COLUMNS ─────────────────────────────────────────
    # Column headers: vector composite symbols via place_composite_symbol
    from atlas_common import place_composite_symbol

    bbox_main = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sym_y_data = total_height + 0.16
    sym_fy = bbox_main.y0 + (sym_y_data - y0) / (y1 - y0) * bbox_main.height
    for oi_idx, octant in enumerate(OCTANT_ORDER):
        xc_data = OCTANT_START + oi_idx * COL_SPACING + COL_SPACING / 2
        sym_fx = bbox_main.x0 + (xc_data - x0) / (x1 - x0) * bbox_main.width
        place_composite_symbol(fig, sym_fx, sym_fy, octant)

    # Values per metric × octant
    for row in rows:
        col = row["col"]
        yo = row["y"]
        metric_color = row["color"]
        cont_med = continental_medians.get(col, np.nan)
        if pd.isna(cont_med):
            continue

        for oi, octant in enumerate(OCTANT_ORDER):
            xc = OCTANT_START + oi * COL_SPACING + COL_SPACING / 2
            oct_med = octant_medians.loc[octant, col] if octant in octant_medians.index else np.nan
            if pd.isna(oct_med):
                continue

            ax.text(
                xc,
                yo + 0.01,
                fmt_value(oct_med, ""),
                fontsize=5,
                fontweight="bold",
                color=metric_color,
                ha="center",
                va="bottom",
            )

    # ── Titles and labels ────────────────────────────────────────────
    from atlas_common import draw_title

    draw_title(fig, "Walking Distances to Services and Green Space")

    # Bottom labels
    bottom_y = -0.06
    ax.text(
        VISUAL_X_MAX / 2,
        bottom_y,
        f"Distribution of walking distances by city medians across {n_cities} cities",
        fontsize=6,
        color=DARK,
        ha="center",
        va="top",
    )
    mid_right = (GLOBAL_X + OCTANT_START + 7 * COL_SPACING + COL_SPACING / 2) / 2
    ax.text(
        mid_right,
        bottom_y,
        "Median walking distances (m) by street and morphological type",
        fontsize=6,
        color=DARK,
        ha="center",
        va="top",
    )

    # ── Legend key row (shared function) ─────────────────────────
    from atlas_common import draw_legend_row

    draw_legend_row(fig)

    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    city_medians, classified, octant_medians, continental_medians, thresholds = load_data()

    print("\nBuilding Plate 5 — Access...")
    fig_comb = build_combined(city_medians, classified, octant_medians, continental_medians, n_cities=len(city_medians))
    out = OUTPUT_DIR / "plate5_access.pdf"
    fig_comb.savefig(out, dpi=300, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig_comb)

    print("\nDone.")
