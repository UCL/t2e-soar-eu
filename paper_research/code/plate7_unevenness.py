"""
Plate 7 — Unevenness of Access.

Donut matrix where donut SIZE scales with the P75/P25 ratio.  The
inner hole represents P25 (fixed baseline).  The outer ring grows
with the ratio — small donut = equal access, large donut = unequal.
The ring thickness directly IS the inequality gap.

Columns = 8 octant types, rows = service categories.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
    GREY,
    METRIC_LAYERS,
    OCTANT_ORDER,
    OUTPUT_DIR,
    SYM_INCHES,
    apply_atlas_style,
    classify_octants,
    draw_legend_row,
    draw_title,
    load_all_cached,
    place_composite_symbol,
    standard_margins,
)

apply_atlas_style()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Build from shared METRIC_LAYERS for consistent colours
_METRIC_MAP = {m["col"]: m for layer in METRIC_LAYERS for m in layer["metrics"]}
INEQ_SERVICES = [
    {"col": m["col"], "name": m["name"], "color": m["color"]}
    for m in [
        _METRIC_MAP["cc_trees_nearest_max_1600"],
        _METRIC_MAP["cc_green_nearest_max_1600"],
        _METRIC_MAP["cc_business_and_services_nearest_max_1600"],
        _METRIC_MAP["cc_retail_nearest_max_1600"],
        _METRIC_MAP["cc_transport_nearest_max_1600"],
        _METRIC_MAP["cc_eat_and_drink_nearest_max_1600"],
        _METRIC_MAP["cc_health_and_medical_nearest_max_1600"],
        _METRIC_MAP["cc_education_nearest_max_1600"],
        _METRIC_MAP["cc_attractions_and_activities_nearest_max_1600"],
        _METRIC_MAP["cc_arts_and_entertainment_nearest_max_1600"],
        _METRIC_MAP["cc_religious_nearest_max_1600"],
    ]
]

P_LO, P_HI = 25, 75

# Absolute distance scale: distance in metres → radius in data coords.
# All donuts share this scale so sizes are directly comparable.
DIST_MAX = 1600.0  # maximum distance on the scale (metres, matches max query distance)
R_MAX = 0.38  # radius at DIST_MAX in data coords
R_MIN = 0.12  # minimum outer radius so small donuts remain visible


# ============================================================================
# DATA
# ============================================================================


def load_data():
    needed = list(AXIS_COLS.values()) + [s["col"] for s in INEQ_SERVICES]
    df = load_all_cached(columns=needed)
    print(f"  {len(df):,} nodes, {df['bounds_fid'].nunique()} cities")
    classified, _ = classify_octants(df)
    print(f"  {len(classified):,} classified nodes")
    return classified


def compute_inequality(classified):
    """Return dict of {col: {octant: (p25, p75)}}."""
    results = {}
    for svc in INEQ_SERVICES:
        col = svc["col"]
        vals = classified[["octant", col]].dropna(subset=[col])
        pairs = {}
        for octant in OCTANT_ORDER:
            ov = vals.loc[vals["octant"] == octant, col]
            if len(ov) < 50:
                pairs[octant] = (np.nan, np.nan)
                continue
            p25 = ov.quantile(P_LO / 100)
            p75 = ov.quantile(P_HI / 100)
            pairs[octant] = (p25, p75)
        results[col] = pairs
    return results


# ============================================================================
# FIGURE
# ============================================================================


def _dist_to_r(d):
    """Convert distance in metres to radius in data coords."""
    return (d / DIST_MAX) * R_MAX


def draw_donut(ax, cx, cy, p25, p75, color):
    """Draw a donut with absolute distance scaling.

    Inner radius = P25, outer radius = P75, on a shared global scale.
    """
    base_rgb = mcolors.to_rgb(color)
    inner_r = max(_dist_to_r(p25), R_MIN * 0.4)
    outer_r = max(_dist_to_r(p75), R_MIN)
    gap = p75 - p25

    # Outer filled circle (the ring = inequality band)
    outer = mpatches.Circle(
        (cx, cy),
        outer_r,
        facecolor=base_rgb,
        edgecolor=base_rgb,
        linewidth=0.4,
        zorder=2,
    )
    ax.add_patch(outer)

    # Inner hole (the well-served core)
    inner = mpatches.Circle(
        (cx, cy),
        max(inner_r, 0.005),
        facecolor=BG,
        edgecolor=(*base_rgb, 0.40),
        linewidth=0.3,
        zorder=3,
    )
    ax.add_patch(inner)

    # Gap label placed by caller at row baseline
    return gap


def build_figure(inequality):
    """Donut matrix: rows = services, columns = octants."""
    n_svc = len(INEQ_SERVICES)
    n_oct = len(OCTANT_ORDER)

    fig_w = 7.5
    fig_h = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    margins = standard_margins(
        fig,
        top_inches=1.10,
        bottom_inches=0.52,
        left_inches=1.10,
        right_inches=0.40,
    )
    fig.subplots_adjust(**margins)

    # ── Draw donuts ────────────────────────────────────────────────
    ROW_STEP = 0.95  # row spacing between service rows
    LABEL_OFFSET = ROW_STEP * 0.36  # fixed distance below row centre for labels

    for si, svc in enumerate(INEQ_SERVICES):
        col = svc["col"]
        y = (n_svc - 1 - si) * ROW_STEP
        label_y = y - LABEL_OFFSET  # fixed baseline for this row

        for oi, octant in enumerate(OCTANT_ORDER):
            x = oi
            p25, p75 = inequality[col].get(octant, (np.nan, np.nan))
            if np.isnan(p25):
                continue
            gap = draw_donut(ax, x, y, p25, p75, svc["color"])
            # Label at fixed row baseline
            ax.text(
                x,
                label_y,
                f"{gap:.0f}m",
                fontsize=6,
                fontweight="medium",
                color=DARK,
                ha="center",
                va="top",
                zorder=4,
            )

    # ── Row labels ─────────────────────────────────────────────────
    for si, svc in enumerate(INEQ_SERVICES):
        y = (n_svc - 1 - si) * ROW_STEP
        ax.text(
            -0.55,
            y,
            svc["name"],
            fontsize=7,
            fontweight="bold",
            color=DARK,
            ha="right",
            va="center",
        )

    # ── Faint horizontal guide lines per row (start after label gap) ──
    for si in range(n_svc):
        y = (n_svc - 1 - si) * ROW_STEP
        ax.plot([-0.35, n_oct - 0.4], [y, y], color=GREY, lw=0.4, alpha=0.20, zorder=0)

    # ── Axis setup ─────────────────────────────────────────────────
    y_max = (n_svc - 1) * ROW_STEP
    ax.set_xlim(-0.6, n_oct - 0.4)
    ax.set_ylim(-0.45, y_max + 0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Octant symbols ─────────────────────────────────────────────
    fig.canvas.draw()
    fig_w_in, fig_h_in = fig.get_size_inches()
    bbox = ax.get_position()
    x0_data, x1_data = ax.get_xlim()

    for oi, octant in enumerate(OCTANT_ORDER):
        fx = bbox.x0 + (oi - x0_data) / (x1_data - x0_data) * bbox.width
        sym_fy = bbox.y1 + 0.050
        place_composite_symbol(fig, fx, sym_fy, octant, size_inches=SYM_INCHES)

    # ── Title, subtitle, legend ────────────────────────────────────
    draw_title(fig, "Within-Type Variation in Walking Distance")

    fig.text(
        0.5,
        0.42 / fig_h_in,
        f"Inner ring = P{P_LO} distance, outer ring = P{P_HI} distance "
        f"(shared scale).  Number = P{P_HI} \u2013 P{P_LO} gap in metres.",
        fontsize=6,
        color=DARK,
        ha="center",
        va="bottom",
    )

    draw_legend_row(fig, y_inches=0.25)

    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data...")
    classified = load_data()

    print("\nComputing inequality ratios...")
    inequality = compute_inequality(classified)
    for svc in INEQ_SERVICES:
        col = svc["col"]
        pairs = [(p25, p75) for p25, p75 in inequality[col].values() if not np.isnan(p25)]
        gaps = [p75 - p25 for p25, p75 in pairs]
        print(f"  {svc['name']}: gap {min(gaps):.0f}m -- {max(gaps):.0f}m")

    print("\nBuilding Plate 7 -- Unevenness of Access...")
    fig = build_figure(inequality)
    out = OUTPUT_DIR / "plate7_unevenness.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)
    print("\nDone.")
