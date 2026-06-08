"""
Plate 11 — Same Density, Different City.

Holds population density constant via quintile binning and shows that
morphological form still predicts service access.  Within each density
band, the 8 octant types spread across a wide range of distances —
the "form gap" that density alone cannot explain.

Layout: 4 stacked sub-panels (retail, education, transport, green
space), each showing 5 density quintile rows with 8 octant-coloured
dots.  An octant key row at the top maps colours to morphological types.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
    GREY,
    GRID_COLOR,
    OCTANT_COLORS,
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

PANELS = [
    {
        "col": "cc_retail_nearest_max_1600",
        "name": "Retail",
        "color": "#2D6A4F",
    },
    {
        "col": "cc_education_nearest_max_1600",
        "name": "Education",
        "color": "#9467bd",
    },
    {
        "col": "cc_transport_nearest_max_1600",
        "name": "Transport",
        "color": "#4A90D9",
    },
    {
        "col": "cc_green_nearest_max_1600",
        "name": "Green space",
        "color": "#52B788",
    },
]

N_Q = 5  # density quintiles


# ============================================================================
# DATA
# ============================================================================


def load_data():
    """Load cached data, classify octants, bin by population density."""
    needed = list(AXIS_COLS.values()) + [p["col"] for p in PANELS] + ["density"]
    df = load_all_cached(columns=needed)
    print(f"  {len(df):,} nodes, {df['bounds_fid'].nunique()} cities")

    classified, _ = classify_octants(df)
    classified = classified.dropna(subset=["density"])
    # Remove non-positive densities (artefacts from interpolation)
    classified = classified[classified["density"] > 0].copy()

    classified["dq"] = (
        pd.qcut(
            classified["density"],
            N_Q,
            labels=False,
            duplicates="drop",
        )
        + 1
    )  # 1-indexed: Q1 = sparsest, Q5 = densest

    dq_stats = classified.groupby("dq")["density"].agg(["median", "min", "max"])
    print(f"  Density quintiles: {len(dq_stats)}")
    for q in dq_stats.index:
        row = dq_stats.loc[q]
        print(f"    Q{q}: {row['min']:,.0f} – {row['max']:,.0f}  (med {row['median']:,.0f})")

    return classified, dq_stats


# ============================================================================
# FIGURE
# ============================================================================


def build_figure(classified, dq_stats):
    """Build the 4-panel density-controlled dot-strip figure."""
    n_panels = len(PANELS)
    fig_w = 7.5
    fig_h = 10.0  # use full Elsevier text height

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, fig_h),
        facecolor=BG,
    )
    if n_panels == 1:
        axes = [axes]

    margins = standard_margins(
        fig,
        top_inches=1.35,
        bottom_inches=0.90,
        left_inches=1.05,
        right_inches=0.60,
    )
    fig.subplots_adjust(**margins, hspace=0.30)

    quintiles = sorted(classified["dq"].unique())

    for pi, panel in enumerate(PANELS):
        ax = axes[pi]
        ax.set_facecolor(BG)
        col = panel["col"]

        # ── Compute quintile × octant medians ──────────────────────
        qo = classified.groupby(["dq", "octant"])[col].median().unstack("octant")
        q_all = classified.groupby("dq")[col].median()

        # Determine x-range from data
        all_vals = qo.values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        x_max = np.percentile(all_vals, 98) * 1.15

        for qi, q in enumerate(quintiles):
            y = qi

            # Overall quintile median — grey reference tick
            ref = q_all.get(q, np.nan)
            if not pd.isna(ref):
                ax.plot(
                    [ref, ref],
                    [y - 0.32, y + 0.32],
                    color=GREY,
                    lw=0.6,
                    alpha=0.35,
                    zorder=1,
                )

            # Collect octant values for span line
            oct_vals = {}
            for octant in OCTANT_ORDER:
                if octant in qo.columns and q in qo.index:
                    v = qo.loc[q, octant]
                    if not pd.isna(v):
                        oct_vals[octant] = v

            if len(oct_vals) >= 2:
                lo, hi = min(oct_vals.values()), max(oct_vals.values())
                # Span line
                ax.plot(
                    [lo, hi],
                    [y, y],
                    color=GREY,
                    lw=0.7,
                    alpha=0.2,
                    zorder=1,
                )
                # Form gap annotation
                gap = hi - lo
                ax.text(
                    hi + x_max * 0.015,
                    y,
                    f"Δ{gap:.0f}m",
                    fontsize=5,
                    color=GREY,
                    va="center",
                    ha="left",
                )

            # Octant dots — nudge vertically only when nearly identical x
            sorted_octs = sorted(oct_vals.items(), key=lambda kv: kv[1])
            min_gap = x_max * 0.004  # only near-identical values
            dy = 0.065  # subtle up/down nudge
            offsets = {o: 0.0 for o, _ in sorted_octs}
            # Find pairs that are essentially on top of each other
            for j in range(1, len(sorted_octs)):
                prev_oct, prev_x = sorted_octs[j - 1]
                cur_oct, cur_x = sorted_octs[j]
                if cur_x - prev_x < min_gap:
                    # Only nudge if neither was already nudged
                    if offsets[prev_oct] == 0.0:
                        offsets[prev_oct] = -dy
                    offsets[cur_oct] = dy

            for octant in OCTANT_ORDER:
                if octant in oct_vals:
                    ax.scatter(
                        oct_vals[octant],
                        y + offsets[octant],
                        color=OCTANT_COLORS[octant],
                        s=30,
                        zorder=3,
                        edgecolors="white",
                        linewidth=0.4,
                    )

        # ── Axis styling ───────────────────────────────────────────
        ax.set_yticks(range(len(quintiles)))
        ylabels = []
        for q in quintiles:
            med = dq_stats.loc[q, "median"]
            ylabels.append(f"Q{q}  {med:,.0f}/km²")
        ax.set_yticklabels(ylabels, fontsize=6, color=GREY)
        ax.tick_params(axis="y", length=0, pad=4)

        ax.set_xlim(0, x_max)
        ax.tick_params(axis="x", labelsize=5, colors=GREY, length=2)

        # Panel title — top centre, inside the axes
        ax.set_title(
            panel["name"],
            fontsize=8,
            fontweight="bold",
            color=DARK,
            pad=4,
        )

        # "Density" label above the y-axis labels (every panel)
        ax.text(
            -0.005,
            1.0,
            "Density",
            transform=ax.transAxes,
            fontsize=6,
            fontweight="bold",
            color=GREY,
            va="bottom",
            ha="right",
        )

        # Faint x-grid
        ax.grid(axis="x", color=GRID_COLOR, alpha=0.2, lw=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_ylim(-0.6, len(quintiles) - 0.4)
        ax.invert_yaxis()  # densest at top

    # ── Octant key row (between title and panels) ──────────────────
    fig.canvas.draw()
    fig_w_in, fig_h_in = fig.get_size_inches()
    # Centre the symbols vertically within the top margin
    # Title is at ~0.08" from top; first panel starts at top_inches=1.35"
    sym_centre_y = 1.0 - 0.62 / fig_h_in
    n_oct = len(OCTANT_ORDER)
    key_span = 0.78
    x_start = 0.5 - key_span / 2
    x_step = key_span / n_oct

    DOT_SIZE = 11  # match chart dot size

    for i, octant in enumerate(OCTANT_ORDER):
        cx = x_start + (i + 0.5) * x_step
        place_composite_symbol(fig, cx, sym_centre_y, octant, size_inches=SYM_INCHES)
        # Coloured dot just below the irregularity sub-symbol
        dot_y = sym_centre_y - SYM_INCHES * 2.4 / fig_h_in
        fig.text(
            cx,
            dot_y,
            "\u25cf",
            fontsize=DOT_SIZE,
            color=OCTANT_COLORS[octant],
            ha="center",
            va="center",
        )

    # ── Title ──────────────────────────────────────────────────────
    draw_title(fig, "Service Distances within Population-Density Quintiles")

    # Subtitle above legend
    fig.text(
        0.5,
        0.40 / fig_h_in,
        "Median distance (metres) to nearest service, by morphological type "
        "within population density quintiles.  \u0394 = form gap.",
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
    classified, dq_stats = load_data()

    print("\nBuilding Plate 11 — Same Density, Different City...")
    fig = build_figure(classified, dq_stats)
    out = OUTPUT_DIR / "plate11_density_form.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)
    print("\nDone.")
