#!/usr/bin/env python3
"""Experiment 14: Age Nautilus — sorted street values as radial fans.

Three rows: <15, Working age, 65+
Eight columns: one per octant
Each cell: sorted street values as radial lines (nautilus fan style).
Grey outline for the European-wide distribution as reference.
Consistent normalisation per row across all octants.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
    GREY,
    OCTANT_COLORS,
    OCTANT_ORDER,
    OUTPUT_DIR,
    apply_atlas_style,
    classify_octants,
    load_all_cached,
)

apply_atlas_style()

MAX_SAMPLE = 3000

DEMOGRAPHIC_ROWS = [
    {"col": "density", "name": "Pop. density", "color": "#b2182b", "log": True},
    {"col": "emp_%", "name": "Employment", "color": "#2166ac"},
    {"col": "y_lt15_%", "name": "Age < 15", "color": "#66c2a5"},
    {"col": "y_1564_%", "name": "Working age", "color": "#fc8d62"},
    {"col": "y_ge65_%", "name": "Age 65+", "color": "#8da0cb"},
]


def main():
    print("Loading data...")
    needed = list(AXIS_COLS.values()) + [g["col"] for g in DEMOGRAPHIC_ROWS]
    df = load_all_cached(columns=needed)
    classified, _ = classify_octants(df)
    print(f"  {len(classified):,} classified segments")

    n_rows = len(DEMOGRAPHIC_ROWS)  # 5
    n_cols = len(OCTANT_ORDER)  # 8

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.5, n_rows * 1.15 + 1.5),
        subplot_kw={"projection": "polar"},
        facecolor=BG,
    )

    rng = np.random.default_rng(42)

    for ri, ag in enumerate(DEMOGRAPHIC_ROWS):
        col = ag["col"]
        ag["color"]
        use_log = ag.get("log", False)

        # Global stats
        all_vals = classified[col].dropna().values
        if col.endswith("_%"):
            all_vals = np.clip(all_vals, 0, 1)
        if use_log:
            all_vals = np.log1p(np.maximum(all_vals, 0))
        global_max = np.nanpercentile(all_vals, 99)
        if global_max == 0:
            global_max = 1
        eu_median = np.nanmedian(all_vals)
        med_r = eu_median / global_max

        # EU reference sorted sample
        eu_sample = rng.choice(all_vals, min(MAX_SAMPLE, len(all_vals)), replace=False)
        eu_sorted = np.sort(eu_sample)
        eu_radii = eu_sorted / global_max
        eu_n = len(eu_radii)
        eu_angles = np.linspace(0, 2 * np.pi, eu_n, endpoint=False)
        np.append(eu_angles, eu_angles[0])
        np.append(eu_radii, eu_radii[0])

        for ci, octant in enumerate(OCTANT_ORDER):
            ax = axes[ri, ci]
            ax.set_facecolor(BG)
            ax.set_theta_zero_location("N")
            oc = OCTANT_COLORS[octant]

            oct_vals = classified.loc[classified["octant"] == octant, col].dropna().values
            if col.endswith("_%"):
                oct_vals = np.clip(oct_vals, 0, 1)
            if use_log:
                oct_vals = np.log1p(np.maximum(oct_vals, 0))
            if len(oct_vals) < 20:
                ax.set_visible(False)
                continue

            if len(oct_vals) > MAX_SAMPLE:
                oct_vals = rng.choice(oct_vals, MAX_SAMPLE, replace=False)

            vals_sorted = np.sort(oct_vals)
            radii = vals_sorted / global_max
            n = len(radii)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

            # (EU reference removed — colour change communicates above/below median)

            # Octant radial lines — coloured above median, grey below
            step = max(1, n // 150)
            for j in range(0, n, step):
                if radii[j] >= med_r:
                    lc = oc
                    la = 0.4
                else:
                    lc = GREY
                    la = 0.15
                ax.plot([angles[j], angles[j]], [0, radii[j]], color=lc, lw=0.15, alpha=la)

            # Octant envelope
            angles_closed = np.append(angles, angles[0])
            radii_closed = np.append(radii, radii[0])
            ax.plot(angles_closed, radii_closed, color=oc, lw=0.6, alpha=0.8)

            # Octant median value — below the plot
            oct_median = np.nanmedian(vals_sorted)
            if use_log:
                # Convert back from log for display
                oct_median = np.expm1(oct_median)
            if oct_median >= 1000:
                label = f"{oct_median:,.0f}"
            elif oct_median >= 1:
                label = f"{oct_median:.1f}"
            elif col.endswith("_%"):
                label = f"{oct_median:.0%}"
            else:
                label = f"{oct_median:.2f}"
            ax.text(
                0.5,
                -0.05,
                label,
                fontsize=6,
                fontweight="bold",
                color=oc,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )

            ax.set_ylim(0, 1.35)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(alpha=0.15)
            for sp in ax.spines.values():
                sp.set_visible(False)

    # Layout
    from atlas_common import standard_margins

    fig_h = fig.get_size_inches()[1]
    margins = standard_margins(fig)
    margins["top"] = 1.0 - 0.8 / fig_h  # 0.8" from top for title + composite symbols
    margins["bottom"] = 0.65 / fig_h  # 0.65" for numbers + legend + description
    margins["left"] = 0.55 / 7.5  # 0.55" for rotated row labels
    plt.subplots_adjust(hspace=0.0, wspace=0.05, **margins)
    fig.canvas.draw()

    # Column headers: vertically stacked composite symbols (matching Plates 3 & 4)
    from atlas_common import place_composite_symbol

    for ci, octant in enumerate(OCTANT_ORDER):
        ax_top = axes[0, ci]
        bbox = ax_top.get_position()
        cx = bbox.x0 + bbox.width / 2
        cy = bbox.y1 + 0.04
        place_composite_symbol(fig, cx, cy, octant)

    # Row labels — close to the plots
    for ri, ag in enumerate(DEMOGRAPHIC_ROWS):
        ax_left = axes[ri, 0]
        bbox = ax_left.get_position()
        cy = bbox.y0 + bbox.height / 2
        fig.text(
            bbox.x0 - 0.01,
            cy,
            ag["name"],
            fontsize=7,
            fontweight="bold",
            color=DARK,
            rotation=90,
            va="center",
            ha="right",
        )

    from atlas_common import draw_title

    draw_title(fig, "Demographics by Morphological Type")
    # ── Legend row (shared function) ─────────────────────────────
    from atlas_common import draw_legend_row

    draw_legend_row(fig)

    fig.text(
        0.5,
        0.065,
        "Sorted street-level values. Colour = above European median. Grey = below.",
        fontsize=6,
        color=DARK,
        ha="center",
    )

    out = OUTPUT_DIR / "plate8_demographics.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
