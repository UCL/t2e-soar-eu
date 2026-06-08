#!/usr/bin/env python3
"""Plate 4 — Building and Block Form by Morphological Type.

Deviation-spine (lollipop) figure: 8 panels (one per octant type, 2×4 grid),
each showing how that octant's median deviates from the European median on
building morphology metrics.  Z-scored using MAD for robustness.  The most
extreme deviation in each panel is annotated with its real-unit value.

Inspired by the Continental Fingerprints figure (archive).
"""

import sys
from pathlib import Path

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
    SYM_INCHES,
    apply_atlas_style,
    classify_octants,
    draw_legend_row,
    draw_title,
    load_all_cached,
    load_hq_fids,
    place_composite_symbol,
    standard_margins,
)

# ============================================================================
# METRICS — building & block morphology at 400m (counts non-weighted, rest distance-weighted)
# ============================================================================

METRICS = [
    # (column, short_label, colour, unit_format)
    # ── Building ──
    ("cc_building_400_nw", "Count", "#8c564b", "{:.0f}"),
    ("cc_area_median_400_wt", "Area", "#A0522D", "{:.0f}"),
    ("cc_volume_median_400_wt", "Volume", "#2F4F4F", "{:.0f}"),
    ("cc_mean_height_median_400_wt", "Height", "#D4764E", "{:.1f}"),
    ("cc_mean_height_mad_400_wt", "Height MAD", "#4878CF", "{:.1f}"),
    ("cc_perimeter_median_400_wt", "Perimeter", "#6B8E23", "{:.0f}"),
    ("cc_corners_median_400_wt", "Corners", "#556B2F", "{:.0f}"),
    ("cc_shared_wall_ratio_median_400_wt", "Shared walls", "#E74C3C", "{:.2f}"),
    ("frontage_max", "Frontage", "#C0392B", "{:.2f}"),
    ("cc_orientation_mad_400_wt", "Orient. MAD", "#D4AC0D", "{:.1f}"),
    # ── Block ──
    ("cc_block_400_nw", "Count", "#228B22", "{:.0f}"),
    ("cc_block_far_median_400_wt", "FSI", "#A8201A", "{:.2f}"),
    ("cc_block_covered_ratio_median_400_wt", "GSI", "#B8600A", "{:.2f}"),
    ("cc_block_l_median_400_wt", "Floors", "#6C3483", "{:.1f}"),
    ("cc_block_osr_median_400_wt", "OSR", "#2E86C1", "{:.1f}"),
    ("cc_block_perimeter_median_400_wt", "Block perim.", "#1E8449", "{:.0f}"),
    # ── Mixed use ──
    ("cc_hill_q0_400_nw", "Richness 400m", "#E67E22", "{:.1f}"),
    ("cc_hill_q1_400_nw", "Diversity 400m", "#D35400", "{:.1f}"),
    # ── Network ──
    ("cc_density_400", "Streets 400m", "#7f7f7f", "{:.0f}"),
    ("cc_density_800", "Streets 800m", "#8f8f8f", "{:.0f}"),
    ("cc_density_1200", "Streets 1200m", "#9f9f9f", "{:.0f}"),
]

# Subheading positions: (metric index, label)
METRIC_GROUPS = [
    (0, "BUILDING"),
    (10, "BLOCK"),
    (16, "MIXED USE"),
    (18, "NETWORK"),
]

# ============================================================================
# MAIN
# ============================================================================


def build_plate(df, out_path, subtitle_extra=""):
    """Build plate 4 from pre-loaded data, saving to out_path."""

    n_cities = df["bounds_fid"].nunique() if "bounds_fid" in df.columns else "?"
    print(f"  {len(df):,} street segments from {n_cities} cities")

    classified, thresholds = classify_octants(df)
    print(f"  {len(classified):,} classified segments")

    # Keep only columns we need.
    metric_cols = [m[0] for m in METRICS]
    keep = ["octant"] + [c for c in metric_cols if c in classified.columns]
    classified = classified[keep].copy()

    # Compute European median and MAD per metric.
    eu_medians = {}
    eu_mads = {}
    for col, *_ in METRICS:
        if col not in classified.columns:
            continue
        vals = classified[col].dropna().values
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad == 0:
            mad = np.nanstd(vals)
        if mad == 0:
            mad = 1.0
        eu_medians[col] = med
        eu_mads[col] = mad

    # Compute per-octant medians.
    octant_medians = classified.groupby("octant")[metric_cols].median()

    # Available metrics (some columns may be missing).
    avail_metrics = [(col, label, color, ufmt) for col, label, color, ufmt in METRICS if col in octant_medians.columns]
    n_metrics = len(avail_metrics)

    # Build display rows: interleave subheadings with metrics at regular spacing.
    # Each row is either ("metric", index) or ("heading", label).
    group_start_indices = {idx for idx, _ in METRIC_GROUPS}
    display_rows = []
    for i, m in enumerate(avail_metrics):
        orig_idx = METRICS.index(m) if m in METRICS else i
        if orig_idx in group_start_indices:
            label = next(lbl for idx, lbl in METRIC_GROUPS if idx == orig_idx)
            display_rows.append(("heading", label))
        display_rows.append(("metric", i))

    n_display = len(display_rows)
    EXTRA_GAP = 0.6  # extra space before non-first group headings

    # Build y positions with extra gap before subsequent headings.
    y_all = np.zeros(n_display)
    y = 0.0
    heading_count = 0
    for di, (kind, _val) in enumerate(display_rows):
        if kind == "heading":
            if heading_count > 0:
                y += EXTRA_GAP
            heading_count += 1
        y_all[di] = y
        y += 1.0

    # Map metric indices to their y positions.
    y_positions = np.zeros(n_metrics)
    subheading_positions = []
    for di, (kind, val) in enumerate(display_rows):
        if kind == "metric":
            y_positions[val] = y_all[di]
        else:
            subheading_positions.append((y_all[di], val))
    total_y = y_all[-1]

    # Z-score each octant's median relative to the European median.
    z_scores = {}
    for octant in OCTANT_ORDER:
        zs = []
        for col, *_ in avail_metrics:
            raw = octant_medians.loc[octant, col]
            z = (raw - eu_medians[col]) / eu_mads[col]
            zs.append(z)
        z_scores[octant] = zs

    # ── Figure: 2 rows × 4 cols (intense top, light bottom) ────────────
    nrows, ncols = 2, 4
    # Reorder: intense (H**) in top row, light (L**) in bottom row.
    row_order = [
        ["HHH", "HHL", "HLH", "HLL"],  # Intense
        ["LHH", "LHL", "LLH", "LLL"],  # Light
    ]

    7.5 / ncols  # 1.875" per panel
    panel_h = 3.5
    fig = plt.figure(
        figsize=(7.5, min(10.0, nrows * panel_h + 2.6)),
        facecolor=BG,
    )

    max_z = 4.0
    annot_buf = 0.35  # buffer between dot and annotation text

    for ri, row_octants in enumerate(row_order):
        for ci, octant in enumerate(row_octants):
            ax = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1)
            ax.set_facecolor(BG)

            values = [np.clip(z, -max_z, max_z) for z in z_scores[octant]]
            octant_color = OCTANT_COLORS[octant]

            # Zero reference line — clipped to metric range (first to last metric)
            y_top = y_positions[0] - 0.5
            y_bot = y_positions[-1] + 0.5
            ax.plot([0, 0], [y_top, y_bot], color=octant_color, linewidth=0.4, alpha=1.0, zorder=0)

            # Faint background shading — clipped to metric range
            ax.fill_between([0, max_z], y_top, y_bot, color="#E8F5E9", alpha=0.12, zorder=0)
            ax.fill_between([-max_z, 0], y_top, y_bot, color="#FFF3E0", alpha=0.10, zorder=0)

            # Profile contour (thin connecting line).
            ax.plot(values, y_positions, color="#C0C0C0", linewidth=0.4, alpha=0.3, zorder=1)

            # Lollipop stems and dots with value annotations.
            for j, val in enumerate(values):
                yj = y_positions[j]
                ax.plot(
                    [0, val],
                    [yj, yj],
                    color=octant_color,
                    linewidth=1.4,
                    alpha=0.7,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.scatter(
                    val,
                    yj,
                    s=24,
                    color=octant_color,
                    zorder=3,
                    edgecolors="white",
                    linewidth=0.4,
                )
                # Annotate every lollipop with real value.
                col_j, _, _, ufmt_j = avail_metrics[j]
                raw_val = octant_medians.loc[octant, col_j]
                if not np.isnan(raw_val):
                    annot_text = ufmt_j.format(raw_val)
                    if val >= 0:
                        ax.text(
                            val + annot_buf,
                            yj,
                            annot_text,
                            fontsize=5,
                            color=octant_color,
                            ha="left",
                            va="center",
                            zorder=4,
                        )
                    else:
                        ax.text(
                            val - annot_buf,
                            yj,
                            annot_text,
                            fontsize=5,
                            color=octant_color,
                            ha="right",
                            va="center",
                            zorder=4,
                        )

            # Metric labels on left (only first column).
            ax.set_yticks(y_positions)
            if ci == 0:
                ax.set_yticklabels(
                    [label for _, label, _, _ in avail_metrics],
                    fontsize=6,
                    fontweight="normal",
                    color=GREY,
                )
                # Group subheadings — right-aligned with tick labels
                for sy, slabel in subheading_positions:
                    ax.annotate(
                        slabel,
                        xy=(0, sy),
                        xycoords=("axes fraction", "data"),
                        fontsize=7,
                        fontweight="bold",
                        color=DARK,
                        ha="right",
                        va="center",
                        xytext=(-3, 0),
                        textcoords="offset points",
                    )
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=3)

            ax.set_xlim(-max_z - 1.0, max_z + 1.0)  # buffer for annotations
            ax.set_ylim(-1.0, total_y + 0.5)
            ax.invert_yaxis()
            ax.set_xticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_title("", pad=10)  # reserve space for symbols

    # Figure title at top.
    title = "Building and Block Form by Morphological Type"
    draw_title(fig, title)

    fig_h = fig.get_size_inches()[1]
    margins = standard_margins(fig)
    margins["left"] = 0.85 / 7.5  # 0.85" for metric labels + left padding
    margins["right"] = 1.0 - 0.15 / 7.5  # 0.15" right padding
    margins["top"] = 1.0 - 0.75 / fig_h  # 0.75" from top for title + vertical symbols
    margins["bottom"] = 0.85 / fig_h  # 0.85" from bottom for subtitle + legend
    plt.subplots_adjust(
        hspace=0.18,
        wspace=0.05,
        **margins,
    )

    # Force layout so transData is accurate, then place symbols.
    fig.canvas.draw()

    # Vertically stacked composite symbols (matching Plate 4)
    # Position just above the clipped spine top (y_positions[0] - 0.5 in data)
    spine_top_data = y_positions[0] - 0.5
    for ri, row_octants in enumerate(row_order):
        for ci, octant in enumerate(row_octants):
            ax = fig.axes[ri * ncols + ci]
            # Use transData to correctly handle inverted y-axis
            display_xy = ax.transData.transform((0, spine_top_data))
            fig_xy = fig.transFigure.inverted().transform(display_xy)
            cx = fig.axes[ri * ncols + ci].get_position().x0 + fig.axes[ri * ncols + ci].get_position().width / 2
            # Offset by composite half-height so bottom sub-symbol
            # sits just above the spine top
            fig_h_in = fig.get_size_inches()[1]
            composite_half_h = (SYM_INCHES * 1.3 + SYM_INCHES * 0.5) / fig_h_in
            cy = fig_xy[1] + composite_half_h + 0.005
            place_composite_symbol(fig, cx, cy, octant)  # default SYM_INCHES

    # Subtitle above legend
    fig_h = fig.get_size_inches()[1]
    fig.text(
        0.5,
        0.45 / fig_h,
        f"Deviation from the European median, normalised by median absolute deviation. Numbers show median values.{(' ' + subtitle_extra + '.') if subtitle_extra else ''}",
        fontsize=6,
        color=DARK,
        ha="center",
        va="bottom",
    )

    # ── Legend row (shared function) ─────────────────────────────
    draw_legend_row(fig)

    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def main():
    apply_atlas_style()

    # Load all cached cities.
    print("Loading cached data …")
    needed = [m[0] for m in METRICS] + list(AXIS_COLS.values())
    df = load_all_cached(columns=needed)

    # HQ subset (<10% ML) — primary plate
    hq_fids = load_hq_fids(threshold=0.10)
    df_hq = df[df["bounds_fid"].isin(hq_fids)].copy()
    print(f"\n── HQ subset ({len(hq_fids)} cities, <10% ML) — primary ──")
    build_plate(df_hq, OUTPUT_DIR / "plate4_buildings.pdf", subtitle_extra=f"{len(hq_fids)} cities with ≥90% cadastral building sources")

    # Full dataset — supplementary
    print("\n── Full dataset — supplementary ──")
    build_plate(df, OUTPUT_DIR / "plate4_buildings_full.pdf", subtitle_extra="all 626 cities")


if __name__ == "__main__":
    main()
