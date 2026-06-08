#!/usr/bin/env python3
"""Generate supplementary tables and figures for building source composition analysis.

Reads the building source audit CSVs (from s05a) and produces:

Tables:
    table_building_source_country.tex   — country-level source composition
    table_building_metric_sensitivity.tex — metric sensitivity to ML fraction

Figures:
    fig_building_source_map.pdf         — choropleth of ML fraction per city
    fig_building_source_sensitivity.pdf — within-city effect sizes

Examples:
    uv run python paper_data/code/s05c_building_source_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import BOUNDS_PATH, CSV_DIR, FIG_DIR, MAP_EXTENT, TABLE_DIR
from matplotlib.colors import BoundaryNorm

# Atlas-consistent style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper_research" / "code"))
from atlas_common import BG, DARK, GREY, apply_atlas_style

apply_atlas_style()

NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"


# ============================================================================
# 1. Country source composition table
# ============================================================================


def make_country_table():
    """Generate LaTeX table of country-level building source composition."""
    df = pd.read_csv(CSV_DIR / "building_source_by_country.csv")
    df = df.sort_values("overall_ml_fraction", ascending=True)

    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"  \toprule",
        r"  \textbf{Country} & \textbf{Cities} & \textbf{Buildings} & \textbf{ML (\%)} & \textbf{Min (\%)} & \textbf{Max (\%)} \\",
        r"  \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"  {row['country']} & {int(row['n_cities'])} & {int(row['total_buildings']):,} "
            f"& {row['overall_ml_fraction']*100:.1f} "
            f"& {row['min_ml_fraction']*100:.1f} "
            f"& {row['max_ml_fraction']*100:.1f} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])

    out = TABLE_DIR / "table_building_source_country.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote: {out}")


# ============================================================================
# 2. Metric sensitivity table
# ============================================================================


def make_sensitivity_table():
    """Generate LaTeX table of metric sensitivity to ML fraction."""
    corr = pd.read_csv(CSV_DIR / "building_source_correlations.csv")
    metrics = pd.read_csv(CSV_DIR / "building_source_metrics.csv")

    mw_cols = {
        c.replace("mw_r_", ""): c
        for c in metrics.columns
        if c.startswith("mw_r_")
    }

    lines = [
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"  \toprule",
        r"  \textbf{Metric} & \textbf{Cross-city $\rho$} & \textbf{$p$} & \textbf{Within-city $r$} & \\",
        r"  \midrule",
    ]
    for _, row in corr.iterrows():
        metric = row["metric"]
        rho = row["spearman_rho"]
        p = row["p_value"]
        mw_col = mw_cols.get(metric)
        if mw_col is not None:
            r_val = metrics[mw_col].median()
            r_str = f"{r_val:+.2f}"
        else:
            r_str = "---"
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        p_str = f"$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"  {metric.replace('_', ' ').title()} & {rho:+.3f} & {p_str} & {r_str} & {stars} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}"])

    out = TABLE_DIR / "table_building_metric_sensitivity.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote: {out}")


# ============================================================================
# 3. Source composition map (atlas plate style)
# ============================================================================


def make_source_map():
    """Generate choropleth map of ML fraction per city, atlas plate style.

    Follows atlas sizing conventions:
    - 7.5 × 5.5" (Elsevier text width × compact height)
    - BG background, white country fills, GREY 0.25pt borders
    - City centroids as round dots (not boundary polygons)
    - Dot legend top-left (matching plate2 city-list pattern)
    - 6pt description centred at bottom
    """
    bounds = gpd.read_file(BOUNDS_PATH, layer="bounds")
    source = pd.read_csv(CSV_DIR / "building_source_metrics.csv", dtype={"bounds_fid": str})
    bounds["bounds_fid"] = bounds["bounds_fid"].astype(str)
    merged = bounds.merge(source[["bounds_fid", "ml_fraction"]], on="bounds_fid")

    # European outline
    world = gpd.read_file(NE_URL)
    europe = world[world["CONTINENT"] == "Europe"].to_crs(3035)

    # ── Layout ──
    fig_w, fig_h = 7.5, 6.5
    title_h = 0.045  # top title band
    desc_h = 0.045  # bottom description band
    map_bot = desc_h
    map_h = 1.0 - title_h - map_bot

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Title (10pt bold, DARK, 0.08" from top — matching draw_title)
    y_title = 1.0 - 0.08 / fig_h
    fig.text(0.5, y_title, "Building Footprint Source Composition",
             fontsize=10, fontweight="bold", color=DARK, ha="center", va="top")

    ax = fig.add_axes([0.02, map_bot, 0.96, map_h])
    ax.set_facecolor(BG)

    # Country borders (white fill, grey edge — matching plate2)
    europe.plot(ax=ax, color="white", edgecolor=GREY, linewidth=0.25, alpha=0.5)

    # Discrete bins
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    labels_text = ["<5%", "5\u201310%", "10\u201320%", "20\u201330%", "30\u201350%", ">50%"]
    cmap = plt.cm.RdYlGn_r
    norm = BoundaryNorm(bins, cmap.N)

    # City centroids as circles
    centroids = merged.geometry.centroid
    colors = [cmap(norm(v)) for v in merged["ml_fraction"]]
    ax.scatter(
        centroids.x, centroids.y,
        s=12, c=colors, edgecolor="none", alpha=0.85, zorder=3,
    )

    ax.set_xlim(MAP_EXTENT["x_min"], MAP_EXTENT["x_max"])
    ax.set_ylim(MAP_EXTENT["y_min"], MAP_EXTENT["y_max"])
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Dot legend, top-left, overlaid on map ──
    # Positioned in figure fractions matching plate2's city-list legend.
    n_items = len(labels_text)
    leg_x = 0.04
    leg_top = map_bot + map_h - 0.03
    item_h = 0.032  # vertical step per item

    # Header
    fig.text(leg_x + 0.012, leg_top + 0.015,
             "ML-derived fraction", fontsize=7, fontweight="bold",
             color=DARK, va="bottom")

    # One small axes for the dots + labels
    leg_height = n_items * item_h + 0.01
    leg_ax = fig.add_axes([leg_x, leg_top - leg_height, 0.11, leg_height])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(-0.5, n_items - 0.5)
    leg_ax.set_facecolor("none")
    leg_ax.axis("off")

    for i, (lo, hi, label) in enumerate(zip(bins[:-1], bins[1:], labels_text)):
        yi = n_items - 1 - i  # top to bottom
        mid = (lo + hi) / 2
        color = cmap(norm(mid))
        leg_ax.scatter(0.08, yi, s=28, color=color, edgecolor="none", zorder=5)
        leg_ax.text(0.22, yi, label, fontsize=7, color=DARK, va="center")

    # ── Description, centred at bottom (6pt, DARK) ──
    fig.text(
        0.5, desc_h * 0.45,
        "ML-derived building footprint fraction across 626 SOAR-EU urban centres.",
        fontsize=6, color=DARK, ha="center", va="center",
    )

    out = FIG_DIR / "fig_building_source_map.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"Wrote: {out}")


# ============================================================================
# 4. Within-city effect sizes (single panel)
# ============================================================================


def make_sensitivity_figure():
    """Generate single-panel figure of within-city effect sizes."""
    source = pd.read_csv(CSV_DIR / "building_source_metrics.csv")

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    fig.set_facecolor(BG)

    effect_cols = {
        "Shared\nwall ratio": "mw_r_shared_wall_ratio",
        "Corners": "mw_r_corners",
        "Fractal\ndimension": "mw_r_fractal_dimension",
        "Compact-\nness": "mw_r_compactness",
        "Shape\nindex": "mw_r_shape_index",
    }

    positions = range(len(effect_cols))
    for i, (label, col) in enumerate(effect_cols.items()):
        vals = source[col].dropna()
        ax.boxplot(
            vals, positions=[i], widths=0.5, vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#d1e5f0", edgecolor=GREY, linewidth=0.8),
            medianprops=dict(color="#b2182b", linewidth=1.5),
            whiskerprops=dict(color=GREY, linewidth=0.8),
            capprops=dict(color=GREY, linewidth=0.8),
            flierprops=dict(marker=".", markersize=2, color="#999999", alpha=0.3),
        )

    ax.axhline(0, color=GREY, linewidth=0.5, linestyle="--")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(effect_cols.keys()), fontsize=6, color=DARK)
    ax.set_ylabel("Rank-biserial $r$ (community vs ML)", fontsize=7, color=DARK)
    ax.tick_params(labelsize=6, colors=DARK)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.text(
        0.98, 0.03, "Negative $r$: ML buildings have lower values",
        transform=ax.transAxes, fontsize=5.5, color=GREY, va="bottom", ha="right",
    )

    fig.tight_layout()
    out = FIG_DIR / "fig_building_source_sensitivity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote: {out}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating building source supplementary outputs...\n")
    make_country_table()
    make_sensitivity_table()
    make_source_map()
    make_sensitivity_figure()
    print("\nDone.")
