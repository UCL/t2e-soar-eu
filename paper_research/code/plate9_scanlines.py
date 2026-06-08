#!/usr/bin/env python3
"""Plate — Scan-line maps of Europe (combined).

Two stacked panels on a single 7.5" x 10" page. Line thickness and opacity are
interpolated from nearby city metric values using Gaussian-weighted
inverse-distance interpolation. Lines are drawn only over study-area land
(EU-27 + NO, LI, CH).

Top:    Tree canopy access (horizontal lines, green)
Bottom: Eat & Drink access (horizontal lines, red)
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    BG,
    BOUNDARIES_PATH,
    DARK,
    GREY,
    OUTPUT_DIR,
    apply_atlas_style,
    draw_title,
    load_all_cached,
)

apply_atlas_style()

# ── Map extent (EPSG:3035) — wide crop to fill page width ──
X_MIN, X_MAX = 2200000, 6400000
Y_MIN, Y_MAX = 1400000, 4200000

MIN_CITY_SEGMENTS = 500

# ── Study area countries (EU-27 + NO, LI, CH) ──
# Natural Earth has ISO_A2 = "-99" for France and Norway, so also match by NAME.
STUDY_ISO2 = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    "NO",
    "LI",
    "CH",
}
STUDY_NAMES = {"France", "Norway"}

# ── Metric definitions ──
PANELS = [
    {
        "col": "cc_trees_nearest_max_1600",
        "label": "Tree canopy access",
        "desc": (
            "City-level median nearest-distance to tree canopy (1600 m network radius). "
            "Gaussian-weighted spatial interpolation. Thicker lines = closer access."
        ),
        "color": "#1E8C4A",
        "invert": True,
        "orientation": "horizontal",
    },
    {
        "col": "cc_eat_and_drink_nearest_max_1600",
        "label": "Eat & drink access",
        "desc": (
            "City-level median nearest-distance to eat & drink (1600 m network radius). "
            "Gaussian-weighted spatial interpolation. Thicker lines = closer access."
        ),
        "color": "#d62728",
        "invert": True,
        "orientation": "horizontal",
    },
]

SEARCH_RADIUS = 60_000  # metres — interpolation radius (very tight)
DECAY_SIGMA = 12_000  # metres — Gaussian decay (very local, individual cities)
N_LINES = 150  # number of scan lines per panel
N_SAMPLES = 1200  # sample points per line (dense enough to resolve tight peaks)
LW_MIN = 0.05  # minimum line width — hairline far from cities
LW_MAX = 1.2  # maximum line width — bloom near cities
ALPHA_CONST = 0.75  # constant opacity — no fading, just thickness varies
CONTRAST_GAMMA = 1.5  # power curve > 1: thin baseline, gradual bloom near cities


def load_data():
    """Load city-level medians and centroids."""
    print("Loading segment data...")
    cols = [p["col"] for p in PANELS]
    df = load_all_cached(columns=cols)
    print(f"  {len(df):,} segments loaded")

    city_n = df.groupby("bounds_fid").size().rename("n_seg")
    city_meds = df.groupby("bounds_fid")[cols].median()
    city_meds = city_meds.join(city_n)
    city_meds = city_meds[city_meds["n_seg"] >= MIN_CITY_SEGMENTS].copy()
    city_meds = city_meds.dropna(how="all", subset=cols)
    print(f"  {len(city_meds)} cities with >= {MIN_CITY_SEGMENTS} segments")

    print("  Loading boundaries...")
    bounds_geo = gpd.read_file(
        BOUNDARIES_PATH,
    ).to_crs(3035)
    bounds_geo = bounds_geo.drop_duplicates("bounds_fid").set_index("bounds_fid")
    bounds_geo["cx"] = bounds_geo.geometry.centroid.x
    bounds_geo["cy"] = bounds_geo.geometry.centroid.y
    city_meds = city_meds.join(bounds_geo[["cx", "cy"]])
    city_meds = city_meds.dropna(subset=["cx", "cy"])
    print(f"  {len(city_meds)} cities with coordinates")

    return city_meds


def load_europe():
    """Load Natural Earth outlines: all Europe for borders, study area for land mask."""
    print("  Loading Natural Earth...")
    world = gpd.read_file("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip")
    europe_all = world[world["CONTINENT"] == "Europe"].to_crs(3035)
    mask = world["ISO_A2"].isin(STUDY_ISO2) | world["NAME"].isin(STUDY_NAMES)
    study = world[mask].to_crs(3035)
    print(f"  Study area: {len(study)} countries")
    return europe_all, study


def normalise_metric(values, invert=False):
    """Normalise values to [0, 1] using p2-p98 clipping."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values), 0.0, 0.0
    p2, p98 = np.nanpercentile(valid, [2, 98])
    if p98 == p2:
        return np.zeros_like(values), p2, p98
    normed = np.clip((values - p2) / (p98 - p2), 0, 1)
    if invert:
        normed = 1.0 - normed
    return normed, p2, p98


def _smooth_line(vals, sigma_samples=10):
    """Smooth values along a line, preserving NaN (water) gaps."""
    from scipy.ndimage import gaussian_filter1d

    land = ~np.isnan(vals)
    if land.sum() < 3:
        return vals
    # Smooth only land values, leaving NaN gaps intact
    smoothed = vals.copy()
    filled = np.where(land, vals, 0.0)
    weights = land.astype(float)
    s_filled = gaussian_filter1d(filled, sigma=sigma_samples, mode="constant", cval=0)
    s_weights = gaussian_filter1d(weights, sigma=sigma_samples, mode="constant", cval=0)
    valid = s_weights > 0.01
    smoothed[valid & land] = s_filled[valid & land] / s_weights[valid & land]
    return smoothed


# Fade smoothing: controls how gradually lines ramp up/down around cities.
# ~10 samples ≈ 35 km fade distance at 1200 samples across 4200 km.
SMOOTH_SIGMA_SAMPLES = 18


def interpolate_scanlines(city_coords, city_values, tree, orientation, search_radius, n_lines, n_samples, land_geom):
    """Compute interpolated scan-line data, masked to study-area land only."""
    lines = []

    if orientation == "horizontal":
        y_positions = np.linspace(Y_MIN, Y_MAX, n_lines)
        x_positions = np.linspace(X_MIN, X_MAX, n_samples)
        for y in y_positions:
            pts = np.column_stack([x_positions, np.full(n_samples, y)])
            vals = _interpolate_points(pts, city_coords, city_values, tree, search_radius)
            on_land = shapely.within(shapely.points(pts[:, 0], pts[:, 1]), land_geom)
            vals[~on_land] = np.nan
            vals = _smooth_line(vals, SMOOTH_SIGMA_SAMPLES)
            lines.append((pts, vals))
    else:
        x_positions = np.linspace(X_MIN, X_MAX, n_lines)
        y_positions = np.linspace(Y_MIN, Y_MAX, n_samples)
        for x in x_positions:
            pts = np.column_stack([np.full(n_samples, x), y_positions])
            vals = _interpolate_points(pts, city_coords, city_values, tree, search_radius)
            on_land = shapely.within(shapely.points(pts[:, 0], pts[:, 1]), land_geom)
            vals[~on_land] = np.nan
            vals = _smooth_line(vals, SMOOTH_SIGMA_SAMPLES)
            lines.append((pts, vals))

    return lines


def _interpolate_points(pts, city_coords, city_values, tree, search_radius):
    """Batch Gaussian-weighted interpolation with distance decay."""
    n = len(pts)
    result = np.zeros(n)
    neighbours = tree.query_ball_point(pts, r=search_radius)
    for i, nbrs in enumerate(neighbours):
        if len(nbrs) == 0:
            result[i] = 0.0
            continue
        dists = np.linalg.norm(city_coords[nbrs] - pts[i], axis=1)
        weights = np.exp(-0.5 * (dists / DECAY_SIGMA) ** 2)
        vals = city_values[nbrs]
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            result[i] = 0.0
            continue
        result[i] = np.average(vals[mask], weights=weights[mask])
    return result


def draw_scanlines(ax, lines_data, color, lw_min, lw_max):
    """Draw scan lines with varying width, constant colour; skip water (NaN)."""
    r, g, b = plt.matplotlib.colors.to_rgb(color)
    base_color = (r, g, b, ALPHA_CONST)
    for pts, vals in lines_data:
        if np.all(np.isnan(vals)):
            continue
        n = len(pts)
        segments = np.zeros((n - 1, 2, 2))
        segments[:, 0, :] = pts[:-1]
        segments[:, 1, :] = pts[1:]
        land = ~np.isnan(vals[:-1]) & ~np.isnan(vals[1:])
        if land.sum() == 0:
            continue
        seg_vals = (np.nan_to_num(vals[:-1]) + np.nan_to_num(vals[1:])) / 2.0
        seg_vals = np.power(seg_vals, CONTRAST_GAMMA)
        lws = lw_min + (lw_max - lw_min) * seg_vals
        lc = mcoll.LineCollection(
            segments[land],
            linewidths=lws[land],
            colors=[base_color] * land.sum(),
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
        ax.add_collection(lc)


def _draw_panel(fig, spec, city_meds, city_coords, tree, europe_all, land_union, map_rect, bar_rect, label_y, desc_y):
    """Draw one scanline panel (map + left bar + label + description)."""
    map_left, map_bottom, map_w, map_h = map_rect
    bar_left, bar_bottom, bar_w_ax, bar_h = bar_rect

    print(f"\n  {spec['label']} ({spec['orientation']})")

    # Panel label
    fig.text(
        map_left + map_w / 2,
        label_y,
        spec["label"],
        fontsize=8,
        fontweight="bold",
        color=DARK,
        ha="center",
        va="center",
    )

    # Map axes
    ax = fig.add_axes([map_left, map_bottom, map_w, map_h])
    ax.set_facecolor(BG)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # European outline for geographic context
    europe_all.plot(ax=ax, color="white", edgecolor="#999999", linewidth=0.3, alpha=0.6, zorder=1)

    # Normalise
    raw_vals = city_meds[spec["col"]].values.copy()
    normed, p2, p98 = normalise_metric(raw_vals, invert=spec["invert"])

    # Interpolate (only over study-area land)
    t1 = time.time()
    lines_data = interpolate_scanlines(
        city_coords,
        normed,
        tree,
        orientation=spec["orientation"],
        search_radius=SEARCH_RADIUS,
        n_lines=N_LINES,
        n_samples=N_SAMPLES,
        land_geom=land_union,
    )
    print(f"  Interpolation: {time.time() - t1:.1f}s")

    # Draw scan lines
    draw_scanlines(ax, lines_data, spec["color"], LW_MIN, LW_MAX)

    # ── Left bar: vertical line with same bloom effect as scanlines ──
    # For horizontal scanlines: each line is at a fixed Y
    y_arr = np.array([pts[0, 1] for pts, _ in lines_data])
    mean_normed = np.array(
        [np.nanmean(vals[~np.isnan(vals)]) if np.any(~np.isnan(vals)) else 0.0 for _, vals in lines_data]
    )
    raw_means = p2 + (1.0 - mean_normed) * (p98 - p2) if spec["invert"] else p2 + mean_normed * (p98 - p2)

    bar_ax = fig.add_axes([bar_left, bar_bottom, bar_w_ax, bar_h])
    bar_ax.set_facecolor(BG)
    bar_ax.set_ylim(Y_MIN, Y_MAX)
    bar_ax.set_xlim(0, 1)
    bar_ax.axis("off")

    # Vertical line segments with same gamma-driven bloom, constant colour
    n_bar = len(y_arr)
    bar_x = 0.35
    bar_segments = np.zeros((n_bar - 1, 2, 2))
    bar_segments[:, 0, 0] = bar_x
    bar_segments[:, 0, 1] = y_arr[:-1]
    bar_segments[:, 1, 0] = bar_x
    bar_segments[:, 1, 1] = y_arr[1:]
    bar_seg_vals = (mean_normed[:-1] + mean_normed[1:]) / 2.0
    bar_seg_vals = np.power(bar_seg_vals, CONTRAST_GAMMA)
    r, g, b = plt.matplotlib.colors.to_rgb(spec["color"])
    bar_lws = LW_MIN + (LW_MAX - LW_MIN) * bar_seg_vals
    bar_color = (r, g, b, ALPHA_CONST)
    bar_lc = mcoll.LineCollection(
        bar_segments,
        linewidths=bar_lws,
        colors=[bar_color] * len(bar_seg_vals),
        capstyle="butt",
        joinstyle="round",
    )
    bar_ax.add_collection(bar_lc)

    # Tick labels to the right of the bar line
    tick_ys = np.linspace(Y_MIN, Y_MAX, 9)
    for ty in tick_ys[1:-1]:
        idx = np.argmin(np.abs(y_arr - ty))
        val_m = raw_means[idx]
        bar_ax.text(0.75, ty, f"{val_m:.0f} m", fontsize=5.5, fontweight="bold", color=GREY, ha="left", va="center")

    # Description text below the panel
    map_left, map_bottom, map_w, _ = map_rect
    fig.text(
        map_left + map_w / 2, desc_y, spec["desc"], fontsize=6, color=DARK, ha="center", va="center", linespacing=1.3
    )


def build_combined_plate(city_meds, city_coords, tree, europe_all, land_union):
    """Build a single plate with both scanline panels stacked vertically."""
    fig_w, fig_h = 7.5, 10.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Horizontal layout
    left_margin = 0.01
    right_margin = 0.99
    bar_w = 0.05
    bar_gap = 0.005
    map_left = left_margin + bar_w + bar_gap
    map_w = right_margin - map_left

    # Vertical layout (figure fractions)
    title_h = 0.035
    label_h = 0.025
    desc_h = 0.04  # description text below each panel
    gap_h = 0.015
    bottom_pad = 0.012
    panel_h = (1.0 - title_h - 2 * (label_h + desc_h) - gap_h - bottom_pad) / 2.0

    # Top panel positions (from top down)
    top_panel_top = 1.0 - title_h - label_h
    top_panel_bot = top_panel_top - panel_h
    top_label_y = top_panel_top + label_h * 0.4
    top_desc_y = top_panel_bot - desc_h * 0.5

    # Bottom panel positions
    bot_panel_top = top_panel_bot - desc_h - gap_h - label_h
    bot_panel_bot = bot_panel_top - panel_h
    bot_label_y = bot_panel_top + label_h * 0.4
    bot_desc_y = bot_panel_bot - desc_h * 0.5

    # Title
    draw_title(fig, "Continental Access Gradients", y_inches=0.08)

    # Top panel (green space)
    _draw_panel(
        fig,
        PANELS[0],
        city_meds,
        city_coords,
        tree,
        europe_all,
        land_union,
        map_rect=(map_left, top_panel_bot, map_w, panel_h),
        bar_rect=(left_margin, top_panel_bot, bar_w, panel_h),
        label_y=top_label_y,
        desc_y=top_desc_y,
    )

    # Bottom panel (eat & drink)
    _draw_panel(
        fig,
        PANELS[1],
        city_meds,
        city_coords,
        tree,
        europe_all,
        land_union,
        map_rect=(map_left, bot_panel_bot, map_w, panel_h),
        bar_rect=(left_margin, bot_panel_bot, bar_w, panel_h),
        label_y=bot_label_y,
        desc_y=bot_desc_y,
    )

    return fig


if __name__ == "__main__":
    t0 = time.time()
    print("Building combined scan-line plate...")
    city_meds = load_data()
    europe_all, study = load_europe()

    city_coords = city_meds[["cx", "cy"]].values.copy()
    tree = cKDTree(city_coords)

    print("  Building study-area land union...")
    land_union = shapely.union_all(study.geometry.values)
    shapely.prepare(land_union)

    fig = build_combined_plate(city_meds, city_coords, tree, europe_all, land_union)
    out = OUTPUT_DIR / "plate9_scanlines.png"
    fig.savefig(out, dpi=450, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")
