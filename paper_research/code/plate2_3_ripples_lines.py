"""
Plate 2/3 combined — Ripple Map (top) + Stacked Lines (bottom).

Top half: concentric ripple map showing morphological character across Europe.
Bottom half: thin stacked lines per city, grouped by country (north → south),
hierarchically clustered within each country.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
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

MIN_CITY_SEGMENTS = 2000
MIN_COUNTRY_SEGMENTS = 5000

# Map extent (EPSG:3035)
X_MIN, X_MAX = 2500000, 5900000
Y_MIN, Y_MAX = 1300000, 4300000

AXIS_ORDER = ["intensity", "continuity", "irregularity"]
FIXED_COLORS = {
    "intensity": "#b2182b",
    "continuity": "#2166ac",
    "irregularity": "#1a9641",
}

LINE_COLORS = {
    "intensity": "#b2182b",
    "continuity": "#2166ac",
    "irregularity": "#1a9641",
}


def _ring_radius(v, spacing, ring_index, n_rings=3):
    """Ring radius: outer ring fixed, inner rings scale inward by value.

    ring_index 0 = innermost (lowest value), n_rings-1 = outermost (highest).
    Outer ring is always at n_rings * spacing.
    Inner rings shrink toward centre for low values, approach outer for high values.
    Minimum gap of 0.3 * spacing between adjacent rings.
    """
    r_outer = n_rings * spacing
    # How far inward from outer ring this ring sits
    slots_from_outer = n_rings - 1 - ring_index
    # At v=1: ring is close to outer (gap = 0.3*spacing per slot)
    # At v=0: ring collapses toward centre (gap = 0.9*spacing per slot)
    gap_per_slot = spacing * (0.5 + 0.5 * (1 - v))
    return r_outer - slots_from_outer * gap_per_slot


def _ring_alpha(v, alpha_max):
    """Ring opacity from normalised value.

    Mild gamma (v**0.6) lifts mid-range values so they remain legible without
    raising the ceiling for high-value cities. Shared by plates 2 and 3 so the
    two panels agree visually.
    """
    return alpha_max * v**0.6


def load_data():
    print("Loading data...")
    df = load_all_cached(columns=list(AXIS_COLS.values()))
    bounds = gpd.read_file(
        BOUNDARIES_PATH,
        columns=["bounds_fid", "label", "country"],
    )
    bounds_geo = gpd.read_file(
        BOUNDARIES_PATH,
    ).to_crs(3035)
    bounds_geo["cx"] = bounds_geo.geometry.centroid.x
    bounds_geo["cy"] = bounds_geo.geometry.centroid.y

    df = df.merge(bounds[["bounds_fid", "label", "country"]], on="bounds_fid")

    # Continuity uses max(left, right)
    if "frontage_left" in df.columns and "frontage_right" in df.columns:
        df["_fr_max"] = df[["frontage_left", "frontage_right"]].max(axis=1)
    else:
        df["_fr_max"] = df[AXIS_COLS["continuity"]]

    axis_cols = [AXIS_COLS["intensity"], "_fr_max", AXIS_COLS["irregularity"]]
    city_n = df.groupby("bounds_fid").size().rename("n_seg")
    city_meds = df.groupby("bounds_fid")[axis_cols].median()
    city_meds = city_meds.join(city_n)
    city_meds = city_meds[city_meds["n_seg"] >= MIN_CITY_SEGMENTS].copy()
    city_meds = city_meds.join(bounds.drop_duplicates("bounds_fid").set_index("bounds_fid")[["label", "country"]])
    city_meds.columns = ["intensity", "continuity", "irregularity", "n_seg", "city", "country"]
    city_meds = city_meds.dropna(subset=["intensity", "continuity", "irregularity"])

    # Normalise: percentile rank, then remap so threshold = 0.75
    axis_thresholds = {"intensity": 1.0, "continuity": 0.75, "irregularity": 4.0}
    from scipy.stats import rankdata

    for col in AXIS_ORDER:
        vals = city_meds[col].values
        thresh = axis_thresholds[col]
        # Percentile rank (0–1)
        pct = rankdata(vals, method="average") / len(vals)
        # Find what percentile the threshold sits at
        thresh_pct = np.mean(vals <= thresh)
        # Remap: below threshold stretches [0, thresh_pct] → [0, 0.75]
        #         above threshold stretches [thresh_pct, 1] → [0.75, 1]
        normed = np.where(
            pct <= thresh_pct,
            0.75 * pct / thresh_pct if thresh_pct > 0 else 0.0,
            0.75 + 0.25 * (pct - thresh_pct) / (1 - thresh_pct) if thresh_pct < 1 else 1.0,
        )
        city_meds[col + "_n"] = np.clip(normed, 0, 1)

    # Add geometry info for ripple map
    bounds_dedup = bounds_geo.drop_duplicates("bounds_fid").set_index("bounds_fid")
    bounds_dedup["area_m2"] = bounds_dedup.geometry.area
    city_meds = city_meds.join(bounds_dedup[["cx", "cy", "area_m2"]])
    city_meds = city_meds.dropna(subset=["cx", "cy"])

    # Country filter for lines panel
    country_n = city_meds.groupby("country")["n_seg"].sum()
    valid_countries = country_n[country_n >= MIN_COUNTRY_SEGMENTS].index

    # Hierarchical clustering for line ordering
    cities_for_lines = city_meds[city_meds["country"].isin(valid_countries)].copy()
    all_vals = cities_for_lines[["intensity_n", "continuity_n", "irregularity_n"]].values
    global_dist = pdist(all_vals, metric="euclidean")
    global_Z = linkage(global_dist, method="ward", optimal_ordering=True)
    global_leaf_order = leaves_list(global_Z)
    global_rank = np.empty(len(cities_for_lines), dtype=int)
    for rank, leaf_idx in enumerate(global_leaf_order):
        global_rank[leaf_idx] = rank
    cities_for_lines["_global_rank"] = global_rank

    # Country order: north → south
    bg = bounds_geo.drop_duplicates("bounds_fid").set_index("bounds_fid")
    bg = bg[bg.index.isin(cities_for_lines.index)].copy()
    bg["country"] = cities_for_lines["country"]
    country_centroids = bg.dissolve(by="country").centroid
    country_lat = {
        c: country_centroids[c].y for c in cities_for_lines["country"].unique() if c in country_centroids.index
    }
    countries_ordered = sorted(country_lat, key=lambda c: -country_lat[c])

    max_cities = max(len(cities_for_lines[cities_for_lines["country"] == c]) for c in countries_ordered)

    # European outline
    print("  Loading Natural Earth...")
    world = gpd.read_file("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip")
    europe = world[world["CONTINENT"] == "Europe"].to_crs(3035)

    return city_meds, cities_for_lines, countries_ordered, max_cities, europe


def build_ripple_plate(city_meds, europe):
    """Build plate 2A: ripple map showing morphological character across Europe."""
    fig_w, fig_h = 7.5, 7.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Layout — compact bottom: legend, gap, description, map (no gap)
    title_h = 0.04
    bottom_pad = 0.005
    legend_h = 0.025  # legend row
    desc_h = 0.015  # description text
    gap_below = 0.012  # gap between legend and description
    map_bot = bottom_pad + legend_h + gap_below + desc_h
    map_h = 1.0 - title_h - map_bot

    draw_title(fig, "Three Morphological Axes Across European Cities", y_inches=0.08)

    map_pad_x = 0.02
    ax = fig.add_axes([map_pad_x, map_bot, 1.0 - 2 * map_pad_x, map_h])
    ax.set_facecolor(BG)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    europe.plot(ax=ax, color="white", edgecolor=GREY, linewidth=0.25, alpha=0.5)

    # Ripple parameters
    ripple_scale = 1.5
    _lw_min, _lw_max = 0.3, 0.65
    _alpha_min, alpha_max = 0.0, 0.9

    city_meds["eff_radius"] = np.sqrt(city_meds["area_m2"] / np.pi)
    r_vals = city_meds["eff_radius"].values
    r_min, r_max = r_vals.min(), r_vals.max()
    r_norm = (r_vals - r_min) / (r_max - r_min)
    r_scaled = 0.4 + 1.1 * np.sqrt(r_norm)
    city_meds["ring_spacing"] = r_scaled * r_max * ripple_scale / 3

    print(f"  Ripple map: {len(city_meds)} cities, 3 rings each")

    city_meds_sorted = city_meds.sort_values("eff_radius", ascending=False)
    theta = np.linspace(0, 2 * np.pi, 120)

    for ci, (_, city) in enumerate(city_meds_sorted.iterrows()):
        cx, cy = city["cx"], city["cy"]
        spacing = city["ring_spacing"]
        z = 2 + ci * 2

        max_v = max(city[ak + "_n"] for ak in AXIS_ORDER)
        r_outer = _ring_radius(max_v, spacing, 2) + spacing * 0.2
        ax.fill(cx + r_outer * np.cos(theta), cy + r_outer * np.sin(theta), color="white", alpha=0.20, zorder=z)

        axis_by_val = sorted(AXIS_ORDER, key=lambda ak: city[ak + "_n"])

        for ri, axis_key in enumerate(axis_by_val):
            v = city[axis_key + "_n"]
            r = _ring_radius(v, spacing, ri)
            alpha = _ring_alpha(v, alpha_max)
            # Inner rings (low-value axes) get a small line-weight bonus to
            # compensate for their smaller circumference. ri=0 is innermost.
            lw = 0.7 + 0.25 * (2 - ri) / 2
            color = FIXED_COLORS[axis_key]

            if lw < 0.02 or alpha < 0.02:
                continue

            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z + 1)

    # --- Largest cities legend overlaid on top-left of map ---
    n_legend_cities = 20
    top_cities = city_meds_sorted.head(n_legend_cities)

    # Position in figure fractions — top half, left side
    city_leg_top = map_bot + map_h - 0.02
    city_leg_bottom = map_bot + map_h * 0.45
    city_spacing_y = (city_leg_top - city_leg_bottom) / n_legend_cities
    ring_r_fig = 0.0105

    theta_leg = np.linspace(0, 2 * np.pi, 120)
    for ci, (_, city) in enumerate(top_cities.iterrows()):
        cy_pos = city_leg_top - ci * city_spacing_y
        leg_ax = fig.add_axes(
            [0.02 - ring_r_fig, cy_pos - ring_r_fig, ring_r_fig * 2, ring_r_fig * 2],
            aspect="equal",
        )
        leg_ax.set_xlim(-1.5, 1.5)
        leg_ax.set_ylim(-1.5, 1.5)
        leg_ax.set_facecolor("none")
        leg_ax.axis("off")

        leg_spacing = 0.45  # normalised spacing for legend rings
        axis_by_val = sorted(AXIS_ORDER, key=lambda ak: city[ak + "_n"])
        for ri, axis_key in enumerate(axis_by_val):
            v = city[axis_key + "_n"]
            r = _ring_radius(v, leg_spacing, ri)
            a = _ring_alpha(v, alpha_max)
            xs = r * np.cos(theta_leg)
            ys = r * np.sin(theta_leg)
            if a < 0.02:
                continue
            leg_lw = 0.7 + 0.25 * (2 - ri) / 2
            leg_ax.plot(xs, ys, color=FIXED_COLORS[axis_key], lw=leg_lw, alpha=a, solid_capstyle="round")

        label = city["city"]
        if "[" in label:
            label = label.split("[")[0].strip()
        fig.text(0.02 + ring_r_fig + 0.005, cy_pos, label, fontsize=5, color=DARK, ha="left", va="center")

    # Horizontal legend directly below map
    legend_items = [
        ("Intensity", "#b2182b"),
        ("Continuity", "#2166ac"),
        ("Irregularity", "#1a9641"),
    ]
    circle_r_fig = 0.009
    leg_y = bottom_pad + desc_h + gap_below + legend_h * 0.5

    # Description text at bottom
    fig.text(
        0.5,
        bottom_pad + desc_h * 0.5,
        "Concentric rings per city: ring colour = axis, opacity = value, radius scales with city area.",
        fontsize=6,
        color=DARK,
        ha="center",
        va="center",
    )
    leg_spacing_x = 0.14
    leg_start_x = 0.5 - (len(legend_items) - 1) * leg_spacing_x / 2

    for i, (axis, col) in enumerate(legend_items):
        lx = leg_start_x + i * leg_spacing_x
        leg_ax = fig.add_axes(
            [lx - circle_r_fig, leg_y - circle_r_fig, circle_r_fig * 2, circle_r_fig * 2],
            aspect="equal",
        )
        leg_ax.set_xlim(-1.5, 1.5)
        leg_ax.set_ylim(-1.5, 1.5)
        leg_ax.set_facecolor("none")
        leg_ax.axis("off")
        leg_ax.add_patch(plt.Circle((0, 0), 1.3, fill=False, edgecolor=col, lw=0.5, alpha=1.0))
        leg_ax.add_patch(plt.Circle((0, 0), 0.85, fill=False, edgecolor=col, lw=0.5, alpha=0.6))
        leg_ax.add_patch(plt.Circle((0, 0), 0.4, fill=False, edgecolor=col, lw=0.5, alpha=0.3))
        fig.text(lx + circle_r_fig + 0.006, leg_y, axis, fontsize=6, color=DARK, ha="left", va="center")

    return fig


def build_lines_plate(cities_for_lines, countries_ordered, max_cities):
    """Build plate 2B: stacked lines per city grouped by country."""
    n_countries = len(countries_ordered)

    # Size to content: ~0.16" per country row + overhead
    row_h_in = 0.16
    overhead_in = 0.85  # title + desc + padding + legend
    fig_w = 7.5
    fig_h = max(4.0, n_countries * row_h_in + overhead_in)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    title_h = 0.40 / fig_h  # title area (absolute ~0.40")
    desc_h = 0.30 / fig_h  # description area
    bottom_pad = 0.15 / fig_h
    label_w = 0.08  # left-side country labels
    legend_h = 0.04  # bottom legend row
    lines_bot = bottom_pad + desc_h + legend_h
    lines_h = 1.0 - title_h - lines_bot
    lines_left = label_w
    lines_w = 1.0 - label_w - 0.01

    draw_title(fig, "City Morphology by Country", y_inches=0.08)

    ax = fig.add_axes([lines_left, lines_bot, lines_w, lines_h])
    ax.set_facecolor(BG)
    ax.set_xlim(0, max_cities)
    ax.set_ylim(0, n_countries)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    line_lw = 1.4  # fixed line width
    line_alpha_min = 0.0  # fully transparent for zero values
    line_alpha_max = 0.95  # solid for high values
    line_gap = 0.24  # spacing between the 3 lines within each city
    country_gap = 0.45  # extra space between country groups

    # Compute y-centres with tighter city rows and wider country gaps
    row_band = 2 * line_gap + 0.2  # height occupied by one country's lines
    total_h = n_countries * row_band + (n_countries - 1) * country_gap
    ax.set_ylim(0, total_h)

    country_y_centres = {}
    for row_idx, country in enumerate(reversed(countries_ordered)):
        y_centre = row_idx * (row_band + country_gap) + row_band / 2
        country_y_centres[country] = y_centre
        cities = cities_for_lines[cities_for_lines["country"] == country].sort_values("_global_rank")
        for ci_idx, (_, c) in enumerate(cities.iterrows()):
            for offset, axis_key in [(line_gap, "intensity"), (0.0, "continuity"), (-line_gap, "irregularity")]:
                v = c[axis_key + "_n"]
                alpha = _ring_alpha(v, line_alpha_max)
                ax.plot(
                    [ci_idx + 0.08, ci_idx + 0.92],
                    [y_centre + offset, y_centre + offset],
                    color=LINE_COLORS[axis_key],
                    lw=line_lw,
                    alpha=alpha,
                    solid_capstyle="butt",
                )

    # Country labels
    for country, y_centre in country_y_centres.items():
        row_frac = y_centre / total_h
        row_mid = lines_bot + row_frac * lines_h
        fig.text(
            lines_left - 0.012, row_mid, country, fontsize=5, fontweight="medium", color=DARK, ha="right", va="center"
        )

    # Description
    desc_y = bottom_pad + desc_h / 2
    fig.text(
        0.5,
        desc_y,
        "Three lines per city (intensity / continuity / irregularity), clustered within country, north to south.",
        fontsize=6,
        color=DARK,
        ha="center",
        va="center",
    )

    # Horizontal legend at bottom
    legend_items = [
        ("Intensity", "#b2182b"),
        ("Continuity", "#2166ac"),
        ("Irregularity", "#1a9641"),
    ]
    leg_y = bottom_pad + desc_h + legend_h * 0.5
    leg_spacing_x = 0.16
    leg_start_x = 0.5 - (len(legend_items) - 1) * leg_spacing_x / 2
    # Legend line length matches one city line
    city_slot_w = lines_w / max_cities  # one city slot in figure fraction
    leg_line_w = city_slot_w * 0.84  # same proportion as city lines (0.08–0.92)
    text_gap = 0.008
    for i, (label, col) in enumerate(legend_items):
        lx = leg_start_x + i * leg_spacing_x
        # Draw a line segment matching the city lines
        line_x0 = lx - leg_line_w / 2 - 0.02
        leg_line = fig.add_axes([line_x0, leg_y - 0.002, leg_line_w, 0.004])
        leg_line.set_xlim(0, 1)
        leg_line.set_ylim(0, 1)
        leg_line.axis("off")
        leg_line.plot([0, 1], [0.5, 0.5], color=col, lw=line_lw, solid_capstyle="butt")
        fig.text(
            line_x0 + leg_line_w + text_gap,
            leg_y,
            label,
            fontsize=6,
            fontweight="bold",
            color=col,
            ha="left",
            va="center",
        )

    return fig


if __name__ == "__main__":
    print("Building separate Ripple + Lines plates...")
    city_meds, cities_for_lines, countries_ordered, max_cities, europe = load_data()

    # Plate 2: Ripples
    print("\n  Building plate 2 (ripples)...")
    fig_a = build_ripple_plate(city_meds, europe)
    out_a = OUTPUT_DIR / "plate2_ripples.png"
    fig_a.savefig(out_a, dpi=450, facecolor=BG)
    print(f"  Saved {out_a}")
    plt.close(fig_a)

    # Plate 3: Lines
    print("\n  Building plate 3 (lines)...")
    fig_b = build_lines_plate(cities_for_lines, countries_ordered, max_cities)
    out_b = OUTPUT_DIR / "plate3_lines.png"
    fig_b.savefig(out_b, dpi=450, facecolor=BG)
    print(f"  Saved {out_b}")
    plt.close(fig_b)

    print("Done.")
