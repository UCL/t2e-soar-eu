#!/usr/bin/env python3
"""Plate 12 — Form and Access (replaces the density scatter).

Street-level access mapped across the intensity x continuity form space as a
smooth surface; two panels (commercial service + green space). City dots mark
each city's median form position, and Closest/Farthest city series are listed
beneath. Pooled across cities (descriptive); the within-city decomposition in
the analytical supplement isolates the within-city component.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    BOUNDARIES_PATH,
    DARK,
    GREY,
    OUTPUT_DIR,
    apply_atlas_style,
    classify_octants,
    draw_title,
    load_all_cached,
)

apply_atlas_style()

FSI = AXIS_COLS["intensity"]
FRONT = AXIS_COLS["continuity"]
SVC_COLS = [
    "cc_retail_nearest_max_1600",
    "cc_health_and_medical_nearest_max_1600",
    "cc_education_nearest_max_1600",
    "cc_eat_and_drink_nearest_max_1600",
    "cc_business_and_services_nearest_max_1600",
]
GREEN_COL = "cc_green_nearest_max_1600"

# Thematic single-hue gradients from the atlas palette: saturated = closer, pale = farther.
CMAP_COMM = LinearSegmentedColormap.from_list("comm_red", ["#d62728", "#E58A8A", "#FBE9E9"])
CMAP_GREEN = LinearSegmentedColormap.from_list("green_grn", ["#1E8449", "#74B591", "#E8F3EC"])
DOT = "#1a1a2e"  # single neutral dot colour (atlas DARK), used on plot and in lists

FSI_TICKS = [0.1, 0.2, 0.5, 1.0, 2.0]
N_LIST = 8
MIN_COUNT = 80
NBINS = 52


def _gradient_legend(ax, cmap, vmin, vmax):
    """Slim atlas-style gradient key to the right of a panel (no colorbar box)."""
    cax = ax.inset_axes([1.035, 0.18, 0.045, 0.64])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([])
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.text(0.5, 1.07, f"{vmax:.0f} m", transform=cax.transAxes, fontsize=5, color=GREY, ha="center", va="bottom")
    cax.text(0.5, -0.07, f"{vmin:.0f} m", transform=cax.transAxes, fontsize=5, color=GREY, ha="center", va="top")
    cax.text(
        -0.35, 0.5, "farther", transform=cax.transAxes, fontsize=4.6, color=GREY, ha="center", va="center", rotation=90
    )


def _surface(ax, lx, y, dist, xlim, title, city_df, cmap):
    xedges = np.linspace(xlim[0], xlim[1], NBINS + 1)
    yedges = np.linspace(0, 1, NBINS + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XC, YC = np.meshgrid(xc, yc)

    ok = np.isfinite(dist)
    med, _, _, _ = binned_statistic_2d(lx[ok], y[ok], dist[ok], statistic="median", bins=[xedges, yedges])
    cnt, _, _, _ = binned_statistic_2d(lx[ok], y[ok], dist[ok], statistic="count", bins=[xedges, yedges])
    mask = cnt >= MIN_COUNT
    grid = np.where(mask, med, np.nanmedian(med[mask]))
    grid = gaussian_filter(grid, sigma=1.2)
    grid = np.where(mask, grid, np.nan)
    vmin, vmax = np.nanpercentile(med[mask], [4, 96])

    # Filled contour bands + crisp iso-distance lines (the contours carry the structure)
    levels = np.linspace(vmin, vmax, 11)
    ax.contourf(XC, YC, grid.T, levels=levels, cmap=cmap, extend="both", antialiased=True)
    ax.contour(XC, YC, grid.T, levels=levels, colors="white", linewidths=0.4, alpha=0.7)

    # City dots at their median form position (neutral; dominant octant is
    # ~always Light-Freestanding across cities, so it carries no signal here).
    cdots = city_df.dropna(subset=[FSI, FRONT])
    ax.scatter(np.log10(cdots[FSI]), cdots[FRONT], s=10, c=DOT, edgecolors="white", linewidths=0.4, alpha=0.9, zorder=5)

    # Quadrant labels (white halo keeps them legible over any fill). The
    # street-level classification thresholds are deliberately not drawn: city
    # medians all sit well below them, so the lines would imply a relationship
    # to the dots that does not exist.
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for fx, fy, lbl, ha, va in [
        (0.03, 0.96, "Light–Attached", "left", "top"),
        (0.97, 0.96, "Intense–Attached", "right", "top"),
        (0.03, 0.04, "Light–Freestanding", "left", "bottom"),
        (0.97, 0.04, "Intense–Freestanding", "right", "bottom"),
    ]:
        ax.text(fx, fy, lbl, fontsize=6, color=DARK, ha=ha, va=va, transform=ax.transAxes, path_effects=halo, zorder=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_xticks([np.log10(t) for t in FSI_TICKS])
    ax.set_xticklabels([f"{t:g}" for t in FSI_TICKS])
    ax.set_xlabel("Intensity — FSI", fontsize=7, color=DARK)
    ax.set_ylabel("Continuity — frontage ratio", fontsize=7, color=DARK)
    ax.set_title(title, fontsize=8, fontweight="bold", color=DARK, pad=5)
    ax.tick_params(labelsize=6, colors=GREY, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    _gradient_legend(ax, cmap, vmin, vmax)


def _lists(ax, city_df, metric):
    ax.axis("off")
    s = city_df.dropna(subset=[metric]).sort_values(metric)
    closest = s.head(N_LIST)
    farthest = s.tail(N_LIST).iloc[::-1]
    for sub, hdr, x0 in [(closest, "Closest", 0.0), (farthest, "Farthest", 0.52)]:
        ax.text(x0, 1.0, hdr, fontsize=6.5, fontweight="bold", color=DARK, ha="left", va="top", transform=ax.transAxes)
        for i, (_, r) in enumerate(sub.iterrows()):
            yy = 0.87 - i * 0.108
            ax.plot(x0 + 0.018, yy, "o", ms=3.0, color=DOT, transform=ax.transAxes, clip_on=False)
            ax.text(
                x0 + 0.055,
                yy,
                f"{r['label']}, {r['country']}",
                fontsize=5.6,
                color=DARK,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )


def main():
    print("Loading data...")
    df = load_all_cached(columns=list(AXIS_COLS.values()) + SVC_COLS + [GREEN_COL])
    cl, _ = classify_octants(df)
    cl = cl[(cl[FSI] > 0.05) & cl[FRONT].between(0, 1)].copy()
    cl["service"] = cl[SVC_COLS].mean(axis=1)
    cl["green"] = cl[GREEN_COL]
    print(f"  {len(cl):,} classified segments")

    city = df.groupby("bounds_fid")[[FSI, FRONT] + SVC_COLS + [GREEN_COL]].median()
    city["service"] = city[SVC_COLS].mean(axis=1)
    city["green"] = city[GREEN_COL]
    try:
        bounds = gpd.read_file(BOUNDARIES_PATH)
        city = city.merge(
            bounds[["bounds_fid", "label", "country"]], left_index=True, right_on="bounds_fid", how="left"
        )
    except Exception:
        city["label"] = ""
        city["country"] = ""
    city = city[(city[FSI] > 0.05) & city[FRONT].between(0, 1)].copy()

    lx = np.log10(cl[FSI].to_numpy())
    y = cl[FRONT].to_numpy()
    xlim = (np.nanpercentile(lx, 1), np.nanpercentile(lx, 99))

    fig = plt.figure(figsize=(7.5, 8.2), facecolor=BG)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.6, 1.0], hspace=0.32, wspace=0.34, left=0.075, right=0.93, top=0.92, bottom=0.10
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(BG)
    _surface(ax_a, lx, y, cl["service"].to_numpy(), xlim, "Commercial service access", city, CMAP_COMM)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(BG)
    _surface(ax_b, lx, y, cl["green"].to_numpy(), xlim, "Green-space access", city, CMAP_GREEN)

    _lists(fig.add_subplot(gs[1, 0]), city, "service")
    _lists(fig.add_subplot(gs[1, 1]), city, "green")

    draw_title(fig, "Form and Access")
    fig.text(
        0.5,
        0.022,
        "Street-level medians across the form space, pooled across cities.  "
        "Saturated = closer, pale = farther; small dots are cities at their median form position.",
        fontsize=6,
        color=DARK,
        ha="center",
        va="bottom",
    )

    out = OUTPUT_DIR / "plate12_formspace.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
