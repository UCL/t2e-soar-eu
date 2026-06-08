"""
Generate example SOAR metrics visualization for Amsterdam.

Creates a 2×2 figure showing:
- Panel A: Network centrality (cc_beta_800)
- Panel B: Retail catchment at 400m (cc_retail_400_nw)
- Panel C: Green and blue space proximity (cc_green_nearest_max_1600)
- Panel D: Population density (Eurostat Census Grid 2021)

This figure demonstrates what the SOAR dataset looks like in practice.
"""

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from config import FIG_DIR, PROCESSED_DIR, apply_plot_style
from matplotlib.cm import ScalarMappable

# ============================================================================
# SETUP
# ============================================================================

# Configuration
CITY_FID = 1590  # Amsterdam
CITY_NAME = "Amsterdam"
METRICS_PATH = PROCESSED_DIR / f"metrics_{CITY_FID}.gpkg.zip"

# Metrics to visualize
METRICS = [
    {
        "column": "cc_beta_800",
        "title": "A. Network Centrality",
        "label": "Closeness (800m)",
        "cmap": "viridis",
        "description": "Higher = more connected",
    },
    {
        "column": "cc_retail_400_nw",
        "title": "B. Retail Catchment",
        "label": "Retail POIs within 400m",
        "cmap": "YlOrRd",  # Yellow=few, red=many
        "description": "Higher = more retail nearby",
    },
    {
        "column": "cc_green_nearest_max_1600",
        "title": "C. Green (and Blue) Space",
        "label": "Distance to green/blue (m)",
        "cmap": "Greens_r",  # Dark green=close (low), white=far (high)
        "description": "Lower = closer to parks/water",
    },
    {
        "column": "density",
        "title": "D. Population Density",
        "label": "Persons / km²",
        "cmap": "magma",
        "description": "Higher = more residents nearby",
    },
]

# Zoom extent for Amsterdam (central area crop)
# Coordinates in EPSG:3035 (ETRS89-LAEA)
# Full extent: x 3962048-3982968, y 3251007-3280979
# This crops to central ~12km x 12km area
ZOOM_BOUNDS = {
    "xmin": 3966500,
    "xmax": 3978500,
    "ymin": 3260000,
    "ymax": 3272000,
}


def main() -> int:
    apply_plot_style()

    print("=" * 70)
    print("GENERATING EXAMPLE METRICS FIGURE (AMSTERDAM)")
    print("=" * 70)

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print(f"\nLoading {CITY_NAME} data from {METRICS_PATH}...")

    if not METRICS_PATH.exists():
        print(f"  Error: {METRICS_PATH} not found")
        print("  Run the SOAR processing pipeline first to generate city metrics")
        return 1

    # Load streets layer
    streets_gdf = gpd.read_file(METRICS_PATH, layer="streets")
    print(f"  Loaded {len(streets_gdf):,} street segments")

    # Check that required columns exist
    for metric in METRICS:
        col = metric["column"]
        if col not in streets_gdf.columns:
            print(f"  Warning: Column '{col}' not found in data")
            # List available columns for debugging
            cc_cols = [c for c in streets_gdf.columns if c.startswith("cc_")]
            print(f"  Available cc_ columns: {cc_cols[:10]}...")
            return 1

    # ========================================================================
    # CREATE FIGURE
    # ========================================================================

    print("\nCreating figure...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        col = metric["column"]

        # Get data and compute percentile-based limits to handle outliers
        values = streets_gdf[col].dropna()
        vmin = np.percentile(values, 2)
        vmax = np.percentile(values, 98)

        print(f"  {metric['title']}: {col}")
        print(f"    Range: {values.min():.2f} to {values.max():.2f}")
        print(f"    Display range (2-98%): {vmin:.2f} to {vmax:.2f}")

        # Plot streets colored by metric
        streets_gdf.plot(
            column=col,
            ax=ax,
            cmap=metric["cmap"],
            linewidth=0.6,
            vmin=vmin,
            vmax=vmax,
            legend=False,
        )

        # Apply zoom/crop
        ax.set_xlim(ZOOM_BOUNDS["xmin"], ZOOM_BOUNDS["xmax"])
        ax.set_ylim(ZOOM_BOUNDS["ymin"], ZOOM_BOUNDS["ymax"])

        # Style
        ax.set_title(metric["title"], fontsize=13, fontweight="bold", pad=6)
        ax.set_axis_off()
        ax.set_aspect("equal")

        # Vertical colorbar on right (no label on it)
        sm = ScalarMappable(cmap=metric["cmap"], norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, aspect=25)
        cbar.ax.tick_params(labelsize=9)

        # Separate x-axis label underneath the plot
        ax.set_axis_on()
        ax.set_xlabel(metric["label"], fontsize=11)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Spacing: enough wspace for colorbars, modest hspace
    plt.subplots_adjust(wspace=0.15, hspace=0.10)

    # ========================================================================
    # SAVE
    # ========================================================================

    png_path = FIG_DIR / "fig_example_amsterdam.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
