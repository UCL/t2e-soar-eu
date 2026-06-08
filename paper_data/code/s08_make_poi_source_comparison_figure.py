"""
Spatial comparison of Overture POIs vs official registry data.

Side-by-side point maps for Amsterdam (vs BAG) and Lyon (vs SIRENE),
showing retail POI locations in inner-city areas. Demonstrates visually
how Overture spatial patterns relate to official registries.
"""

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from config import (
    BAG_USAGE_MAPPING,
    BOUNDS_PATH,
    FIG_DIR,
    OVERTURE_CATEGORY_MAPPING,
    OVERTURE_DIR,
    SIRENE_APE_MAPPING,
    VALIDATION_DIR,
    apply_plot_style,
)
from shapely.geometry import box

# ============================================================================
# CONFIGURATION
# ============================================================================

CITIES = [
    {
        "name": "Amsterdam",
        "bounds_fid": 1590,
        "country": "NL",
        "official_source": "BAG",
        "official_path": VALIDATION_DIR / "bag_netherlands.gpkg",
        "official_layer": "buildings",
        "official_filter_col": "gebruiksdoel",
        "official_filter_vals": BAG_USAGE_MAPPING["retail"],
        # Inner city zoom (EPSG:3035) — 3km×2km centred on retail density peak
        "zoom": {"xmin": 3972000, "xmax": 3975000, "ymin": 3263000, "ymax": 3265000},
    },
    {
        "name": "Lyon",
        "bounds_fid": 3978,
        "country": "FR",
        "official_source": "SIRENE",
        "official_path": VALIDATION_DIR / "sirene_france.gpkg",
        "official_layer": "establishments",
        "official_filter_col": "APE_code",
        "official_filter_vals": SIRENE_APE_MAPPING["retail"],
        # Inner city zoom (EPSG:3035) — 3km×2km centred on retail density peak
        "zoom": {"xmin": 3917400, "xmax": 3920400, "ymin": 2529700, "ymax": 2531700},
    },
]

OVERTURE_RETAIL_CLASSES = OVERTURE_CATEGORY_MAPPING["retail"]

# Colours
BG_LAND = "#f7f7f7"
BG_BUILDING = "#e8e8e8"
BG_STREET = "#d5d5d5"
BG_WATER = "#d4e6f1"
COL_OFFICIAL = "#c0392b"
COL_OVERTURE = "#2471a3"

# ============================================================================
# HELPERS
# ============================================================================


def draw_background(ax, overture_path, zoom_gdf, zoom):
    """Draw buildings and streets as light background context."""
    # Buildings — light grey fill reveals water as negative space
    buildings = gpd.read_file(overture_path, layer="buildings", mask=zoom_gdf)
    if len(buildings) > 0:
        buildings.plot(ax=ax, color=BG_BUILDING, edgecolor="none", linewidth=0)

    # Streets — very thin grey lines
    edges = gpd.read_file(overture_path, layer="edges", mask=zoom_gdf)
    if len(edges) > 0:
        edges.plot(ax=ax, color=BG_STREET, linewidth=0.3)


def draw_scale_bar(ax, zoom, length_m=500):
    """Draw a minimal scale bar in the bottom-left."""
    bar_x = zoom["xmin"] + 100
    bar_y = zoom["ymin"] + 100
    ax.plot(
        [bar_x, bar_x + length_m],
        [bar_y, bar_y],
        color="#333333",
        linewidth=1.5,
        solid_capstyle="butt",
    )
    # Tick ends
    tick_h = 30
    for x in [bar_x, bar_x + length_m]:
        ax.plot([x, x], [bar_y - tick_h, bar_y + tick_h], color="#333333", linewidth=1)
    label = f"{length_m} m" if length_m < 1000 else f"{length_m // 1000} km"
    ax.text(
        bar_x + length_m / 2,
        bar_y + 60,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333",
    )


def draw_north_arrow(ax, zoom, arrow_len=120):
    """Draw a minimal north arrow in the top-right."""
    x = zoom["xmax"] - 120
    y = zoom["ymax"] - 120
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(x, y - arrow_len),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
    )
    ax.text(
        x,
        y + 28,
        "N",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color="#333333",
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    output_path = FIG_DIR / "fig_poi_spatial_comparison.png"
    if output_path.exists():
        print(f"✓ Figure already exists, skipping: {output_path}")
        return 0

    apply_plot_style()

    print("=" * 70)
    print("SPATIAL COMPARISON: OVERTURE vs OFFICIAL REGISTRIES")
    print("=" * 70)

    bounds_gdf = gpd.read_file(BOUNDS_PATH, layer="bounds")

    # Use a figure aspect ratio close to the map extents (3km x 2km => 1.5).
    # With equal-aspect axes, an overly-wide figure causes each map to shrink
    # inside its subplot cell, creating an apparent "gutter" between columns.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8),
        gridspec_kw={"wspace": 0.0, "hspace": 0.08},
    )

    # Layout: cities are columns (Amsterdam left, Lyon right);
    # within each city, Official registry is top row and Overture is bottom row.
    for col_idx, city in enumerate(CITIES):
        name = city["name"]
        fid = city["bounds_fid"]
        zoom = city["zoom"]

        print(f"\n--- {name} (fid={fid}) ---")

        city_boundary = bounds_gdf[bounds_gdf["bounds_fid"] == fid]
        zoom_box = box(zoom["xmin"], zoom["ymin"], zoom["xmax"], zoom["ymax"])
        zoom_gdf = gpd.GeoDataFrame(geometry=[zoom_box], crs="EPSG:3035")
        overture_path = OVERTURE_DIR / f"overture_{fid}.gpkg.zip"
        if not overture_path.exists():
            overture_path = OVERTURE_DIR / f"overture_{fid}.gpkg"

        # --- Load official data ---
        print(f"  Loading {city['official_source']} data...")
        official = gpd.read_file(
            city["official_path"],
            layer=city["official_layer"],
            mask=city_boundary,
        )
        print(f"    Total in city: {len(official):,}")

        filter_col = city["official_filter_col"]
        filter_vals = city["official_filter_vals"]

        if city["country"] == "FR":
            mask = official[filter_col].apply(
                lambda x, _fv=filter_vals: any(str(x).startswith(prefix) for prefix in _fv) if pd.notna(x) else False
            )
        else:
            pattern = "|".join(filter_vals)
            mask = official[filter_col].str.contains(pattern, na=False)

        official_retail = official[mask].copy()
        print(f"    Retail: {len(official_retail):,}")
        official_zoom = official_retail[official_retail.geometry.within(zoom_box)]
        print(f"    In zoom area: {len(official_zoom):,}")

        # --- Load Overture data ---
        print("  Loading Overture data...")
        overture = gpd.read_file(overture_path, layer="places")
        print(f"    Total POIs: {len(overture):,}")
        overture_retail = overture[overture["major_lu_schema_class"].isin(OVERTURE_RETAIL_CLASSES)].copy()
        print(f"    Retail: {len(overture_retail):,}")
        overture_zoom = overture_retail[overture_retail.geometry.within(zoom_box)]
        print(f"    In zoom area: {len(overture_zoom):,}")

        # --- Plot ---
        ax_official = axes[0, col_idx]
        ax_overture = axes[1, col_idx]

        for ax in [ax_official, ax_overture]:
            # Water-coloured background -- buildings will paint over land areas,
            # leaving water bodies as the background colour
            ax.set_facecolor(BG_WATER)
            ax.set_xlim(zoom["xmin"], zoom["xmax"])
            ax.set_ylim(zoom["ymin"], zoom["ymax"])
            ax.set_aspect("equal")
            ax.set_axis_off()

        # Background context (buildings + streets) on both panels
        print("  Drawing background...")
        for ax in [ax_official, ax_overture]:
            draw_background(ax, overture_path, zoom_gdf, zoom)

        # Official data points
        if len(official_zoom) > 0:
            ax_official.scatter(
                official_zoom.geometry.x,
                official_zoom.geometry.y,
                s=2,
                c=COL_OFFICIAL,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
                zorder=5,
            )
        ax_official.set_title(
            f"{name} — {city['official_source']} (n={len(official_zoom):,})",
            fontsize=11,
            fontweight="bold",
            pad=2,
        )

        # Overture data points
        if len(overture_zoom) > 0:
            ax_overture.scatter(
                overture_zoom.geometry.x,
                overture_zoom.geometry.y,
                s=2,
                c=COL_OVERTURE,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
                zorder=5,
            )
        ax_overture.set_title(
            f"{name} — Overture (n={len(overture_zoom):,})",
            fontsize=11,
            fontweight="bold",
            pad=2,
        )

        # Scale bar + north arrow (top row)
        draw_scale_bar(ax_official, zoom)
        draw_north_arrow(ax_official, zoom)

    # Row labels (sources)
    for row_idx, label in enumerate(["Official registry", "Overture Maps"]):
        axes[row_idx, 0].text(
            -0.05,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="right",
        )

    # Minimal legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=BG_WATER, edgecolor="#999999", linewidth=0.5, label="Water"),
        mpatches.Patch(facecolor=BG_BUILDING, edgecolor="#999999", linewidth=0.5, label="Buildings"),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=COL_OFFICIAL, markersize=5, label="Official registry"
        ),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_OVERTURE, markersize=5, label="Overture POI"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "Retail POI Spatial Patterns: Official Registries vs Overture Maps",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    # Manual layout (avoid tight_layout adding inter-column padding).
    fig.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.09)

    # Save (PNG only -- rasterised building footprints render poorly in PDF)
    output_path = FIG_DIR / "fig_poi_spatial_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"\nSaved: {output_path}")

    plt.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
