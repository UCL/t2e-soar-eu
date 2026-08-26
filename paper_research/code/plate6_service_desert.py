"""
Plate 6 — Streets Without Access.

Grid matrix where each cell is a 10×10 waffle chart showing what
fraction of streets have zero services within 400 metres.  White
squares mark the desert fraction; filled squares use the service
colour.  Columns = 8 octant types, rows = service categories.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    DARK,
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

# Desert = nearest distance >= DESERT_THRESHOLD (nothing within walking distance).
DESERT_THRESHOLD = 400  # metres

# Build from shared METRIC_LAYERS for consistent colours
_METRIC_MAP = {m["col"]: m for layer in METRIC_LAYERS for m in layer["metrics"]}
DESERT_SERVICES = [
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

GRID_N = 10  # 10×10 = 100 squares per cell
CELL_SPAN = 0.72  # total cell width/height in data coords
SQ_GAP_FRAC = 0.10  # fraction of square step used as gap


# ============================================================================
# DATA
# ============================================================================


def load_data():
    """Load cached data and classify octants."""
    needed = list(AXIS_COLS.values()) + [s["col"] for s in DESERT_SERVICES]
    df = load_all_cached(columns=needed)
    print(f"  {len(df):,} nodes, {df['bounds_fid'].nunique()} cities")
    classified, _ = classify_octants(df)
    print(f"  {len(classified):,} classified nodes")
    return classified


def compute_desert_fractions(classified):
    """Compute % of segments where nearest service >= DESERT_THRESHOLD, by octant × service."""
    results = {}
    for svc in DESERT_SERVICES:
        col = svc["col"]
        vals = classified[["octant", col]].dropna(subset=[col])
        vals["is_desert"] = (vals[col] >= DESERT_THRESHOLD).astype(float)
        results[col] = vals.groupby("octant")["is_desert"].mean() * 100.0
    return results


# ============================================================================
# WAFFLE DRAWING
# ============================================================================


def draw_waffle(ax, cx, cy, pct, color):
    """Draw a 10×10 waffle grid centred at (cx, cy).

    Squares fill from bottom-left, row by row upward.
    """
    n_filled = round(pct)
    half = CELL_SPAN / 2
    sq_step = CELL_SPAN / GRID_N
    sq_size = sq_step * (1.0 - SQ_GAP_FRAC)
    base_rgb = mcolors.to_rgb(color)

    # No borders — gaps between squares form the grid
    # Empty squares: light warm grey, visible against BG (#FAFAF8)
    EMPTY_RGB = (0.91, 0.90, 0.89)

    count = 0
    for row in range(GRID_N):  # bottom to top
        for col in range(GRID_N):  # left to right
            x = cx - half + col * sq_step
            y = cy - half + row * sq_step
            is_desert = count < n_filled

            fc = EMPTY_RGB if is_desert else base_rgb

            rect = mpatches.Rectangle(
                (x, y),
                sq_size,
                sq_size,
                facecolor=fc,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            ax.add_patch(rect)
            count += 1


# ============================================================================
# FIGURE
# ============================================================================


def build_figure(desert_fractions):
    """Grid matrix with waffle cells."""
    n_svc = len(DESERT_SERVICES)
    n_oct = len(OCTANT_ORDER)

    fig_w = 7.5
    fig_h = 9.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    margins = standard_margins(
        fig,
        top_inches=0.88,
        bottom_inches=0.55,
        left_inches=0.85,
        right_inches=0.15,
    )
    fig.subplots_adjust(**margins)

    # ── Draw waffle cells ──────────────────────────────────────────
    for si, svc in enumerate(DESERT_SERVICES):
        col = svc["col"]
        y = n_svc - 1 - si  # top row = first service

        for oi, octant in enumerate(OCTANT_ORDER):
            x = oi
            frac = desert_fractions[col].get(octant, 0)
            draw_waffle(ax, x, y, frac, svc["color"])

            # Percentage label below the waffle
            ax.text(
                x,
                y - CELL_SPAN / 2 - 0.05,
                f"{frac:.0f}%",
                fontsize=5.5,
                fontweight="bold",
                color=DARK,
                ha="center",
                va="top",
                zorder=3,
            )

    # ── Row labels ─────────────────────────────────────────────────
    for si, svc in enumerate(DESERT_SERVICES):
        y = n_svc - 1 - si
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

    # ── Axis setup ─────────────────────────────────────────────────
    ax.set_xlim(-0.6, n_oct - 0.4)
    ax.set_ylim(-0.65, n_svc - 0.35)
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
        sym_fy = bbox.y1 + 0.025
        place_composite_symbol(fig, fx, sym_fy, octant, size_inches=SYM_INCHES)

    # ── Title, legend, subtitle ────────────────────────────────────
    draw_title(fig, "Service Deserts by Morphological Type")
    # Subtitle above legend
    fig.text(
        0.5,
        0.42 / fig_h_in,
        "Each square = 1% of street segments.  White indicates streets beyond 400-metre walking access.",
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

    print("\nComputing desert fractions...")
    desert_fractions = compute_desert_fractions(classified)
    for svc in DESERT_SERVICES:
        col = svc["col"]
        vals = desert_fractions[col]
        print(f"  {svc['name']}: {vals.min():.0f}% – {vals.max():.0f}%")

    print("\nBuilding Plate 6 — Streets Without Access...")
    fig = build_figure(desert_fractions)
    out = OUTPUT_DIR / "plate6_service_desert.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)
    print("\nDone.")
