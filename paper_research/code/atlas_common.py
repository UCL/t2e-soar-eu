"""Shared configuration, octant classification, and styling for Atlas figures."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ============================================================================
# PATHS
# ============================================================================
if "T2E_DATA_DIR" not in os.environ:
    raise OSError("T2E_DATA_DIR environment variable is not set. See .env.example.")
DATA_DIR = Path(os.environ["T2E_DATA_DIR"])
CACHE_DIR = DATA_DIR / "temp_egs" / "shared_cache"
BOUNDARIES_PATH = DATA_DIR / "datasets" / "boundaries.gpkg"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "atlas"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE CONSTANTS
# ============================================================================
BG = "#FAFAF8"
DARK = "#1a1a2e"
GREY = "#666666"
GRID_COLOR = "#B0ADA8"

# ── Symbol sizing (absolute, DPI-independent) ──
SYM_SOURCE_PX = 600  # matches actual symbol PNG resolution
SYM_INCHES = 0.14  # target physical size in inches


def sym_zoom(fig):
    """Compute OffsetImage zoom for SYM_INCHES physical size."""
    return SYM_INCHES * fig.dpi / SYM_SOURCE_PX


def place_sub_symbol(fig, x, y, octant, part, size_inches=None, color=None):
    """Draw a sub-symbol as a tiny inset axes at (x, y) in figure fraction.

    Vector-quality rendering — no PNG/SVG files needed. The symbol is drawn
    directly using matplotlib patches, so it scales perfectly at any DPI.

    Parameters
    ----------
    fig : matplotlib Figure
    x, y : float — centre position in figure-fraction coordinates
    octant : str — e.g. "HHH"
    part : str — "intensity", "continuity", or "irregularity"
    size_inches : float or None — physical size in inches. If None, uses SYM_INCHES.
    color : str or None — override symbol colour. If None, uses default dark grey.
    """
    global _COL_I, _COL_C, _COL_R
    if size_inches is None:
        size_inches = SYM_INCHES
    fig_w, fig_h = fig.get_size_inches()
    w_frac = size_inches / fig_w
    h_frac = size_inches / fig_h
    ax_s = fig.add_axes([x - w_frac / 2, y - h_frac / 2, w_frac, h_frac])
    ax_s.set_xlim(-1, 1)
    ax_s.set_ylim(-1, 1)
    ax_s.set_aspect("equal", adjustable="datalim")
    ax_s.axis("off")
    ax_s.set_frame_on(False)
    idx = {"intensity": 0, "continuity": 1, "irregularity": 2}[part]
    is_high = octant[idx] == "H"
    draw_fn = {
        "intensity": _draw_triangle,
        "continuity": _draw_blocks,
        "irregularity": _draw_irregularity,
    }[part]
    if color is not None:
        saved = _COL_I, _COL_C, _COL_R
        _COL_I = _COL_C = _COL_R = color
    draw_fn(ax_s, 0, 0, 0.7, 0.7, is_high)
    if color is not None:
        _COL_I, _COL_C, _COL_R = saved
    return ax_s


def place_composite_symbol(fig, x, y, octant, size_inches=None):
    """Draw the full 3-part composite symbol (stacked vertically) at (x, y).

    Parameters
    ----------
    fig : matplotlib Figure
    x, y : float — centre position in figure-fraction coordinates
    octant : str — e.g. "HHH"
    size_inches : float or None — size of each sub-symbol.
    """
    if size_inches is None:
        size_inches = SYM_INCHES
    fig_w, fig_h = fig.get_size_inches()
    spacing = size_inches * 1.3 / fig_h  # vertical spacing in figure fraction
    place_sub_symbol(fig, x, y + spacing, octant, "intensity", size_inches)
    place_sub_symbol(fig, x, y, octant, "continuity", size_inches)
    place_sub_symbol(fig, x, y - spacing, octant, "irregularity", size_inches)


def draw_legend_row(fig, y_inches=0.25):
    """Draw the standard morphological type legend row at the bottom of a figure.

    Always uses figure-fraction coordinates. Position is specified as an
    absolute distance in inches from the bottom of the figure for consistency
    across figures of different heights.

    Parameters
    ----------
    fig : matplotlib Figure
    y_inches : float — distance from bottom of figure in inches (default 0.25")
    """
    fig_w, fig_h = fig.get_size_inches()
    leg_y = y_inches / fig_h  # convert to figure fraction

    legend_items = [
        ("HHH", "intensity", "Intense"),
        ("LLL", "intensity", "Light"),
        None,
        ("HHH", "continuity", "Attached"),
        ("LLL", "continuity", "Freestanding"),
        None,
        ("HHH", "irregularity", "Irregular"),
        ("LLL", "irregularity", "Rectilinear"),
    ]

    sym_w_f = SYM_INCHES / fig_w
    gap_f = 0.005
    pair_gap_f = 0.008
    slash_gap_f = 0.015
    font_leg = 7
    label_widths_f = {
        "Intense": 0.042,
        "Light": 0.030,
        "Attached": 0.050,
        "Freestanding": 0.070,
        "Irregular": 0.052,
        "Rectilinear": 0.062,
    }

    # Compute total width to centre.
    total_w = 0
    for item in legend_items:
        if item is None:
            total_w += 2 * slash_gap_f + 0.005
        else:
            total_w += sym_w_f + gap_f + label_widths_f[item[2]] + pair_gap_f
    total_w -= pair_gap_f

    xc = 0.5 - total_w / 2

    for item in legend_items:
        if item is None:
            xc += slash_gap_f
            fig.text(xc, leg_y, "/", fontsize=font_leg, color=GREY, ha="center", va="center")
            xc += slash_gap_f + 0.005
        else:
            oct_key, axis_name, label = item
            place_sub_symbol(fig, xc + sym_w_f / 2, leg_y, oct_key, axis_name)
            xc += sym_w_f + gap_f
            fig.text(xc, leg_y, label, fontsize=font_leg, color=DARK, ha="left", va="center")
            xc += label_widths_f[label] + pair_gap_f


def draw_title(fig, title, y_inches=0.08):
    """Draw the figure title at a consistent absolute distance from the top.

    Parameters
    ----------
    fig : matplotlib Figure
    title : str
    y_inches : float — distance from top of figure in inches (default 0.15")
    """
    fig_h = fig.get_size_inches()[1]
    y_frac = 1.0 - y_inches / fig_h
    fig.text(0.5, y_frac, title, fontsize=10, fontweight="bold", color=DARK, ha="center", va="top")


def standard_margins(fig, top_inches=0.45, bottom_inches=0.35, left_inches=0.15, right_inches=0.05):
    """Convert absolute inch margins to figure-fraction values.

    Returns a dict suitable for fig.subplots_adjust(**margins).
    """
    fig_w, fig_h = fig.get_size_inches()
    return {
        "top": 1.0 - top_inches / fig_h,
        "bottom": bottom_inches / fig_h,
        "left": left_inches / fig_w,
        "right": 1.0 - right_inches / fig_w,
    }


GRADE_FILLED = "\u25cf"  # ● A-grade
GRADE_HOLLOW = "\u25cb"  # ○ B-grade (POI)


def apply_atlas_style():
    """Set matplotlib rcParams to match the atlas visual language."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # Do NOT use savefig.bbox = tight — it changes output width
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


# ============================================================================
# AXIS COLUMNS (400 m, distance-weighted)
# ============================================================================
AXIS_COLS = {
    "intensity": "cc_block_far_median_400_wt",
    "continuity": "frontage_max",  # max(left, right) street-frontage continuity
    "irregularity": "cc_orientation_mad_400_wt",
}

# Legacy SWR column, retained for comparison
SWR_COL = "cc_shared_wall_ratio_median_400_wt"

# ============================================================================
# OCTANT DEFINITIONS
# ============================================================================
OCTANT_ORDER = ["HHH", "HHL", "HLH", "HLL", "LHH", "LHL", "LLH", "LLL"]

OCTANT_NAMES = {
    "HHH": "Intense attached irregular",
    "HHL": "Intense attached rectilinear",
    "HLH": "Intense freestanding irregular",
    "HLL": "Intense freestanding rectilinear",
    "LHH": "Light attached irregular",
    "LHL": "Light attached rectilinear",
    "LLH": "Light freestanding irregular",
    "LLL": "Light freestanding rectilinear",
}

# Glyph symbology: ▲/▽ intensity, ■/□ continuity, ✳/⊞ irregularity
OCTANT_GLYPHS = {
    "HHH": "\u25b2\u25a0\u2733",  # ▲■✳
    "HHL": "\u25b2\u25a0\u229e",  # ▲■⊞
    "HLH": "\u25b2\u25a1\u2733",  # ▲□✳
    "HLL": "\u25b2\u25a1\u229e",  # ▲□⊞
    "LHH": "\u25bd\u25a0\u2733",  # ▽■✳
    "LHL": "\u25bd\u25a0\u229e",  # ▽■⊞
    "LLH": "\u25bd\u25a1\u2733",  # ▽□✳
    "LLL": "\u25bd\u25a1\u229e",  # ▽□⊞
}

OCTANT_SHORT = {
    "HHH": "Intense\nAttached\nIrregular",
    "HHL": "Intense\nAttached\nRectilinear",
    "HLH": "Intense\nFreestanding\nIrregular",
    "HLL": "Intense\nFreestanding\nRectilinear",
    "LHH": "Light\nAttached\nIrregular",
    "LHL": "Light\nAttached\nRectilinear",
    "LLH": "Light\nFreestanding\nIrregular",
    "LLL": "Light\nFreestanding\nRectilinear",
}

# Colour logic: warm = attached, cool = freestanding.
# Dark = intense (H intensity), light = light (L intensity).
OCTANT_COLORS = {
    "HHH": "#A8201A",  # Intense attached irregular      — crimson
    "HHL": "#1B4F72",  # Intense attached rectilinear    — navy
    "HLH": "#B8600A",  # Intense freestanding irregular   — burnt orange
    "HLL": "#6C3483",  # Intense freestanding rectilinear — purple
    "LHH": "#E74C3C",  # Light attached irregular        — bright red
    "LHL": "#2E86C1",  # Light attached rectilinear      — blue
    "LLH": "#D4AC0D",  # Light freestanding irregular     — gold
    "LLL": "#1E8449",  # Light freestanding rectilinear   — green
}

# ============================================================================
# METRIC LAYER DEFINITIONS  (green & trees + service access)
# ============================================================================
METRIC_LAYERS = [
    {
        "label": "NATURE",
        "metrics": [
            {"col": "cc_trees_nearest_max_1600", "name": "Tree canopy", "color": "#2D6A4F", "grade": "A", "unit": "m"},
            {"col": "cc_green_nearest_max_1600", "name": "Green space", "color": "#52B788", "grade": "A", "unit": "m"},
        ],
    },
    {
        "label": "SERVICES",
        "metrics": [
            {"col": "cc_business_and_services_nearest_max_1600", "name": "Business & Services", "color": "#1f77b4", "grade": "B", "unit": "m"},
            {"col": "cc_retail_nearest_max_1600", "name": "Retail", "color": "#e67e22", "grade": "B", "unit": "m"},
            {"col": "cc_transport_nearest_max_1600", "name": "Transport", "color": "#d4ac0d", "grade": "B", "unit": "m"},
            {"col": "cc_eat_and_drink_nearest_max_1600", "name": "Eat & Drink", "color": "#d62728", "grade": "B", "unit": "m"},
            {"col": "cc_health_and_medical_nearest_max_1600", "name": "Health", "color": "#17a2b8", "grade": "B", "unit": "m"},
            {"col": "cc_education_nearest_max_1600", "name": "Education", "color": "#9467bd", "grade": "B", "unit": "m"},
            {"col": "cc_attractions_and_activities_nearest_max_1600", "name": "Attractions", "color": "#8c564b", "grade": "B", "unit": "m"},
            {"col": "cc_arts_and_entertainment_nearest_max_1600", "name": "Arts & Ent.", "color": "#e377c2", "grade": "B", "unit": "m"},
            {"col": "cc_religious_nearest_max_1600", "name": "Religious", "color": "#7f7f7f", "grade": "B", "unit": "m"},
        ],
    },
]


# ============================================================================
# HELPERS
# ============================================================================


def classify_octants(df):
    """Assign each node to one of 8 octant types.

    Returns (classified_df, thresholds_dict).
    Thresholds:
      - Intensity: FSI >= 1.0
      - Continuity: frontage_max >= 0.75
      - Irregularity: orientation MAD >= 4.0
    """
    cols = AXIS_COLS
    valid = df.dropna(subset=list(cols.values())).copy()

    i_thresh = 1.0
    c_thresh = 0.75
    r_thresh = 4.0

    valid["I"] = np.where(valid[cols["intensity"]] >= i_thresh, "H", "L")
    valid["C"] = np.where(valid[cols["continuity"]] >= c_thresh, "H", "L")
    valid["R"] = np.where(valid[cols["irregularity"]] >= r_thresh, "H", "L")
    valid["octant"] = valid["I"] + valid["C"] + valid["R"]

    thresholds = {
        "intensity": i_thresh,
        "continuity": c_thresh,
        "irregularity": r_thresh,
    }
    return valid, thresholds


# Nearest-distance columns are censored at this network distance: NaN means
# "no instance within NEAREST_CAP m", not missing data.
NEAREST_CAP = 1600


def _fill_censored_nearest(frame):
    """Fill censored nearest-distance NaNs with the cap distance.

    Applied per city file, so a column absent from a city's cache (a coverage
    gap) stays NaN after concatenation and is excluded from statistics, while
    genuine beyond-reach streets are kept at the cap. Distance statistics such
    as medians and desert fractions are unaffected by the exact fill value as
    long as the relevant quantile lies below the cap.
    """
    suffix = f"_nearest_max_{NEAREST_CAP}"
    for c in frame.columns:
        if c.endswith(suffix):
            frame[c] = frame[c].fillna(float(NEAREST_CAP))
    return frame


def load_all_cached(columns=None):
    """Load all cached city parquets into one DataFrame.

    Parameters
    ----------
    columns : list[str] | None
        If given, only these columns are read from each parquet file.
        ``bounds_fid`` is always included.
    """
    if columns is not None:
        columns = sorted(set(columns) | {"bounds_fid"})
    files = sorted(CACHE_DIR.glob("city_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    if columns is not None:
        import pyarrow.parquet as pq

        for f in files:
            available = set(pq.read_schema(f).names)
            cols = [c for c in columns if c in available]
            frames.append(_fill_censored_nearest(pd.read_parquet(f, columns=cols)))
    else:
        for f in files:
            frames.append(_fill_censored_nearest(pd.read_parquet(f)))
    return pd.concat(frames, ignore_index=True)


def load_hq_fids(threshold=0.10):
    """Return set of bounds_fid values for cities with ML fraction <= threshold.

    Reads the building source audit CSV produced by s05b. Returns all city
    fids if the CSV is not found (graceful fallback).
    """
    csv_dir = Path(os.environ.get("T2E_DATA_DIR", "")) / "paper_data_outputs" / "csv"
    source_path = csv_dir / "building_source_metrics.csv"
    if not source_path.exists():
        print(f"  WARNING: {source_path} not found; returning all cities as HQ")
        files = sorted(CACHE_DIR.glob("city_*.parquet"))
        return {int(f.stem.replace("city_", "")) for f in files}
    source_df = pd.read_csv(source_path, dtype={"bounds_fid": int})
    return set(source_df.loc[source_df["ml_fraction"] <= threshold, "bounds_fid"])


def fmt_value(v, unit="m"):
    """Format a metric value for display labels."""
    if pd.isna(v):
        return "—"
    if unit == "%" or unit == "":
        return f"{v:.0%}" if v < 1 else f"{v:.0f}"
    if v >= 1000:
        return f"{v:,.0f}{unit}"
    if v >= 10:
        return f"{v:.0f}{unit}"
    if v >= 1:
        return f"{v:.1f}{unit}"
    return f"{v:.2f}{unit}"


# ============================================================================
# CUSTOM COMPOSITE OCTANT SYMBOLS
# ============================================================================
# Each octant is drawn as three layered sub-symbols:
#   Intensity  — filled ▲ (high) or outline ▽ (low)
#   Continuity — single block ■ (high) or 4 spaced squares □ (low)
#   Irregularity — organic curves (high) or regular grid (low)

# Symbol colours — dark grey for neutral presentation in figures.
# The coloured versions (_COL_I_COLOR etc.) can be used for the
# standalone axes diagram where colour encodes each dimension.
_COL_I = "#444444"  # intensity  (dark grey)
_COL_C = "#444444"  # continuity (dark grey)
_COL_R = "#444444"  # irregularity (dark grey)


def _draw_triangle(ax, cx, cy, sx, sy, is_high):
    """Intensity sub-symbol. sx/sy = half-sizes in data coords."""
    import matplotlib.pyplot as plt

    lw = 0.3
    if is_high:
        tri = plt.Polygon(
            [(cx, cy + sy), (cx - sx, cy - sy), (cx + sx, cy - sy)],
            closed=True,
            facecolor=_COL_I,
            edgecolor=_COL_I,
            lw=lw,
            zorder=10,
            clip_on=False,
        )
    else:
        tri = plt.Polygon(
            [(cx, cy - sy), (cx - sx, cy + sy), (cx + sx, cy + sy)],
            closed=True,
            facecolor="none",
            edgecolor=_COL_I,
            lw=lw,
            zorder=10,
            clip_on=False,
        )
    ax.add_patch(tri)


def _draw_blocks(ax, cx, cy, sx, sy, is_cont):
    """Continuity sub-symbol. sx/sy = half-sizes in data coords."""
    import matplotlib.patches as mpatches

    if is_cont:
        ax.add_patch(
            mpatches.Rectangle(
                (cx - sx, cy - sy),
                2 * sx,
                2 * sy,
                facecolor=_COL_C,
                edgecolor=_COL_C,
                lw=0.15,
                zorder=10,
                clip_on=False,
            )
        )
    else:
        bw = 2 * sx / 3
        bh = 2 * sy / 3
        cox = 2 * sx / 3
        coy = 2 * sy / 3
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx = cx + dx * cox - bw / 2
            by = cy + dy * coy - bh / 2
            ax.add_patch(
                mpatches.Rectangle(
                    (bx, by),
                    bw,
                    bh,
                    facecolor=_COL_C,
                    edgecolor=_COL_C,
                    lw=0.15,
                    zorder=10,
                    clip_on=False,
                )
            )


def _draw_irregularity(ax, cx, cy, sx, sy, is_high):
    """Irregularity sub-symbol: irregular curves (high) or regular grid (low)."""
    lw = 0.3
    if is_high:
        streets = [
            [
                (cx - sx, cy + sy * 0.2),
                (cx - sx * 0.4, cy + sy * 0.7),
                (cx + sx * 0.3, cy + sy * 0.9),
                (cx + sx * 0.8, cy + sy * 0.3),
            ],
            [
                (cx + sx * 0.8, cy + sy * 0.3),
                (cx + sx * 0.9, cy - sy * 0.2),
                (cx + sx * 0.4, cy - sy * 0.7),
                (cx - sx * 0.1, cy - sy * 0.9),
            ],
            [(cx - sx * 0.1, cy - sy * 0.9), (cx - sx * 0.6, cy - sy * 0.5), (cx - sx, cy + sy * 0.2)],
            [(cx - sx * 0.4, cy + sy * 0.7), (cx - sx * 0.1, cy + sy * 0.05), (cx + sx * 0.4, cy - sy * 0.7)],
            [(cx - sx * 0.6, cy - sy * 0.5), (cx - sx * 0.1, cy + sy * 0.05), (cx + sx * 0.8, cy + sy * 0.3)],
        ]
        for pts in streets:
            pts_arr = np.array(pts)
            t = np.linspace(0, 1, len(pts))
            t_fine = np.linspace(0, 1, 25)
            xs = np.interp(t_fine, t, pts_arr[:, 0])
            ys = np.interp(t_fine, t, pts_arr[:, 1])
            ax.plot(xs, ys, color=_COL_R, lw=lw, solid_capstyle="round", zorder=10, clip_on=False)
    else:
        n = 3
        for i in range(n + 1):
            frac = -1 + 2 * i / n
            ax.plot([cx - sx, cx + sx], [cy + sy * frac, cy + sy * frac], color=_COL_R, lw=lw, zorder=10, clip_on=False)
            ax.plot([cx + sx * frac, cx + sx * frac], [cy - sy, cy + sy], color=_COL_R, lw=lw, zorder=10, clip_on=False)


def draw_octant_symbol(ax, cx, cy, octant, sy=None, aspect=1.0, target_inches=0.06):
    """Draw the composite three-part symbol for an octant.

    Sub-symbols are stacked vertically (top to bottom):
      intensity (triangle)  — top
      continuity (blocks)   — middle
      irregularity (grid)   — bottom

    Parameters
    ----------
    ax : matplotlib Axes
    cx, cy : float — centre of the composite symbol (in data coordinates)
    octant : str — e.g. "HHH", "LLH"
    sy : float or None — half-size in y data units. If None, computed from
        target_inches for consistent physical size across all plots.
    aspect : float — x/y scaling factor (ignored if sy is None, since
        _sym_size_from_inches computes both sx and sy).
    target_inches : float — desired half-size of each sub-symbol in inches.
    """
    if sy is None:
        # Convert physical inches to data-coordinate half-sizes
        inv = ax.transData.inverted()
        fig = ax.get_figure()
        px = target_inches * fig.dpi
        o = inv.transform((0, 0))
        d = inv.transform((px, px))
        sx, sy = abs(d[0] - o[0]), abs(d[1] - o[1])
    else:
        sx = sy * aspect
    v_spacing = sy * 2.8
    _draw_triangle(ax, cx, cy + v_spacing, sx, sy, octant[0] == "H")
    _draw_blocks(ax, cx, cy, sx, sy, octant[1] == "H")
    _draw_irregularity(ax, cx, cy - v_spacing, sx, sy, octant[2] == "H")
