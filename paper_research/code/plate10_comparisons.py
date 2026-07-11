"""
Plate 10 — Comparisons.

Four paired comparisons using difference arrows, each reporting a consistent
10-variable set (FSI, Height, GSI, Frontage, MAD, StreetDens, Retail,
Eat & drink, Trees, Green).  Dots are colour-coded per entity; connecting
lines in charcoal.  Entity names and city details are left-aligned in the
left margin near the metric labels.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    BOUNDARIES_PATH,
    DARK,
    GREY,
    OUTPUT_DIR,
    SYM_INCHES,
    apply_atlas_style,
    draw_title,
    load_all_cached,
    standard_margins,
)

apply_atlas_style()

# ── Colours ───────────────────────────────────────────────────────────
COLOR_A = "#2166AC"  # blue for entity A
COLOR_B = "#C45B28"  # warm for entity B
LINE_GREY = "#B0ADA8"  # connecting lines

# ── All available metrics ─────────────────────────────────────────────
ALL_METRICS = {
    "fsi": {"col": "cc_block_far_median_400_wt", "label": "Floor space index (400m)", "higher_better": True},
    "height": {"col": "cc_mean_height_median_400_wt", "label": "Building height (400m)", "higher_better": True},
    "gsi": {"col": "cc_block_covered_ratio_median_400_wt", "label": "Ground space index (400m)", "higher_better": True},
    "swr": {"col": "frontage_max", "label": "Street-frontage ratio", "higher_better": True},
    "irregularity": {"col": "cc_orientation_mad_400_wt", "label": "Orientation MAD (400m)", "higher_better": None},
    "street_density": {"col": "cc_density_800", "label": "Network density (800m)", "higher_better": True},
    "retail": {"col": "cc_retail_nearest_max_1600", "label": "Retail distance", "higher_better": False},
    "eat_drink": {"col": "cc_eat_and_drink_nearest_max_1600", "label": "Eat & drink distance", "higher_better": False},
    "trees": {"col": "cc_trees_nearest_max_1600", "label": "Tree canopy distance", "higher_better": False},
    "green": {"col": "cc_green_nearest_max_1600", "label": "Green space distance", "higher_better": False},
}

# Standard 10-variable set reported for every pair (matches generate_macros.py emit_pair order)
STANDARD_METRICS = [
    "fsi",
    "height",
    "gsi",
    "swr",
    "irregularity",
    "street_density",
    "retail",
    "eat_drink",
    "trees",
    "green",
]

# ── Four paired comparisons ───────────────────────────────────────────
PAIRS = [
    {
        "a_label": "Nordic",
        "b_label": "Mediterranean",
        "a_cities": ["Oslo", "Bergen", "Helsinki", "Stockholm", "Copenhagen"],
        "b_cities": ["Barcelona", "Athens", "Madrid", "Naples", "Thessaloniki"],
        "metrics": STANDARD_METRICS,
    },
    {
        "a_label": "Netherlands",
        "b_label": "Belgium",
        "a_cities": ["Amsterdam", "Utrecht", "Rotterdam [The Hague]"],
        "b_cities": ["Brussels", "Ghent", "Antwerp"],
        "metrics": STANDARD_METRICS,
    },
    {
        "a_label": "Low Countries",
        "b_label": "Baltic",
        "a_countries": ["Netherlands", "Belgium"],
        "b_countries": ["Estonia", "Latvia", "Lithuania"],
        "metrics": STANDARD_METRICS,
    },
    {
        "a_label": "Poland",
        "b_label": "Romania",
        "a_countries": ["Poland"],
        "b_countries": ["Romania"],
        "metrics": STANDARD_METRICS,
    },
]


# ============================================================================
# DATA
# ============================================================================


def load_data():
    """Load segment data and boundaries."""
    print("Loading data...")
    all_cols = list({m["col"] for m in ALL_METRICS.values()} | set(AXIS_COLS.values()))
    df = load_all_cached(columns=all_cols)

    bounds = gpd.read_file(BOUNDARIES_PATH).to_crs(3035)
    bounds["cy"] = bounds.geometry.centroid.y
    df = df.merge(bounds[["bounds_fid", "label", "country", "cy"]], on="bounds_fid")
    print(f"  {len(df):,} segments across {df['bounds_fid'].nunique()} cities")
    return df, bounds


def compute_city_medians(df):
    """City-level medians for all metrics + axis cols."""
    all_cols = list({m["col"] for m in ALL_METRICS.values()} | set(AXIS_COLS.values()))
    available = [c for c in all_cols if c in df.columns]
    city_meds = df.groupby("bounds_fid")[available].median()
    city_meta = df.groupby("bounds_fid")[["country", "cy", "label"]].first()
    return city_meds.join(city_meta)


def z_score(city_meds):
    """Z-score all metric columns."""
    print("Z-scoring...")
    for m in ALL_METRICS.values():
        col = m["col"]
        if col not in city_meds.columns:
            continue
        vals = city_meds[col].dropna().values
        mu, sigma = np.nanmean(vals), np.nanstd(vals)
        if sigma < 1e-12:
            sigma = 1.0
        city_meds[col + "_z"] = (city_meds[col] - mu) / sigma


def _match_cities(city_meds, bounds, names):
    """Find city rows by label name (partial match)."""
    fids = bounds[bounds["label"].isin(names)]["bounds_fid"].values
    matched = city_meds[city_meds.index.isin(fids)]
    if len(matched) < len(names):
        # Try partial match for any missing
        for name in names:
            if not bounds[bounds["label"] == name]["bounds_fid"].isin(matched.index).any():
                partial = bounds[bounds["label"].str.contains(name, case=False, na=False)]
                if len(partial) > 0:
                    extra_fids = partial["bounds_fid"].values
                    matched = pd.concat([matched, city_meds[city_meds.index.isin(extra_fids)]])
    return matched


def assign_entities(city_meds, bounds, df=None):
    """For each pair, return (entity_a_df, entity_b_df)."""
    results = []
    for pair in PAIRS:
        ent_a = _get_entity(city_meds, bounds, pair, "a")
        ent_b = _get_entity(city_meds, bounds, pair, "b")
        results.append((ent_a, ent_b))
        print(f"  {pair['a_label']} ({len(ent_a)} cities) vs {pair['b_label']} ({len(ent_b)} cities)")
    return results


def _get_entity(city_meds, bounds, pair_def, side):
    cities = pair_def.get(f"{side}_cities")
    countries = pair_def.get(f"{side}_countries")
    if cities is not None:
        return _match_cities(city_meds, bounds, cities)
    if countries is not None:
        return city_meds[city_meds["country"].isin(countries)]
    raise ValueError(f"Pair '{pair_def[f'{side}_label']}' has neither cities nor countries")


# ============================================================================
# FIGURE
# ============================================================================


def _wrap_detail(text, max_chars=40):
    """Wrap a detail string into lines of at most max_chars, breaking at commas."""
    if text is None:
        return [""]
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in text.split(",")]
    lines = []
    current = parts[0]
    for p in parts[1:]:
        candidate = current + ", " + p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = p
    lines.append(current)
    return lines


def _entity_detail(pair_def, side):
    """Return a short detail string listing the cities or countries included."""
    cities = pair_def.get(f"{side}_cities")
    countries = pair_def.get(f"{side}_countries")
    label = pair_def[f"{side}_label"]
    if cities is not None:
        short = ["Rotterdam" if c == "Rotterdam [The Hague]" else c for c in cities]
        return ", ".join(short)
    if countries is not None:
        if label in ("Poland", "Romania"):
            return f"All {label.lower()} cities"
        if label == "Low Countries":
            return "Netherlands + Belgium"
        if label == "Baltic":
            return "Estonia, Latvia, Lithuania"
        return " + ".join(countries)
    return None


def fmt_value(v, col):
    """Format raw metric value for display."""
    if "nearest" in col or "density" in col:
        return f"{v:.0f}"
    elif "%" in col:
        return f"{v:.0%}" if v < 1 else f"{v:.1f}%"
    else:
        return f"{v:.2f}"


def build_figure(pairs_data, df_segments, bounds):
    """Build the arrow comparison figure."""
    n_pairs = len(PAIRS)

    # Compute entity z-score medians per pair per metric
    pair_data = []
    for _pi, (pair_def, (ent_a, ent_b)) in enumerate(zip(PAIRS, pairs_data, strict=False)):
        metrics = [ALL_METRICS[k] for k in pair_def["metrics"]]
        rows = []
        for m in metrics:
            zcol = m["col"] + "_z"
            za = ent_a[zcol].median() if zcol in ent_a.columns else np.nan
            zb = ent_b[zcol].median() if zcol in ent_b.columns else np.nan
            rows.append({"label": m["label"], "col": m["col"], "za": za, "zb": zb, "higher_better": m["higher_better"]})
        pair_data.append(rows)

    # Layout constants
    metric_spacing = 0.45
    pair_gap = 0.55
    fig_w = 7.5

    # Compute y positions
    y_positions = []
    pair_centre_y = []
    separator_ys = []
    y = 0
    for pi, rows in enumerate(pair_data):
        group_ys = []
        for mi in range(len(rows)):
            y_positions.append((pi, mi, y))
            group_ys.append(y)
            y -= metric_spacing
        pair_centre_y.append(np.mean(group_ys))
        if pi < len(pair_data) - 1:
            # Centre the separator in the gap between this group's last row
            # (y + metric_spacing) and the next group's first row (y - pair_gap).
            last_row_y = y + metric_spacing
            next_first_y = y - pair_gap
            separator_ys.append((last_row_y + next_first_y) / 2)
        y -= pair_gap

    total_rows = sum(len(pd) for pd in pair_data)
    fig_h = min(10.0, total_rows * 0.36 + n_pairs * 0.45 + 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    # X range from data
    all_z = []
    for rows in pair_data:
        for r in rows:
            if not np.isnan(r["za"]):
                all_z.append(r["za"])
            if not np.isnan(r["zb"]):
                all_z.append(r["zb"])
    x_pad = 0.5
    x_abs = max(abs(min(all_z)), abs(max(all_z))) + x_pad
    x_min, x_max = -x_abs, x_abs

    ax.set_xlim(x_min, x_max)
    all_y = [yp for _, _, yp in y_positions]
    ax.set_ylim(min(all_y) - 0.6, max(all_y) + 0.6)

    # Zero line
    ax.axvline(0, color=GREY, lw=0.4, alpha=0.3, zorder=0)

    # Group separators are drawn full-page-width in figure coordinates after
    # the layout is finalised (see below), so they span the whole page rather
    # than just the plotting axes.

    # Draw connecting lines, dots, and raw value labels
    dot_size = 40
    line_lw = 1.2

    for pi, mi, y_pos in y_positions:
        row = pair_data[pi][mi]
        za, zb = row["za"], row["zb"]
        if np.isnan(za) or np.isnan(zb):
            continue

        # Faint guide line per row
        ax.axhline(y_pos, color=GREY, lw=0.15, alpha=0.12, zorder=0)

        # Charcoal connecting line (fully opaque)
        ax.plot([za, zb], [y_pos, y_pos], color=LINE_GREY, lw=line_lw, solid_capstyle="round", zorder=3)

        # Dots
        ax.scatter(za, y_pos, s=dot_size, c=COLOR_A, zorder=4, edgecolors="white", linewidths=0.3)
        ax.scatter(zb, y_pos, s=dot_size, c=COLOR_B, zorder=4, edgecolors="white", linewidths=0.3)

        # Raw value labels
        raw_a = pairs_data[pi][0][row["col"]].median() if row["col"] in pairs_data[pi][0].columns else np.nan
        raw_b = pairs_data[pi][1][row["col"]].median() if row["col"] in pairs_data[pi][1].columns else np.nan
        if np.isnan(raw_a) or np.isnan(raw_b):
            continue
        fa_str = fmt_value(raw_a, row["col"])
        fb_str = fmt_value(raw_b, row["col"])
        if za <= zb:
            ax.text(
                za - 0.08,
                y_pos,
                fa_str,
                fontsize=5,
                color=COLOR_A,
                ha="right",
                va="center",
                fontweight="bold",
                zorder=5,
            )
            ax.text(
                zb + 0.08, y_pos, fb_str, fontsize=5, color=COLOR_B, ha="left", va="center", fontweight="bold", zorder=5
            )
        else:
            ax.text(
                za + 0.08, y_pos, fa_str, fontsize=5, color=COLOR_A, ha="left", va="center", fontweight="bold", zorder=5
            )
            ax.text(
                zb - 0.08,
                y_pos,
                fb_str,
                fontsize=5,
                color=COLOR_B,
                ha="right",
                va="center",
                fontweight="bold",
                zorder=5,
            )

    # Axes styling
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, colors=DARK, length=3, width=0.4)
    ax.set_xlabel("z-score (city medians)", fontsize=6, color=DARK, labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["bottom"].set_linewidth(0.4)

    # Margins — generous bottom for legend + description
    margins = standard_margins(
        fig,
        top_inches=0.45,
        bottom_inches=1.25,
        left_inches=2.60,
        right_inches=0.20,
    )
    plt.subplots_adjust(**margins)
    fig.canvas.draw()

    # ── Full-page-width separators between comparison groups ───
    from matplotlib.lines import Line2D

    for y_sep in separator_ys:
        disp = ax.transData.transform((0, y_sep))
        fy = fig.transFigure.inverted().transform(disp)[1]
        sep = Line2D([0.0, 1.0], [fy, fy], transform=fig.transFigure, color=GREY, lw=0.6, alpha=0.30, zorder=0)
        fig.add_artist(sep)

    # ── Left-side labels: entity names + city detail ───────────
    fig_w_in, fig_h_in = fig.get_size_inches()
    sym_size = SYM_INCHES * 0.80
    sym_gap_x = sym_size * 1.3 / fig_w_in  # horizontal spacing in figure fraction
    line_h = 0.015  # vertical spacing between lines in figure fraction

    bbox_ax = ax.get_position()
    entity_x = bbox_ax.x0 * 0.35  # left-of-centre in left margin
    total_sym_w = 3 * sym_gap_x

    # Vertical spacing (figure fraction) for the left-margin entity labels.
    name_to_city = line_h * 0.95  # heading -> its first city line (was ~1.9)
    city_line_step = line_h * 0.7  # between wrapped city lines
    entity_gap = line_h * 1.5  # entity A's last city -> entity B heading

    for pi, pair_def in enumerate(PAIRS):
        cy = pair_centre_y[pi]
        disp = ax.transData.transform((0, cy))
        fy = fig.transFigure.inverted().transform(disp)[1]

        # Entity A: heading + city detail (left-aligned)
        y_a_name = fy + line_h * 2.5
        fig.text(
            entity_x,
            y_a_name,
            pair_def["a_label"],
            fontsize=7,
            fontweight="bold",
            color=COLOR_A,
            ha="center",
            va="center",
        )
        a_detail = _entity_detail(pair_def, "a")
        a_lines = _wrap_detail(a_detail, max_chars=40) if a_detail else []
        for li, line in enumerate(a_lines):
            fig.text(
                entity_x,
                y_a_name - name_to_city - li * city_line_step,
                line,
                fontsize=5,
                fontweight="normal",
                color=GREY,
                ha="center",
                va="center",
            )

        # Entity B: heading + city detail, below entity A's block
        a_last_y = y_a_name - name_to_city - max(len(a_lines) - 1, 0) * city_line_step
        y_b_name = a_last_y - entity_gap
        fig.text(
            entity_x,
            y_b_name,
            pair_def["b_label"],
            fontsize=7,
            fontweight="bold",
            color=COLOR_B,
            ha="center",
            va="center",
        )
        b_detail = _entity_detail(pair_def, "b")
        b_lines = _wrap_detail(b_detail, max_chars=40) if b_detail else []
        for li, line in enumerate(b_lines):
            fig.text(
                entity_x,
                y_b_name - name_to_city - li * city_line_step,
                line,
                fontsize=5,
                fontweight="normal",
                color=GREY,
                ha="center",
                va="center",
            )

    # Metric sub-labels (right-aligned near axes)
    for pi, mi, y_pos in y_positions:
        disp = ax.transData.transform((0, y_pos))
        fy = fig.transFigure.inverted().transform(disp)[1]
        fig.text(
            bbox_ax.x0 - 0.008,
            fy,
            pair_data[pi][mi]["label"],
            fontsize=6,
            color=GREY,
            ha="right",
            va="center",
        )

    # Title
    draw_title(fig, "Paired City Comparisons")

    # Descriptive text
    fig.text(
        0.5,
        0.60 / fig_h_in,
        "Z-scored city-level medians.  Line length shows the magnitude of difference between paired entities.",
        fontsize=6,
        color=DARK,
        ha="center",
        va="bottom",
    )

    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df, bounds = load_data()
    city_meds = compute_city_medians(df)
    z_score(city_meds)
    pairs_data = assign_entities(city_meds, bounds, df=df)

    print("\nBuilding Plate 10 — Comparisons...")
    fig = build_figure(pairs_data, df, bounds)
    out = OUTPUT_DIR / "plate10_comparisons.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")
    print("\nDone.")
