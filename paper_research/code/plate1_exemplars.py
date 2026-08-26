"""
Plate 1 — Morphological Type Exemplars.

2×4 portrait grid of satellite tiles (one per octant), with composite
sub-symbols interleaved with local axis values beneath each image.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_common import (
    AXIS_COLS,
    BG,
    CACHE_DIR,
    DARK,
    GREY,
    OUTPUT_DIR,
    apply_atlas_style,
    draw_legend_row,
    draw_title,
    place_sub_symbol,
)

apply_atlas_style()

SAT_DIR = OUTPUT_DIR / "satellites"

SEARCH_RADIUS = 200  # metres
tr_4326_to_3035 = Transformer.from_crs(4326, 3035, always_xy=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper_data" / "code"))
from config import PROCESSED_DIR


def _compute_local_values(meta_list):
    """Compute fresh axis medians within 200m of each exemplar.

    Uses bbox-filtered geometry from the GeoPackage (fast) and looks up
    metric values from the parquet cache via GeoPackage FID (1-based
    sequential, so cache row = FID - 1).
    """
    import pyogrio

    axis_cols = list(AXIS_COLS.values())

    for m in meta_list:
        fid = m.get("bounds_fid")
        if fid is None:
            raise ValueError(f"Missing bounds_fid for {m['city']}")
        gpkg = PROCESSED_DIR / f"metrics_{fid}.gpkg.zip"
        cache_file = CACHE_DIR / f"city_{fid}.parquet"
        if not gpkg.exists():
            raise FileNotFoundError(f"GeoPackage not found: {gpkg}")
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {cache_file}")

        ex, ey = tr_4326_to_3035.transform(m["lon"], m["lat"])
        r = SEARCH_RADIUS + 50

        # Bbox-filtered geometry with FID as index
        gdf = pyogrio.read_dataframe(
            gpkg, layer="streets", columns=[],
            bbox=(ex - r, ey - r, ex + r, ey + r),
            fid_as_index=True,
        )
        if gdf.empty:
            raise ValueError(f"No streets near {m['city']}")

        dists = gdf.geometry.distance(Point(ex, ey))
        nearby_fids = gdf.index[dists < SEARCH_RADIUS]
        if len(nearby_fids) == 0:
            raise ValueError(f"No streets within {SEARCH_RADIUS}m of {m['city']}")

        # Look up metric values from cache using FID → row index
        cache_df = pd.read_parquet(cache_file, columns=axis_cols)
        # Sanity check: FID-1 positional lookup requires 1:1 row alignment
        n_features = pyogrio.read_info(gpkg, layer="streets")["features"]
        assert len(cache_df) == n_features, f"Cache/GPKG row mismatch for {m['city']}: {len(cache_df)} vs {n_features}"
        cache_rows = nearby_fids - 1  # FID is 1-based
        nearby_metrics = cache_df.iloc[cache_rows]

        for axis, col in AXIS_COLS.items():
            m[axis] = round(float(nearby_metrics[col].median()), 2)
        m["n_nearby"] = len(nearby_fids)


with open(OUTPUT_DIR / "satellite_metadata.json") as f:
    _meta_list = json.load(f)
_compute_local_values(_meta_list)
with open(OUTPUT_DIR / "satellite_metadata.json", "w") as f:
    json.dump(_meta_list, f, indent=2, ensure_ascii=False)
meta_lookup = {m["octant"]: m for m in _meta_list}

# 2 columns × 4 rows: Dense left, Light right
GRID = [
    ("HHH", "LHH"),  # Attached organic
    ("HHL", "LHL"),  # Attached rectilinear
    ("HLH", "LLH"),  # Freestanding organic
    ("HLL", "LLL"),  # Freestanding rectilinear
]


# Layout — tiles are ~2:1 (wide panoramic) to fit 7.5×10" page
TILE_W = 3.71
TILE_H = 1.81
PAD_X = 0.08
PAD_Y = 0.20
LABEL_H = 0.30
TITLE_H = 0.45
FOOTER_H = 0.50
SYM_PX = 28

NCOLS, NROWS = 2, 4
total_w = NCOLS * TILE_W + (NCOLS - 1) * PAD_X
total_h = TITLE_H + NROWS * (TILE_H + LABEL_H) + (NROWS - 1) * PAD_Y + FOOTER_H


def build_plate1():
    fig_w = 7.5
    fig_h = fig_w * total_h / total_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    draw_title(fig, "Eight Morphological Types")

    # Data-coord to figure-fraction converter (used for placing vector symbols)
    bbox_ax = ax.get_position()
    x0d, x1d = ax.get_xlim()
    y0d, y1d = ax.get_ylim()

    def _d2f(dx, dy):
        fx = bbox_ax.x0 + (dx - x0d) / (x1d - x0d) * bbox_ax.width
        fy = bbox_ax.y0 + (dy - y0d) / (y1d - y0d) * bbox_ax.height
        return fx, fy

    for ri, (oct_dense, oct_light) in enumerate(GRID):
        for ci, octant in enumerate([oct_dense, oct_light]):
            meta = meta_lookup.get(octant)
            if not meta:
                continue
            safe_city = meta["city"].replace(" ", "_").replace("/", "_")
            tp = SAT_DIR / f"sat_{octant}_{safe_city}.png"
            if not tp.exists():
                print(f"  Missing: {tp.name}")
                continue

            x0 = ci * (TILE_W + PAD_X)
            y_top = total_h - TITLE_H - ri * (TILE_H + LABEL_H + PAD_Y)
            y_bot = y_top - TILE_H

            # Crop to tile aspect (~2:1)
            tile = mpimg.imread(str(tp))
            th, tw = tile.shape[:2]
            target_ratio = TILE_W / TILE_H
            if tw / th < target_ratio:
                nh = int(tw / target_ratio)
                t = (th - nh) // 2
                tile = tile[t : t + nh, :]
            elif tw / th > target_ratio:
                nw = int(th * target_ratio)
                left = (tw - nw) // 2
                tile = tile[:, left : left + nw]
            ax.imshow(tile, extent=[x0, x0 + TILE_W, y_bot, y_top], aspect="auto", zorder=2)

            cx = x0 + TILE_W / 2

            # City, Country on top
            city_y = y_top + 0.03
            ax.text(
                cx,
                city_y,
                f"{meta['city']}, {meta['country']}",
                fontsize=8,
                fontweight="bold",
                color=DARK,
                ha="center",
                va="bottom",
            )

            # Symbol-value row underneath the image
            i_v = str(meta.get("intensity", "?"))
            c_v = str(meta.get("continuity", "?"))
            r_v = str(meta.get("irregularity", "?"))
            parts = ["intensity", "continuity", "irregularity"]
            vals = [i_v, c_v, r_v]

            row_y = y_bot - 0.14
            sym_w = 0.18
            txt_gap = 0.04
            pair_gap = 0.12
            txt_widths = [len(v) * 0.055 for v in vals]
            total_row_w = 3 * sym_w + 3 * txt_gap + sum(txt_widths) + 2 * pair_gap
            xc = cx - total_row_w / 2

            for si in range(3):
                sfx, sfy = _d2f(xc + sym_w / 2, row_y)
                place_sub_symbol(fig, sfx, sfy, octant, parts[si])
                xc += sym_w + txt_gap
                ax.text(xc, row_y, vals[si], fontsize=7, color=DARK, ha="left", va="center")
                xc += txt_widths[si] + pair_gap

            # Type name removed — legend key at bottom provides the vocabulary.

    # ── Legend key row (shared function) ─────────────────────────
    draw_legend_row(fig)

    # Footer
    ax.text(total_w / 2, 0.05, "Imagery: Esri World Imagery (ArcGIS)", fontsize=5, color=GREY, ha="center", va="bottom")

    return fig


if __name__ == "__main__":
    print("Building Plate 1 — Exemplars...")
    fig = build_plate1()
    out = OUTPUT_DIR / "plate1_exemplars.pdf"
    fig.savefig(out, dpi=300, facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")
