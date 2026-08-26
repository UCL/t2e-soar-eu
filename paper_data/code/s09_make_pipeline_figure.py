"""
Generate pipeline flow diagram for SOAR dataset (vertical layout).

Top-to-bottom flow reflecting execution order:
  Row 1: GHS-UCDB → Stage 1 (boundary filtering)
  Row 2: Overture → Stage 5   |   Copernicus → Stages 2-4  (parallel, both use Stage 1 output)
  Row 3: Eurostat + all above → Stage 6 (metric computation)
  Row 4: Streets, Buildings, Blocks output layers

No external data dependencies — pure matplotlib drawing.

Usage:
    python paper_data/code/s09_make_pipeline_figure.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from config import FIG_DIR

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ============================================================================
# COLOURS
# ============================================================================

C_INPUT = "#D6EAF8"
C_INPUT_EDGE = "#5DADE2"
C_PROC = "#FCF3CF"
C_PROC_EDGE = "#F4D03F"
C_OUTPUT = "#D5F5E3"
C_OUTPUT_EDGE = "#58D68D"
C_LINE = "#B0B0B0"
C_LABEL = "#2C3E50"

# ============================================================================
# HELPERS
# ============================================================================


def draw_box(ax, cx, cy, w, h, label, sublabel=None, fc="#FFF", ec="#333"):
    """Draw a rounded box with centred label and optional sublabel."""
    box = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.01",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha="center", va="center", fontsize=8, fontweight="bold", color=C_LABEL, zorder=3)
        ax.text(
            cx, cy - 0.13, sublabel, ha="center", va="center", fontsize=6.5, color="#555555", style="italic", zorder=3
        )
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8, fontweight="bold", color=C_LABEL, zorder=3)


def conn(ax, x1, y1, x2, y2):
    """Draw a plain connector line."""
    ax.plot([x1, x2], [y1, y2], color=C_LINE, linewidth=0.9, solid_capstyle="round", zorder=1)


# ============================================================================
# FIGURE
# ============================================================================


def main() -> int:
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    # Box dimensions
    BW = 2.4  # standard box width
    BH = 0.55  # standard box height

    # Row y-centres (top to bottom)
    Y1 = 6.2  # Row 1: boundary filtering
    Y2 = 4.8  # Row 2: parallel ingestion
    Y3 = 3.2  # Row 3: metric computation
    Y4 = 1.8  # Row 4: outputs

    CX = 5.5  # horizontal centre

    # ========================================================================
    # ROW 1 — GHS-UCDB → BOUNDARY FILTERING (Stage 1 centred on CX so the
    #          line to Stage 6 goes straight down)
    # ========================================================================

    x_stg1 = CX  # centred on page
    x_ghsl = CX - BW - 0.8  # GHS-UCDB to its left with a gap

    draw_box(ax, x_ghsl, Y1, BW, BH, "GHS-UCDB", "Urban centre boundaries", fc=C_INPUT, ec=C_INPUT_EDGE)

    draw_box(
        ax, x_stg1, Y1, BW, BH, "1. Boundary Filtering", "Filter GHS-UCDB to EU, label cities", fc=C_PROC, ec=C_PROC_EDGE
    )

    conn(ax, x_ghsl + BW / 2, Y1, x_stg1 - BW / 2, Y1)

    # ========================================================================
    # ROW 2 — PARALLEL INGESTION
    #   Left half:  Overture input → Stage 5
    #   Right half: Copernicus input → Stages 2-4  (input left, processing right)
    # ========================================================================

    # Symmetric placement: processing boxes equidistant from CX
    INP_W = 1.8  # input box width
    SPREAD = 1.4  # offset of processing box centre from CX

    # Overture pair (left): input on far left, processing inward
    x_ov_pr = CX - SPREAD
    x_ov_in = x_ov_pr - BW / 2 - 0.15 - INP_W / 2

    draw_box(ax, x_ov_in, Y2, INP_W, BH, "Overture Maps", "Streets, POIs, buildings", fc=C_INPUT, ec=C_INPUT_EDGE)

    draw_box(
        ax, x_ov_pr, Y2, BW, BH, "5. Overture Extraction", "Network, POIs, buildings, infra.", fc=C_PROC, ec=C_PROC_EDGE
    )

    conn(ax, x_ov_in + INP_W / 2, Y2, x_ov_pr - BW / 2, Y2)

    # Copernicus pair (right): processing inward, input on far right
    x_cop_pr = CX + SPREAD
    x_cop_in = x_cop_pr + BW / 2 + 0.15 + INP_W / 2

    draw_box(
        ax,
        x_cop_pr,
        Y2,
        BW,
        BH,
        "2\u20134. Copernicus Ingestion",
        "Blocks, tree canopy, heights",
        fc=C_PROC,
        ec=C_PROC_EDGE,
    )

    draw_box(ax, x_cop_in, Y2, INP_W, BH, "Copernicus", "Land cover, trees, heights", fc=C_INPUT, ec=C_INPUT_EDGE)

    conn(ax, x_cop_in - INP_W / 2, Y2, x_cop_pr + BW / 2, Y2)

    # Stage 1 feeds both ingestion bundles (symmetric fan-out)
    conn(ax, x_stg1 - 0.8, Y1 - BH / 2, x_ov_pr, Y2 + BH / 2)
    conn(ax, x_stg1 + 0.8, Y1 - BH / 2, x_cop_pr, Y2 + BH / 2)

    # ========================================================================
    # ROW 3 — METRIC COMPUTATION (wide box, fan-in from above + Eurostat)
    # ========================================================================

    w6 = 4.0

    draw_box(
        ax,
        CX,
        Y3,
        w6,
        BH,
        "6. Metric Computation",
        "Centrality, accessibility, morphology, green space, demographics",
        fc=C_PROC,
        ec=C_PROC_EDGE,
    )

    # Eurostat input to the left
    x_eu = CX - w6 / 2 - 1.5

    draw_box(ax, x_eu, Y3, 2.0, BH, "Eurostat Census", "Population, demographics", fc=C_INPUT, ec=C_INPUT_EDGE)

    conn(ax, x_eu + 2.0 / 2, Y3, CX - w6 / 2, Y3)

    # Ingestion → Stage 6
    conn(ax, x_ov_pr, Y2 - BH / 2, CX - 0.5, Y3 + BH / 2)
    conn(ax, x_cop_pr, Y2 - BH / 2, CX + 0.5, Y3 + BH / 2)

    # Stage 1 also feeds Stage 6 directly (boundaries.gpkg used for spatial joins)
    # Both centred at CX — straight vertical drop
    conn(ax, CX, Y1 - BH / 2, CX, Y3 + BH / 2)

    # ========================================================================
    # ROW 4 — OUTPUT LAYERS (three boxes side by side)
    # ========================================================================

    xo = [2.8, 5.5, 8.2]

    draw_box(ax, xo[0], Y4, BW, BH, "Streets Layer", "100+ metrics, multi-scale", fc=C_OUTPUT, ec=C_OUTPUT_EDGE)

    draw_box(ax, xo[1], Y4, BW, BH, "Buildings Layer", "Morphological attributes", fc=C_OUTPUT, ec=C_OUTPUT_EDGE)

    draw_box(ax, xo[2], Y4, BW, BH, "Blocks Layer", "Coverage ratios", fc=C_OUTPUT, ec=C_OUTPUT_EDGE)

    for x in xo:
        conn(ax, CX, Y3 - BH / 2, x, Y4 + BH / 2)

    # ========================================================================
    # Annotation
    # ========================================================================

    ax.text(
        CX,
        0.9,
        "GeoPackage  \u00b7  EPSG:3035  \u00b7  626 cities \u00d7 3 layers each",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5),
        zorder=2,
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    out_pdf = FIG_DIR / "fig_pipeline.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close()
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
