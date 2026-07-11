"""Generate the metric-families overview figure for the data paper.

Schematic (no data inputs): open sources on the left, street-segment metric
families on the right, on a shared six-row grid. Each family box has three
zones: title (top), contents (middle, up to two lines), and a meta row
(bottom) carrying the source tags and the scale annotation.
Writes PDF (manuscript) and PNG (preview) to paper_data/outputs/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#c3c2b7"
BOX_FILL = "#f9f9f7"

SOURCES = [
    ("S1", "#2a78d6", "Overture Maps", "street network (2026)"),
    ("S2", "#1baf7a", "Overture Maps", "POIs & infrastructure (2026)"),
    ("S3", "#eda100", "Overture Maps", "building footprints (2026)"),
    ("S4", "#008300", "Copernicus", "Urban Atlas & Street\nTree Layer (2021)"),
    ("S5", "#4a3aa7", "Copernicus", "Digital Height Model (2012)"),
    ("S6", "#e34948", "Eurostat", "Census Grid, 1 km² (2021)"),
]
COLOURS = {tag: colour for tag, colour, _, _ in SOURCES}

FAMILIES = [
    (
        "Network centrality",
        "closeness (beta, harmonic, Hillier, farness),\nbetweenness, segment density, cycles",
        "6 thresholds · 400–9,600 m",
        ["S1"],
    ),
    (
        "Land-use & infrastructure accessibility",
        "counts (unweighted + distance-weighted), nearest\ndistance; 11 land-use + 3 infrastructure categories",
        "5 thresholds · 200–1,600 m",
        ["S1", "S2"],
    ),
    (
        "Mixed-use diversity",
        "Hill numbers q = 0, 1, 2\n(unweighted + distance-weighted)",
        "5 thresholds · 200–1,600 m",
        ["S1", "S2"],
    ),
    (
        "Building & block morphology",
        "contextual building morphometrics; block Spacematrix\n(GSI, FSI, OSR, storeys); street-frontage ratio",
        "200 + 400 m",
        ["S1", "S3", "S4", "S5"],
    ),
    (
        "Green-space & tree-canopy proximity",
        "nearest distance;\ngreen / canopy area within catchment",
        "areas 200–800 m · nearest to 1,600 m",
        ["S1", "S4"],
    ),
    (
        "Demographics",
        "population, density, age, employment,\nnationality, migration (counts + shares)",
        "interpolated from 1 km² grid",
        ["S6"],
    ),
]

# Shared six-row grid.
ROW_TOP = 0.865
ROW_H = 0.127
ROW_GAP = 0.006

SRC_X, SRC_W = 0.02, 0.26
FAM_X, FAM_W = 0.42, 0.565


def row_y(i: int) -> float:
    return ROW_TOP - i * (ROW_H + ROW_GAP) - ROW_H


def rounded_box(ax, x, y, w, h, fill=BOX_FILL, edge=HAIRLINE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.7,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Framing note and column headers.
    ax.text(
        0.5,
        0.985,
        "GHS-UCDB R2024A boundaries define the 626 urban centres · all layers projected to EPSG:3035",
        ha="center",
        va="top",
        fontsize=7.0,
        color=INK_2,
    )
    ax.text(
        SRC_X + SRC_W / 2, 0.915, "Open sources", ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold"
    )
    ax.text(
        FAM_X + FAM_W / 2,
        0.915,
        "Streets layer · pre-computed metric families (>100 columns per street)",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
        fontweight="bold",
    )

    # Source boxes (left column, same row grid as the families).
    for i, (tag, colour, org, detail) in enumerate(SOURCES):
        y = row_y(i)
        rounded_box(ax, SRC_X, y, SRC_W, ROW_H)
        ax.add_patch(Rectangle((SRC_X + 0.014, y + ROW_H / 2 - 0.018), 0.016, 0.036, facecolor=colour, edgecolor="none"))
        ax.text(SRC_X + 0.040, y + ROW_H - 0.026, org, ha="left", va="center", fontsize=7.4, color=INK, fontweight="bold")
        ax.text(
            SRC_X + 0.040,
            y + ROW_H - 0.050,
            detail,
            ha="left",
            va="top",
            fontsize=6.2,
            color=INK_2,
            linespacing=1.25,
        )

    # Pipeline arrow, centred between the columns.
    mid_y = (ROW_TOP + row_y(len(SOURCES) - 1)) / 2
    ax.add_patch(
        FancyArrow(
            SRC_X + SRC_W + 0.018,
            mid_y,
            FAM_X - SRC_X - SRC_W - 0.036,
            0,
            width=0.0012,
            head_width=0.016,
            head_length=0.014,
            facecolor=INK_2,
            edgecolor="none",
            length_includes_head=True,
        )
    )
    arrow_cx = (SRC_X + SRC_W + FAM_X) / 2
    ax.text(arrow_cx, mid_y + 0.042, "six-stage\npipeline", ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.25)
    ax.text(arrow_cx, mid_y - 0.046, "fixed\nparameters", ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.25)

    # Family boxes (right column): title / contents / meta row.
    for i, (name, contents, scale, tags) in enumerate(FAMILIES):
        y = row_y(i)
        rounded_box(ax, FAM_X, y, FAM_W, ROW_H, fill="#ffffff")
        ax.text(FAM_X + 0.014, y + ROW_H - 0.019, name, ha="left", va="center", fontsize=7.4, color=INK, fontweight="bold")
        ax.text(
            FAM_X + 0.014,
            y + ROW_H - 0.041,
            contents,
            ha="left",
            va="top",
            fontsize=6.3,
            color=INK_2,
            linespacing=1.3,
        )
        # Meta row: source tags at left, scale annotation at right.
        for j, tag in enumerate(tags):
            ax.add_patch(
                Rectangle(
                    (FAM_X + 0.014 + j * 0.019, y + 0.010),
                    0.011,
                    0.020,
                    facecolor=COLOURS[tag],
                    edgecolor="none",
                )
            )
        ax.text(
            FAM_X + FAM_W - 0.014,
            y + 0.020,
            scale,
            ha="right",
            va="center",
            fontsize=6.2,
            color=MUTED,
            style="italic",
        )

    # Footnote.
    ax.text(
        0.5,
        0.012,
        "Each city GeoPackage also carries a buildings layer (per-footprint morphometrics) and a blocks layer "
        "(Spacematrix indicators);\ncoloured squares mark the sources contributing to each family, matching the source boxes at left.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
        linespacing=1.35,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_metric_families.{ext}", dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_metric_families.pdf'}")


if __name__ == "__main__":
    main()
