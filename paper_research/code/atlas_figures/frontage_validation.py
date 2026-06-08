"""
Appendix figure — Frontage ratio validation.

Panel A (top): Ridgeline KDE densities of city-median frontage by country.
               Face-validity check: reproduces known country-level housing
               traditions (NL/BE high, Nordic low, Romania higher than
               eastern neighbours).
Panel B (bottom): Street-level frontage distribution split by intensity
                  (low FSI vs high FSI), with the 0.75 threshold and the
                  density crossover at ~0.66 marked.  Shows that the
                  high-intensity subset ramps monotonically toward a mode
                  at 1 (perimeter-block fabric) while the low-intensity
                  subset has a broader interior peak and a declining tail.

Follows atlas plate styling conventions.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atlas_common import (
    AXIS_COLS,
    BOUNDARIES_PATH,
    DARK,
    GREY,
    GRID_COLOR,
    OUTPUT_DIR,
    apply_atlas_style,
    draw_title,
    load_all_cached,
    standard_margins,
)

apply_atlas_style()

C_THRESH = 0.75
FSI_THRESHOLD = 1.0
CROSSOVER = 0.66  # density crossover: where high-FSI overtakes low-FSI

# ── Load data ────────────────────────────────────────────────────────
print("Loading data …")
fr_col = AXIS_COLS["continuity"]
fsi_col = AXIS_COLS["intensity"]
df = load_all_cached(columns=[fr_col, fsi_col, AXIS_COLS["irregularity"], "bounds_fid"])
print(f"  {len(df):,} segments, {df['bounds_fid'].nunique()} cities")

# City-level medians (for Panel A)
city = df.groupby("bounds_fid")[fr_col].median().rename("med_fr").reset_index()

# Country labels (for Panel A)
bounds = pyogrio.read_dataframe(
    BOUNDARIES_PATH, columns=["bounds_fid", "label", "country"], read_geometry=False
)
bounds["bounds_fid"] = bounds["bounds_fid"].astype(int)
city = city.merge(bounds, on="bounds_fid")

# Street-level frontage split by intensity (for Panel B)
mask_complete = df[[fsi_col, fr_col]].notna().all(axis=1)
fsi_clean = df.loc[mask_complete, fsi_col].values
fr_clean = df.loc[mask_complete, fr_col].values
low_mask = fsi_clean < FSI_THRESHOLD
fr_low = fr_clean[low_mask]
fr_high = fr_clean[~low_mask]
print(f"  low-intensity (FSI < {FSI_THRESHOLD}):   {len(fr_low):>12,}")
print(f"  high-intensity (FSI >= {FSI_THRESHOLD}):  {len(fr_high):>12,}")

# Country ordering (by median, ascending) — filter to ≥3 cities
country_counts = city["country"].value_counts()
valid_countries = [c for c in country_counts.index if country_counts[c] >= 3]
country_medians = city[city["country"].isin(valid_countries)].groupby("country")["med_fr"].median()
country_order = country_medians.sort_values().index.tolist()

# ── Figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.5))
margins = standard_margins(fig, top_inches=0.45, bottom_inches=0.50, left_inches=1.55, right_inches=0.15)

gs = fig.add_gridspec(
    2, 1,
    height_ratios=[2.8, 1],
    hspace=0.22,
    **margins,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# =====================================================================
# PANEL A — Ridgeline KDE densities by country
# =====================================================================
ROW_HEIGHT = 0.16       # spacing between row baselines
UNIFORM_PEAK = 0.12     # all curves scaled to same peak height
KDE_BW = 0.11           # bandwidth for city-level medians on [0, 1]
MIN_KDE_CITIES = 5      # below this, draw rug ticks instead

n_countries = len(country_order)
xs = np.linspace(0, 1, 300)

for row_idx, country in enumerate(country_order):
    yo = row_idx * ROW_HEIGHT
    sub = city.loc[city["country"] == country, "med_fr"].dropna().values

    if len(sub) >= MIN_KDE_CITIES:
        kde = gaussian_kde(sub, bw_method=KDE_BW)
        density = kde(xs)
        peak = density.max()
        if peak > 0:
            density = density / peak * UNIFORM_PEAK

        ax_a.fill_between(xs, yo, yo + density, color=DARK, alpha=0.20, edgecolor="none", zorder=2)
        ax_a.plot(xs, yo + density, color=DARK, lw=0.5, alpha=0.55, zorder=3)
    else:
        # Rug ticks for small samples
        for v in sub:
            ax_a.plot([v, v], [yo, yo + UNIFORM_PEAK * 0.6], color=DARK, lw=0.8, alpha=0.4, zorder=2)

    # Baseline
    ax_a.plot([0, 1], [yo, yo], color=GRID_COLOR, lw=0.25, alpha=0.3, zorder=1)

    # Country median — drop line
    med = np.nanmedian(sub)
    if len(sub) >= MIN_KDE_CITIES:
        kde_at_med = kde(med)[0] / peak * UNIFORM_PEAK if peak > 0 else 0
    else:
        kde_at_med = UNIFORM_PEAK * 0.6
    ax_a.plot([med, med], [yo, yo + kde_at_med], color="#C0392B", ls=":", lw=0.7, alpha=0.7, zorder=4)

    # Country label
    n = len(sub)
    ax_a.text(
        -0.02, yo + 0.015,
        f"{country}  ({n})",
        fontsize=5.5, color=DARK, ha="right", va="bottom",
    )

# Grand median across all cities
grand_med = city["med_fr"].median()
ax_a.axvline(grand_med, color=GREY, linewidth=0.6, linestyle=":", alpha=0.4, zorder=0)

ax_a.set_xlim(-0.02, 1.0)
ax_a.set_ylim(-0.04, n_countries * ROW_HEIGHT + 0.06)
ax_a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_a.tick_params(axis="x", labelsize=6, colors=GREY)
ax_a.set_yticks([])
ax_a.set_xlabel("City-median frontage ratio", fontsize=7, color=DARK)
ax_a.spines["bottom"].set_visible(True)
ax_a.spines["bottom"].set_color(GRID_COLOR)
ax_a.spines["bottom"].set_linewidth(0.5)

ax_a.text(
    -0.02, 1.01, "A", fontsize=10, fontweight="bold", color=DARK,
    transform=ax_a.transAxes, va="bottom",
)
ax_a.text(
    0.03, 1.01, "City-median frontage by country", fontsize=7, color=GREY,
    transform=ax_a.transAxes, va="bottom",
)

# =====================================================================
# PANEL B — Street-level frontage distribution by intensity
# =====================================================================
# Overlaid density histograms of street-level frontage for low-FSI
# (< 1.0) and high-FSI (>= 1.0) subsets.  The y-axis is clipped above
# the interior so the distributions' interior structure is readable;
# the dominant mode near zero is truncated by design (its density far
# exceeds the interior; the clipped fraction is annotated in-panel).

LOW_COLOR = "#4A6FA5"   # muted blue
HIGH_COLOR = "#C0392B"  # atlas red (matches Panel A drop-line colour)

bins = np.arange(0.0, 1.02, 0.02)
hist_low, _ = np.histogram(fr_low, bins=bins, density=True)
hist_high, _ = np.histogram(fr_high, bins=bins, density=True)
centers = (bins[:-1] + bins[1:]) / 2

# Interior maximum across both subsets (exclude first two bins = zero spike)
interior_max = max(hist_low[2:].max(), hist_high[2:].max())
y_max_clip = interior_max * 1.15

# Draw low-intensity first (behind), then high-intensity (in front)
ax_b.fill_between(
    centers, 0, hist_low,
    color=LOW_COLOR, alpha=0.45,
    step="mid",
    label=f"Low intensity (FSI $<$ {FSI_THRESHOLD}):  n = {len(fr_low) / 1e6:.1f}M",
    zorder=2,
)
ax_b.plot(
    centers, hist_low,
    color=LOW_COLOR, lw=0.7, alpha=0.85,
    drawstyle="steps-mid",
    zorder=3,
)

ax_b.fill_between(
    centers, 0, hist_high,
    color=HIGH_COLOR, alpha=0.40,
    step="mid",
    label=f"High intensity (FSI $\\geq$ {FSI_THRESHOLD}):  n = {len(fr_high) / 1e6:.1f}M",
    zorder=4,
)
ax_b.plot(
    centers, hist_high,
    color=HIGH_COLOR, lw=0.7, alpha=0.90,
    drawstyle="steps-mid",
    zorder=5,
)

# Classification threshold (0.75) — dashed
ax_b.axvline(C_THRESH, color=DARK, ls="--", lw=0.8, alpha=0.75, zorder=6)
ax_b.text(
    C_THRESH + 0.008, y_max_clip * 0.96,
    f"threshold = {C_THRESH}",
    fontsize=5.5, color=DARK, va="top", ha="left",
)

# Density crossover (~0.66) — dotted grey
ax_b.axvline(CROSSOVER, color=GREY, ls=":", lw=0.7, alpha=0.75, zorder=6)
ax_b.text(
    CROSSOVER - 0.008, y_max_clip * 0.96,
    f"crossover $\\approx$ {CROSSOVER}",
    fontsize=5.5, color=GREY, va="top", ha="right",
)

# Annotate the clipped zero spike
frac_low_zero = float((fr_low < 0.02).mean())
frac_high_zero = float((fr_high < 0.02).mean())
ax_b.text(
    0.015, y_max_clip * 0.93,
    f"(zero mode truncated:\n{100 * frac_low_zero:.0f}% low, {100 * frac_high_zero:.0f}% high\nin [0, 0.02))",
    fontsize=5, color=GREY, va="top", ha="left", linespacing=1.2,
)

# Axes
ax_b.set_xlim(-0.02, 1.0)
ax_b.set_ylim(0, y_max_clip)
ax_b.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_b.tick_params(axis="x", labelsize=6, colors=GREY)
ax_b.tick_params(axis="y", labelsize=6, colors=GREY)
ax_b.set_xlabel("Street-level frontage ratio", fontsize=7, color=DARK)
ax_b.set_ylabel("Density (interior shown)", fontsize=7, color=DARK)
ax_b.spines["bottom"].set_visible(True)
ax_b.spines["bottom"].set_color(GRID_COLOR)
ax_b.spines["bottom"].set_linewidth(0.5)
ax_b.spines["left"].set_visible(True)
ax_b.spines["left"].set_color(GRID_COLOR)
ax_b.spines["left"].set_linewidth(0.5)

ax_b.legend(
    loc="upper left",
    bbox_to_anchor=(0.24, 0.98),
    fontsize=5.5,
    frameon=False,
    handlelength=1.5,
)

ax_b.text(
    -0.02, 1.04, "B", fontsize=10, fontweight="bold", color=DARK,
    transform=ax_b.transAxes, va="bottom",
)
ax_b.text(
    0.03, 1.04, "Street-level frontage distribution by intensity", fontsize=7, color=GREY,
    transform=ax_b.transAxes, va="bottom",
)

# ── Title ────────────────────────────────────────────────────────────
draw_title(fig, "Street-frontage ratio across European cities")

# ── Save ─────────────────────────────────────────────────────────────
out_dir = OUTPUT_DIR / "validation"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "frontage_validation.pdf"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
