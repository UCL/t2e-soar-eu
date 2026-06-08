"""Shared display constants, plot styling, and column definitions for the Atlas paper."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ============================================================================
# PATHS — external drive for source data and temp caches
# ============================================================================

if "T2E_DATA_DIR" not in os.environ:
    raise OSError("T2E_DATA_DIR environment variable is not set. See .env.example.")
DATA_DIR = Path(os.environ["T2E_DATA_DIR"])
METRICS_DIR = DATA_DIR / "cities_data" / "processed"
TEMP_BASE = DATA_DIR / "temp_egs"

# ============================================================================
# CATEGORIES
# ============================================================================

CATEGORY_NAMES = {
    "retail": "Retail",
    "eat_and_drink": "Eat & Drink",
    "health_and_medical": "Health & Medical",
    "education": "Education",
    "business_and_services": "Business & Services",
    "accommodation": "Accommodation",
}

CATEGORY_COLORS = {
    "retail": "#1f77b4",
    "eat_and_drink": "#2ca02c",
    "health_and_medical": "#d62728",
    "education": "#9467bd",
    "business_and_services": "#ff7f0e",
    "accommodation": "#8c564b",
}

# ============================================================================
# PLOT SETTINGS
# ============================================================================

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

MAP_EXTENT = {
    "x_min": 2.6e6,
    "x_max": 5.9e6,
    "y_min": 1.55e6,
    "y_max": 4.05e6,
}


def apply_plot_style():
    """Apply standard plot style settings."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)


# ============================================================================
# VARIABLE LABELS (raw column name -> human-readable)
# ============================================================================

VARIABLE_LABELS = {
    # Centrality — closeness
    "cc_beta_400": "Closeness (400m)",
    "cc_beta_800": "Closeness (800m)",
    "cc_beta_1200": "Closeness (1,200m)",
    "cc_beta_1600": "Closeness (1,600m)",
    "cc_beta_4800": "Closeness (4,800m)",
    # Centrality — betweenness
    "cc_betweenness_beta_400": "Betweenness (400m)",
    "cc_betweenness_beta_800": "Betweenness (800m)",
    "cc_betweenness_beta_1200": "Betweenness (1,200m)",
    "cc_betweenness_beta_1600": "Betweenness (1,600m)",
    "cc_betweenness_beta_4800": "Betweenness (4,800m)",
    # Green space
    "cc_green_nearest_max_1600": "Green space distance",
    "cc_trees_nearest_max_1600": "Tree canopy distance",
    "cc_green_area_sum_400_nw": "Green area (400m)",
    "cc_trees_area_sum_400_nw": "Tree area (400m)",
    # Census / demographics
    "density": "Population density",
    "y_lt15": "Population <15",
    "y_1564": "Working age (15\u201364)",
    "y_ge65": "Population 65+",
    "y_ge65_%": "Population 65+ (%)",
    "emp": "Employment ratio",
    # Morphology (400m, distance-weighted medians; non-weighted counts)
    "cc_building_400_nw": "Building count",
    "cc_block_400_nw": "Block count",
    "cc_mean_height_median_400_wt": "Mean height",
    "cc_mean_height_mad_400_wt": "Height variation",
    "cc_area_median_400_wt": "Building area",
    "cc_fractal_dimension_median_400_wt": "Fractal dimension",
    "cc_block_covered_ratio_median_400_wt": "Block coverage (GSI)",
    "cc_shared_wall_ratio_median_400_wt": "Shared wall ratio",
    "frontage_max": "Street-frontage ratio",
    # Spacematrix block metrics (400m)
    "cc_block_far_median_400_wt": "Floor area ratio (FSI)",
    "cc_block_osr_median_400_wt": "Open space ratio (OSR)",
    "cc_block_l_median_400_wt": "Mean floors (L)",
    "cc_block_mean_height_median_400_wt": "Block mean height",
    # Building orientation
    "cc_orientation_mad_400_wt": "Orientation variation (MAD)",
}

# Feature type classification for colour-coding
FEATURE_TYPE_MAP = {
    "cc_beta_400": "centrality",
    "cc_beta_800": "centrality",
    "cc_beta_1200": "centrality",
    "cc_beta_1600": "centrality",
    "cc_beta_4800": "centrality",
    "cc_betweenness_beta_400": "centrality",
    "cc_betweenness_beta_800": "centrality",
    "cc_betweenness_beta_1200": "centrality",
    "cc_betweenness_beta_1600": "centrality",
    "cc_betweenness_beta_4800": "centrality",
    "density": "demographics",
    "y_lt15": "demographics",
    "y_1564": "demographics",
    "y_ge65": "demographics",
    "emp": "demographics",
}

FEATURE_TYPE_COLORS = {
    "centrality": "#4878CF",
    "demographics": "#D65F5F",
    "morphology": "#6A994E",
}

# ============================================================================
# MORPHOLOGY COLUMNS
# ============================================================================

MORPH_COLS = [
    "cc_building_400_nw",
    "cc_block_400_nw",
    "cc_mean_height_median_400_wt",
    "cc_mean_height_mad_400_wt",
    "cc_area_median_400_wt",
    "cc_fractal_dimension_median_400_wt",
    "cc_block_covered_ratio_median_400_wt",
    "cc_shared_wall_ratio_median_400_wt",
    "frontage_max",
    "cc_block_far_median_400_wt",
    "cc_block_osr_median_400_wt",
    "cc_block_l_median_400_wt",
    "cc_block_mean_height_median_400_wt",
    "cc_orientation_mad_400_wt",
    # Additional building morphometrics
    "cc_compactness_median_400_wt",
    "cc_perimeter_median_400_wt",
    "cc_corners_median_400_wt",
    "cc_volume_median_400_wt",
    "cc_block_perimeter_median_400_wt",
    "cc_orientation_median_400_wt",
]

MORPH_COL_NAMES = [
    "Building count",
    "Block count",
    "Mean height",
    "Height variation",
    "Building area",
    "Fractal dimension",
    "Block coverage (GSI)",
    "Shared wall ratio",
    "Street-frontage ratio",
    "Floor area ratio (FSI)",
    "Open space ratio (OSR)",
    "Mean floors (L)",
    "Block mean height",
    "Orientation variation (MAD)",
    "Compactness",
    "Perimeter",
    "Corners",
    "Volume",
    "Block perimeter",
    "Orientation median",
]

MORPH_LABELS = dict(zip(MORPH_COLS, MORPH_COL_NAMES, strict=True))

# ============================================================================
# THREE-AXIS MORPHOLOGICAL GRAMMAR
# ============================================================================
# The organising framework for the Atlas paper: each street segment is
# described by Intensity, Continuity, and Irregularity.

AXIS_COLS = {
    "intensity": "cc_block_far_median_400_wt",  # FSI — how much is built
    "continuity": "frontage_max",  # street-frontage continuity (source-robust)
    "irregularity": "cc_orientation_mad_400_wt",  # planned vs organic layout
}

AXIS_LABELS = {
    "intensity": "Intensity (FSI)",
    "continuity": "Continuity (street-frontage ratio)",
    "irregularity": "Irregularity (orientation MAD)",
}

# Legacy SWR axis column, retained for comparison and backward compatibility
SWR_COL = "cc_shared_wall_ratio_median_400_wt"


def label_features(names):
    """Map a list of raw column names to human-readable labels."""
    return [VARIABLE_LABELS.get(n, n) for n in names]


# ============================================================================
# SHARED CITY DATA LOADER
# ============================================================================

SHARED_CACHE_DIR = TEMP_BASE / "shared_cache"


def load_city_metrics(bounds_fid, columns=None, *, metrics_dir=None, cache_dir=None):
    """Load metrics for a single city, preferring parquet cache over gpkg.

    Parameters
    ----------
    bounds_fid : int
        City identifier.
    columns : list[str] or None
        Columns to return.  If None, return all available columns.
    metrics_dir : str or Path or None
        Path to gpkg.zip directory (fallback).  If None, uses the
        default external drive path.
    cache_dir : str or Path or None
        Path to shared parquet cache.  If None, uses SHARED_CACHE_DIR.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with requested columns (no geometry), or None if the
        city is unavailable.
    """
    from pathlib import Path

    import pandas as pd

    if cache_dir is None:
        cache_dir = SHARED_CACHE_DIR
    cache_file = Path(cache_dir) / f"city_{bounds_fid}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if columns is not None:
            available = [c for c in columns if c in df.columns]
            df = df[available + ["bounds_fid"]]
        return df

    # Fallback: load from gpkg
    if metrics_dir is None:
        return None

    metrics_dir = Path(metrics_dir)
    metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg.zip"
    if not metrics_file.exists():
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
    if not metrics_file.exists():
        return None

    try:
        import geopandas as gpd

        kw = {"layer": "streets"}
        if columns is not None:
            kw["columns"] = columns
        gdf = gpd.read_file(metrics_file, **kw)
        # Drop geometry — callers that need it should load gpkg directly
        df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        df["bounds_fid"] = bounds_fid
        return df
    except Exception:
        return None
