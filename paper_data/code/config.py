"""
Configuration for the SOAR-EU data paper (Scientific Data).

Contains categories, mappings, colors, plot settings, and paper-specific paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ============================================================================
# CATEGORIES
# ============================================================================

CATEGORIES = [
    "retail",
    "eat_and_drink",
    "health_and_medical",
    "education",
    "business_and_services",
]

CATEGORY_NAMES = {
    "retail": "Retail",
    "eat_and_drink": "Eat & Drink",
    "health_and_medical": "Health & Medical",
    "education": "Education",
    "business_and_services": "Business & Services",
}

# ============================================================================
# COLORS
# ============================================================================

QUINTILE_COLORS = {
    1: "#d73027",  # Q1 - lowest (red)
    2: "#fc8d59",  # Q2 - below average (orange)
    3: "#fee08b",  # Q3 - average (yellow)
    4: "#d9ef8b",  # Q4 - above average (light green)
    5: "#1a9850",  # Q5 - highest (green)
}

COUNTRY_COLORS = {
    "FR": "#3498db",  # Blue - France
    "NL": "#e74c3c",  # Red - Netherlands
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
# OFFICIAL DATA CATEGORY MAPPINGS
# ============================================================================

# SIRENE APE code mapping (France)
SIRENE_APE_MAPPING = {
    "retail": ["47"],
    "eat_and_drink": ["56.10A", "56.10B", "56.10C", "56.30Z"],
    "accommodation": ["55.10Z", "55.20Z", "55.30Z", "55.90Z"],
    "health_and_medical": [
        "86.10Z",
        "86.21Z",
        "86.22A",
        "86.22B",
        "86.22C",
        "86.23Z",
        "86.90A",
        "86.90B",
    ],
    "education": ["85.10Z", "85.20Z", "85.31Z", "85.32Z", "85.41Z", "85.42Z"],
    "business_and_services": [
        "62",
        "63",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "78",
        "80",
        "81",
        "82",
    ],
}

# BAG usage mapping (Netherlands)
BAG_USAGE_MAPPING = {
    "retail": ["winkelfunctie"],
    "eat_and_drink": ["bijeenkomstfunctie"],
    "accommodation": ["logiesfunctie"],
    "health_and_medical": ["gezondheidszorgfunctie"],
    "education": ["onderwijsfunctie"],
    "business_and_services": ["kantoorfunctie", "industriefunctie"],
}

# Overture category mapping
OVERTURE_CATEGORY_MAPPING = {
    "retail": ["retail"],
    "eat_and_drink": ["restaurant", "cafe", "bar"],
    "health_and_medical": ["health_and_medical"],
    "education": ["education"],
    "accommodation": ["accommodation"],
    "business_and_services": [
        "automotive",
        "beauty_and_spa",
        "pets",
        "real_estate",
        "travel",
        "home_service",
        "financial_service",
        "private_establishments_and_corporates",
        "business_to_business",
        "professional_services",
        "mass_media",
    ],
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


def assign_quintile(residual: float, residual_std: float) -> int:
    """Assign quality quintile based on standardized residual."""
    z = residual / residual_std if residual_std > 0 else 0
    if z < -1.0:
        return 1
    elif z < -0.25:
        return 2
    elif z < 0.25:
        return 3
    elif z < 1.0:
        return 4
    else:
        return 5


# ============================================================================
# PATHS (paper-specific)
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPER_DATA_DIR = PROJECT_ROOT / "paper_data"
if "T2E_DATA_DIR" not in os.environ:
    raise OSError("T2E_DATA_DIR environment variable is not set. See .env.example.")
DATA_DIR = Path(os.environ["T2E_DATA_DIR"])

# Input paths (shared data on external drive)
BOUNDS_PATH = DATA_DIR / "datasets" / "boundaries.gpkg"
BOUNDS_VALIDATION_PATH = DATA_DIR / "datasets" / "boundaries_validation.gpkg"
OVERTURE_DIR = DATA_DIR / "cities_data" / "overture"
VALIDATION_DIR = DATA_DIR / "validation"
PROCESSED_DIR = DATA_DIR / "cities_data" / "processed"

# Intermediate outputs (on external drive — large, regenerable)
OUTPUT_DIR = DATA_DIR / "paper_data_outputs"
CSV_DIR = OUTPUT_DIR / "csv"

# Final paper outputs (in repo — figures, tables, macros for the manuscript)
PAPER_DIR = PAPER_DATA_DIR / "outputs"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

# Ensure output directories exist
CSV_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
