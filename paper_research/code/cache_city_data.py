"""
Cache city metrics as lightweight parquet files for atlas research.

Reads each city's gpkg.zip once, extracts columns needed for atlas analysis,
writes one parquet per city to the shared cache directory.  Incremental:
skips cities already cached unless --force is passed.

Usage:
    python paper_research/code/cache_city_data.py           # incremental
    python paper_research/code/cache_city_data.py --force    # re-cache all
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_config import AXIS_COLS, METRICS_DIR, MORPH_COLS, SHARED_CACHE_DIR

CACHE_DIR = Path(SHARED_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_NODES = 100

# ============================================================================
# COLUMNS TO CACHE
# ============================================================================

# Morphology — all 13 MORPH_COLS from shared_config (includes 3-axis cols)
# Already defined in shared_config.py

# Centrality
CENTRALITY_COLS = (
    [f"cc_beta_{d}" for d in (400, 800, 1200, 1600, 4800)]
    + [f"cc_betweenness_beta_{d}" for d in (400, 800, 1200, 1600, 4800)]
    + [f"cc_density_{d}" for d in (400, 800, 1200)]
    + ["cc_cycles_1200", "cc_harmonic_1200"]
)

# Land use accessibility
LU_CATEGORIES = [
    "accommodation",
    "active_life",
    "arts_and_entertainment",
    "attractions_and_activities",
    "business_and_services",
    "eat_and_drink",
    "education",
    "health_and_medical",
    "public_services",
    "religious",
    "retail",
]
INFRA_CATEGORIES = ["street_furn", "parking", "transport"]

LU_COLS = (
    [f"cc_{c}_nearest_max_1600" for c in LU_CATEGORIES]
    + [f"cc_{c}_400_nw" for c in LU_CATEGORIES]
    + [f"cc_{c}_nearest_max_1600" for c in INFRA_CATEGORIES]
    + ["cc_hill_q0_400_nw", "cc_hill_q1_400_nw", "cc_hill_q2_400_nw"]
)

# Green space
GREEN_COLS = [
    "cc_green_nearest_max_1600",
    "cc_trees_nearest_max_1600",
    "cc_green_area_sum_400_nw",
    "cc_trees_area_sum_400_nw",
]

# Census / demographics
CENSUS_COLS = [
    "density",
    "emp",
    "emp_%",
    "y_lt15",
    "y_lt15_%",
    "y_1564",
    "y_1564_%",
    "y_ge65",
    "y_ge65_%",
    "m_%",
    "f_%",
]

# 3-axis columns (400_wt) — subset of MORPH_COLS, included for completeness
AXIS_CACHE_COLS = list(AXIS_COLS.values())

ALL_COLS = sorted(set(MORPH_COLS + AXIS_CACHE_COLS + CENTRALITY_COLS + LU_COLS + GREEN_COLS + CENSUS_COLS))

# ============================================================================
# MAIN
# ============================================================================

force = "--force" in sys.argv

print(f"Metrics dir:  {METRICS_DIR}")
print(f"Cache dir:    {CACHE_DIR}")
print(f"Columns:      {len(ALL_COLS)}")
print(f"Force:        {force}")
print()

# Discover available metrics files
metrics_files = sorted(METRICS_DIR.glob("metrics_*.gpkg.zip"))
print(f"Available cities: {len(metrics_files)}")

cached = skipped_exists = skipped_small = errors = 0
t0 = time.time()

for mf in tqdm(metrics_files, desc="Caching"):
    bounds_fid = int(mf.name.split("_", 1)[1].split(".")[0])
    cache_file = CACHE_DIR / f"city_{bounds_fid}.parquet"

    if cache_file.exists() and not force:
        skipped_exists += 1
        continue

    try:
        gdf = gpd.read_file(mf, columns=ALL_COLS, layer="streets")
    except Exception as e:
        tqdm.write(f"  ERROR {bounds_fid}: {e}")
        errors += 1
        continue

    if len(gdf) < MIN_NODES:
        skipped_small += 1
        continue

    available = [c for c in ALL_COLS if c in gdf.columns]
    df = pd.DataFrame(gdf[available])
    df["bounds_fid"] = bounds_fid
    df.to_parquet(cache_file, index=False)
    cached += 1

elapsed = time.time() - t0

print()
print(f"  Cached:          {cached}")
print(f"  Already cached:  {skipped_exists}")
print(f"  Too few nodes:   {skipped_small}")
print(f"  Errors:          {errors}")
print(f"  Time:            {elapsed:.0f}s")

# Write manifest
manifest = pd.DataFrame(
    {"bounds_fid": [int(f.stem.replace("city_", "")) for f in sorted(CACHE_DIR.glob("city_*.parquet"))]}
)
manifest.to_csv(CACHE_DIR / "manifest.csv", index=False)
print(f"  Manifest:        {len(manifest)} cities")
