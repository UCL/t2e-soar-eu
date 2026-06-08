"""Shared constants for the Point A analysis pipeline."""

DEFAULT_CATEGORIES = [
    "retail",
    "eat_and_drink",
    "accommodation",
    "health_and_medical",
    "education",
    "business_and_services",
]

NEAREST_TOLERANCES_M = [50.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 1600.0]

# Spatial block bootstrap configuration.
# Block size is distance-adaptive: each catchment distance d uses d-metre blocks.
SPATIAL_BLOCK_BOOT_N = 200  # bootstrap replicates per city
SPATIAL_BLOCK_BOOT_ALPHA = 0.05  # significance level (95 % CI)
SPATIAL_BLOCK_BOOT_SEED = 42
