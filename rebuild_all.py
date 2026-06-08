#!/usr/bin/env python3
"""Rebuild every paper output downstream of the processed metrics.

Run from the repo root with the project environment:

    uv run python rebuild_all.py

Raw ingestion and generate_metrics are intentionally NOT run here: their inputs
are fixed external releases and the processed GeoPackages already exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Force flags: True = rebuild from scratch, False = reuse existing outputs ──
FORCE_REGISTRY = False   # s00      SIRENE/BAG prep (slow; deterministic from raw files)
FORCE_POI = False        # s01-s04  POI source-substitution (slow; per-city recompute)
FORCE_FRONTAGE = True    # s05b     35 m frontage backfill into GeoPackages
FORCE_CACHE = True        # atlas    per-city parquet cache
WORKERS = "8"
# The derived layer (data-paper figures/tables/macros + atlas plates) always
# rebuilds; it is cheap and is the point of the run.

ROOT = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Resolve T2E_DATA_DIR, loading it from .env if not already in the environment."""
    val = os.environ.get("T2E_DATA_DIR")
    if not val:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("T2E_DATA_DIR"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["T2E_DATA_DIR"] = val
                    break
    if not val:
        sys.exit("T2E_DATA_DIR is not set and could not be read from .env")
    return Path(val)


DATA = data_dir()


def run(*argv: str) -> None:
    print(f"\n$ {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, *argv], cwd=ROOT, check=True)


def rm(*globs: str) -> None:
    for g in globs:
        base = Path(g)
        parent, pattern = (base.parent, base.name)
        for f in sorted(parent.glob(pattern)):
            f.unlink(missing_ok=True)
            print(f"  removed {f}")


def stage(n: int, label: str) -> None:
    print(f"\n===== [{n}/5] {label} =====", flush=True)


stage(1, "registry prep")
if FORCE_REGISTRY:
    rm(str(DATA / "validation/sirene_france.gpkg"),
       str(DATA / "validation/bag_netherlands.gpkg"))
run("paper_data/code/s00_prepare_reference_data.py")

stage(2, "POI validation")
if FORCE_POI:
    rm(str(DATA / "paper_data_outputs/replication/*.parquet"),
       str(DATA / "paper_data_outputs/csv/pointa_*.csv"),
       str(DATA / "paper_data_outputs/csv/pointa_support_thresholds_used.json"))
run("paper_data/code/s01_run_reference_replication.py")
run("paper_data/code/s02_summarise_node_agreement.py")
run("paper_data/code/s03_summarise_city_agreement.py")
run("paper_data/code/s04_build_support_matrix.py")

stage(3, "frontage backfill + cache")
if FORCE_FRONTAGE:
    run("paper_data/code/s05b_backfill_frontage.py", "--workers", WORKERS)
run("paper_research/code/cache_city_data.py", *(["--force"] if FORCE_CACHE else []))

stage(4, "data paper")
rm(str(ROOT / "paper_data/outputs/completeness_coverage.csv"),
   str(DATA / "paper_data_outputs/csv/building_source_*.csv"),
   str(ROOT / "paper_data/outputs/figures/fig_*.pdf"),
   str(ROOT / "paper_data/outputs/figures/fig_*.png"),
   str(ROOT / "paper_data/outputs/tables/table_*.tex"))
run("paper_data/code/s05_audit_processed_outputs.py", "--workers", WORKERS)
run("paper_data/code/s05a_audit_building_sources.py", "--workers", WORKERS)
run("paper_data/code/s05c_building_source_figures.py")
run("paper_data/code/s07_make_example_metric_figure.py")
run("paper_data/code/s08_make_poi_source_comparison_figure.py")
run("paper_data/code/s09_make_pipeline_figure.py")
run("paper_data/code/s10_make_pointa_figures.py")
run("paper_data/code/s06_write_paper_macros.py")

stage(5, "atlas")
run("paper_research/code/generate_macros.py")
run("paper_research/code/supplement_figures.py")
for p in (
    "plate1_exemplars", "plate2_3_ripples_lines", "plate4_buildings", "plate5_access",
    "plate6_service_desert", "plate7_unevenness", "plate8_demographics", "plate9_scanlines",
    "plate10_comparisons", "plate11_density_form", "plate12_density_access",
):
    run(f"paper_research/code/{p}.py")
run("paper_research/code/atlas_figures/frontage_validation.py")

print("\nDONE")
