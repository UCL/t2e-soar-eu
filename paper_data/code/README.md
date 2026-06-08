# Paper Data Workflow

This folder contains the reproducible generation scripts for `paper_data`.
Heavy inputs and cached outputs live under `$T2E_DATA_DIR` (set via environment variable, see root README) as configured in `config.py`.

## Running scripts

Each script is standalone. Run from the repo root:

```bash
cd paper_data/code && uv run python s05_audit_processed_outputs.py
cd paper_data/code && uv run python s06_write_paper_macros.py
```

## Scripts

```text
paper_data/code/
  config.py                              # paths, categories, plot settings
  pointa_constants.py                    # POI validation constants
  pointa_utils.py                        # POI validation utilities

  s00_prepare_reference_data.py          # prepare SIRENE + BAG registry data
  s01_run_reference_replication.py       # recompute POI accessibility from registries
  s02_summarise_node_agreement.py        # node-level agreement summaries
  s03_summarise_city_agreement.py        # city-level agreement summaries
  s04_build_support_matrix.py            # support matrix and minima tables

  s05_audit_processed_outputs.py         # per-column completeness coverage CSV
  s05a_audit_building_sources.py         # building footprint source composition CSVs
  s05b_backfill_frontage.py              # backfill street-frontage ratio into processed outputs
  s05c_building_source_figures.py        # building-source table and figure outputs
  s06_write_paper_macros.py              # LaTeX macros from data + coverage CSV

  s07_make_example_metric_figure.py      # example metric figure
  s08_make_poi_source_comparison_figure.py  # POI spatial comparison figure
  s09_make_pipeline_figure.py            # pipeline diagram
  s10_make_pointa_figures.py             # Point A figures and minima tables
```

## Notes

- Heavy intermediate outputs live in `OUTPUT_DIR` (`$T2E_DATA_DIR/paper_data_outputs`).
- Manuscript-ready artifacts live in `paper_data/outputs/`.
- `s05` produces `completeness_coverage.csv`; `s06` reads this for manuscript macros.
- `s05a` writes building-source CSVs under `OUTPUT_DIR/csv/`; `s05c` consumes them to produce `table_building_source_country.tex`, `table_building_metric_sensitivity.tex`, `fig_building_source_map.pdf`, and `fig_building_source_sensitivity.pdf`.
- `s04` loads frozen thresholds from `paper_data/pointa_thresholds.json` when present, otherwise falls back to hardcoded defaults.
- Scripts are numbered by dependency order but each runs independently.
