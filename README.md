# t2e-soar-eu

SOAR-EU (Scalable Open Automatable Reproducible — European Urban) is a pedestrian-scale urban data model for the EU-funded TWIN2EXPAND project. It produces standardised, multi-scale spatial metrics at street-segment level for 626 urban centres across 28 European countries (EU-27 except Cyprus, plus Norway and Switzerland; Liechtenstein has no qualifying urban centre).

## Installation

```bash
uv sync
```

Project configuration is managed using `pyproject.toml`. [uv](https://docs.astral.sh/uv/) is used for package management: `uv sync` installs all dependencies into a `.venv` folder.

## Configuration

All scripts read the data root from the `T2E_DATA_DIR` environment variable. Set it in your `.env` file or export it in your shell:

```bash
# Option 1: add to .env (recommended)
T2E_DATA_DIR=/path/to/your/data

# Option 2: export in your shell
export T2E_DATA_DIR=/path/to/your/data
```

Copy `.env.example` to `.env` and fill in `T2E_DATA_DIR` and any Zenodo credentials you need:

```bash
cp .env.example .env
```

## Using the pre-computed dataset

If you do not need to regenerate the metrics yourself, the full pre-computed SOAR-EU dataset --- per-city GeoPackages bundled by country, plus the boundaries file and completeness coverage report --- is available on Zenodo:

- **DOI:** [10.5281/zenodo.18961227](https://doi.org/10.5281/zenodo.18961227)
- **Licence:** [Open Database License (ODbL 1.0)](https://opendatacommons.org/licenses/odbl/1-0/)

Download the country bundles you need and unzip the individual `metrics_{bounds_fid}.gpkg.zip` files to work with them in QGIS, GeoPandas, or any GeoPackage-aware tool. The sections below describe how to rebuild the dataset from raw open sources; this is only necessary if you want to change the pipeline parameters or extend the coverage.

## Data Loading

The pipeline requires several external datasets to be downloaded before processing. Each step below produces a GeoPackage that feeds into the next. All commands should be run from the repository root.

All scripts resolve data paths from `T2E_DATA_DIR` automatically (loaded from `.env`). Paths can also be passed as positional arguments to override the defaults.

### Boundaries

Boundaries are extracted from the [GHS Urban Centre Database (GHS-UCDB) R2024A](https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php) produced by the European Commission Joint Research Centre. Urban centres are defined using the [Degree of Urbanisation (DEGURBA)](https://human-settlement.emergency.copernicus.eu/degurba.php) methodology: contiguous 1 km^2 cells with at least 1,500 residents per km^2 and cumulative population of at least 50,000. The dataset is available under the [European Commission reuse policy](https://commission.europa.eu/legal-notice_en#copyright-notice) (Decision 2011/833/EU).

Download the GHS-UCDB GeoPackage from the above link, then run:

```bash
python -m src.data.generate_boundary_polys
```

### Urban Atlas

[Urban Atlas 2021](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2021) (~34 GB FlatGeobuf vectors, [DOI](https://doi.org/10.2909/05ae1ee1-e550-4e66-b74d-4926322d981a)). Download via the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) S3 endpoint (see [CDSE download instructions](#downloading-urban-atlas-and-street-tree-layer-from-cdse) below).

```bash
python -m src.data.load_urban_atlas_blocks
```

### Tree cover

[Street Tree Layer 2021](https://land.copernicus.eu/en/products/urban-atlas/street-tree-layer-2021) (~4 GB FlatGeobuf vectors). Download via CDSE S3 alongside Urban Atlas.

```bash
python -m src.data.load_urban_atlas_trees
```

### Building Heights

[Digital Height Model](https://land.copernicus.eu/local/urban-atlas/building-height-2012) (~1 GB raster).

```bash
python -m src.data.load_bldg_hts_raster
```

### Overture Maps data

Downloads and clips Overture layers (buildings, street edges/nodes, POI places, infrastructure) per city boundary. Each city is saved as a separate GeoPackage.

```bash
python -m src.data.load_overture --parallel_workers 14 --zip
```

> The Overture POI schema is based on [`overture_categories.csv`](https://github.com/OvertureMaps/schema/blob/dev/docs/schema/concepts/by-theme/places/overture_categories.csv).

### Census Data (2021)

[Eurostat Census Grid 2021 V2](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids) — population and demographic statistics aggregated to 1 km² cells. Download the **Version 2021 V2** ZIP and extract it so the GeoPackage lands at:

```text
$T2E_DATA_DIR/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg
```

No preprocessing is needed — the metrics step reads this file directly.

### Metrics

Compute all street-segment metrics:

```bash
python -m src.processing.generate_metrics --zip
```

### Downloading Urban Atlas and Street Tree Layer from CDSE

Both Copernicus datasets are distributed as FlatGeobuf files via the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) S3 endpoint.

1. Create an account at <https://dataspace.copernicus.eu/>.

1. **Generate S3 credentials at the S3 Keys Manager — not the OAuth clients page.**

   Go to **<https://eodata-s3keysmanager.dataspace.copernicus.eu/>** (this is a separate page from the main account dashboard). Click **Add Credentials**, set an expiration date, then **copy the secret key immediately** — it is shown only once and cannot be retrieved afterwards.

   > ⚠️ **Do not use the OAuth client registration page.** CDSE has two completely separate credential systems and only one of them works for S3:
   >
   > | Credential type                         | Used for                      | Where to create                                         | Access key format                         |
   > | --------------------------------------- | ----------------------------- | ------------------------------------------------------- | ----------------------------------------- |
   > | **S3 credentials** ✅ for this pipeline | `s3://eodata/` downloads      | <https://eodata-s3keysmanager.dataspace.copernicus.eu/> | ~20 alphanumeric chars (e.g. `AKIAJX...`) |
   > | OAuth client ❌ wrong for S3            | Token API, Catalogue/STAC API | account dashboard → "OAuth clients" / identity portal   | `sh-<uuid>` (e.g. `sh-8b66...`)           |
   >
   > Symptom of using the wrong one: `aws s3 ls` returns `An error occurred (InvalidAccessKeyId)`. The error is the same whatever the cause, so check your access key format first. If it starts with `sh-` or contains hyphens, you have an OAuth client ID — go back to the S3 Keys Manager URL above and generate an actual S3 credential.

1. Configure the AWS CLI:

```bash
aws configure
# AWS Access Key ID: <20-char alphanumeric key from S3 Keys Manager>
# AWS Secret Access Key: <secret from S3 Keys Manager>
# Default region name: (leave blank)
# Default output format: json

export AWS_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu/
```

1. Verify the credentials work before starting the multi-GB downloads below:

```bash
aws s3 ls s3://eodata/
# Should list top-level prefixes: CLMS/, Sentinel-1/, Sentinel-2/, ...
```

If this returns `InvalidAccessKeyId`, re-read step 2 — almost every occurrence of this error is from using an OAuth client ID instead of an S3 key.

> ℹ️ **Bucket name is lowercase `eodata`** (since 2 April 2026). The previous variants — `EODATA`, `EOCLOUD`, `eocloud`, `DIAS`, `dias` — were [deprecated and removed](https://dataspace.copernicus.eu/news/2026-3-20-changes-earth-observation-data-eodata-repository-bucket-names-planned-2-april-2026). Older tutorials and forum posts still reference `s3://EODATA/` and will fail with a "no such bucket" error.

The CDSE S3 endpoint does not return files inside subdirectories in a flat listing, so `aws s3 cp --recursive` alone downloads nothing. Iterate over city directories:

```bash
# Urban Atlas 2021 (~34 GB)
S3_BASE="s3://eodata/CLMS/land_cover_use_in_priority_areas/urban_atlas/clms_ua_land-cover-land-use_europe_V025ha_3yearly_v1/2021/01/01"
DEST="$T2E_DATA_DIR/UA_2021_3035_eu"
aws s3 ls "$S3_BASE/" | awk '{print $2}' | while read dir; do
    aws s3 cp "$S3_BASE/$dir" "$DEST/$dir" --recursive
done

# Street Tree Layer 2021 (~4 GB)
S3_BASE="s3://eodata/CLMS/land_cover_use_in_priority_areas/urban_atlas/clms_ua_street-tree-layer_europe_V005ha_3yearly_v1/2021/01/01"
DEST="$T2E_DATA_DIR/STL_2021_3035_eu"
aws s3 ls "$S3_BASE/" | awk '{print $2}' | while read dir; do
    aws s3 cp "$S3_BASE/$dir" "$DEST/$dir" --recursive
done
```

Reference: <https://documentation.dataspace.copernicus.eu/APIs/S3.html>

### Zenodo Upload

The processed dataset can be uploaded to Zenodo using `paper_data/zenodo_upload.py`. The script bundles per-city GeoPackages by country (to stay within Zenodo's 100-file limit), sets deposit metadata, and supports resumable uploads.

Ensure `ZENODO_TOKEN` and `ZENODO_RECORD_ID` are set in your `.env` file, then:

```bash
# Preview what will be uploaded
uv run python paper_data/zenodo_upload.py --dry-run --bundle

# Bundle by country and upload (resumable)
uv run python paper_data/zenodo_upload.py --bundle --resume

# Update metadata only
uv run python paper_data/zenodo_upload.py --metadata-only
```

Bundles are saved to `$T2E_DATA_DIR/zenodo_bundles/` by default (override with `--bundle-dir`).

## Data sources

| Source                                    | Content                                  | Licence                                 |
| ----------------------------------------- | ---------------------------------------- | --------------------------------------- |
| GHS-UCDB R2024A                           | Urban centre boundary polygons           | EC reuse policy (Decision 2011/833/EU)  |
| Overture Maps (Transportation, Buildings) | Street networks, building footprints     | ODbL                                    |
| Overture Maps (Places)                    | POI places                               | CDLA-Permissive-2.0                     |
| Overture Maps (Infrastructure)            | Transit stops, street furniture, parking | ODbL                                    |
| Copernicus Urban Atlas 2021               | Land-cover/land-use blocks               | EEA reuse policy (Directive 2003/98/EC) |
| Copernicus Street Tree Layer 2021         | Tree canopy polygons                     | EEA reuse policy (Directive 2003/98/EC) |
| Copernicus Digital Height Model 2012      | Building height raster (10 m, EPSG:3035) | EEA reuse policy (Directive 2003/98/EC) |
| Eurostat Census Grid 2021                 | Population/demographic cells (1 km^2)    | EC reuse policy (Decision 2011/833/EU)  |

## Licence

This repository depends on copy-left open source packages licensed as AGPLv3 and therefore adopts the same licence for the **code**. The **dataset** published on Zenodo is licensed under the [Open Database License (ODbL 1.0)](https://opendatacommons.org/licenses/odbl/1-0/) to comply with share-alike requirements of the Overture Maps layers.

## Papers

- [Data paper](paper_data/README.md) — SOAR-EU dataset description and POI validation (Data in Brief)
- [Atlas paper](paper_research/README.md) — Morphological typology of European cities (CEUS)

## Citation

If you use this dataset or code, please cite:

> Simons, G., Karimi, K., Zhand, S. (2026). SOAR-EU: Scalable Open Automatable Reproducible pedestrian-scale urban metrics for 626 European urban centres. Available at: <https://github.com/UCL/t2e-soar-eu>
