# Paper Data Folder Order

Use this folder as:

- `paper_data/manuscript.tex` and `paper_data/outputs/` for publication artifacts.
- `paper_data/code/` for reproducible generation scripts.
- `paper_data/code/README.md` for the canonical script execution order and Point A step-by-step flow.

## Scientific Data submission package

- `manuscript.tex` — main Data Descriptor (compiles with the SI appended, for a combined working PDF).
- `supplementary_sections.tex` — SI body (S1–S5), shared between the manuscript and the standalone SI.
- `supplementary.tex` — standalone Supplementary Information PDF for submission (`latexmk -pdf supplementary.tex`).
- `outputs/tables/supplementary_table_1_streets_schema.csv` — the S1 schema as a machine-readable Supplementary Table (Scientific Data asks for tables over one A4 page as csv/xlsx); regenerate with `code/s11_export_s1_schema_csv.py`.
- `cover_letter.md` — cover letter draft (suggested reviewers to be filled in).
- Before submission: publish the Zenodo deposit (the reserved DOI currently returns 404 because the record is still a draft).

Recommended separation of concerns:

- Keep heavy/intermediate outputs in `$T2E_DATA_DIR/paper_data_outputs`.
- Keep only manuscript-ready outputs in `paper_data/outputs/`.
- Treat any one-off scripts as temporary and move stable scripts into numbered pipeline files.

---

# Raw Input Data Sizes

The following Copernicus and GHSL datasets are used as inputs for the SOAR pipeline. After initial loading, the raw downloads are no longer needed — only the processed outputs in `datasets/` and `cities_data/` are used downstream.

| Dataset | Directory | Size |
|---|---|---|
| Urban Atlas 2021 (blocks) | `UA_2021_3035_eu/` | 34 GB |
| Street Tree Layer 2021 | `STL_2021_3035_eu/` | 4 GB |
| Building Height 2012 | `Building_Height_2012_3035_eu/` | 1.0 GB |
| GHS-UCDB 2024 (boundaries) | `GHS_UCDB_REGION_EUROPE_R2024A_V1_1/` | 59 MB |
| **Total** | | **~39 GB** |

---

# Reference Datasets

The POI characterisation pipeline compares Overture POI spatial patterns against official national business registries. These datasets are used for the data paper but are **not required** to run the main SOAR pipeline.

## France: SIRENE Business Registry

**Official Name:** Base Sirène des entreprises et de leurs établissements (SIREN, SIRET)

**Source:** Institut national de la statistique et des études économiques (INSEE)

**License:** [Open License 2.0 (Licence Ouverte / Etalab v2.0)](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)

**Description:** National registry of all French businesses and establishments with economic activity codes (APE - Activité Principale Exercée), geographic coordinates, and administrative status.

**Coverage:** ~31 million establishments (as of January 2026)

**Download:**

- **URL:** <https://www.data.gouv.fr/en/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/>
- **Direct link (parquet, ~2.2 GB):** <https://object.files.data.gouv.fr/data-pipeline-open/siren/stock/StockEtablissement_utf8.parquet>
- **Direct link (CSV in ZIP, ~2.8 GB):** <https://object.files.data.gouv.fr/data-pipeline-open/siren/stock/StockEtablissement_utf8.zip>
- **Format:** Parquet (preferred — used directly by `s00_prepare_reference_data.py`) or CSV in ZIP
- **Update frequency:** Monthly (file is the current snapshot — one row per SIRET, active + closed in current state). The `StockEtablissementHistorique_*` files at the same prefix are the multi-row change history and are **not** what this pipeline expects.

**Citation:**

```
INSEE (2026). Base Sirène des entreprises et de leurs établissements (SIREN, SIRET).
Institut National de la Statistique et des Études Économiques.
https://www.data.gouv.fr/en/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/
Retrieved: January 2026
```

**Required columns:**

- `siret` - Unique establishment identifier (14 digits)
- `activitePrincipaleEtablissement` - Economic activity code (APE/NAF)
- `etatAdministratifEtablissement` - Administrative status (A=active, F=closed)
- `coordonneeLambertAbscisseEtablissement` - X coordinate (Lambert 93 projection)
- `coordonneeLambertOrdonneeEtablissement` - Y coordinate (Lambert 93 projection)

**Classification system:** APE codes follow the French NAF classification (Nomenclature d'Activités Française), derived from EU NACE Rev. 2 standard. See: <https://www.insee.fr/en/metadonnees/nafr2/>

**Characterisation usage:** Harmonised to 5 POI categories via APE code mapping (see `paper_data/code/config.py` and `paper_data/code/`).

---

## Netherlands: BAG Building Registry

**Official Name:** Basisregistratie Adressen en Gebouwen (BAG) - Basic Registration of Addresses and Buildings

**Source:** Kadaster (Dutch Cadastre, Land Registry and Mapping Agency)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/)

**Description:** National registry of all buildings and addresses in the Netherlands with usage designations (gebruiksdoel), geometric footprints, and construction status.

**Coverage:** ~10 million buildings with ~18 million address objects (as of January 2026)

**Download:**

- **URL:** <https://www.kadaster.nl/zakelijk/producten/adressen-en-gebouwen/bag-2.0-extract>
- **Direct link:** <https://service.pdok.nl/lv/bag/atom/downloads/lvbag-extract-nl.zip>
- **Format:** XML/GML files (compressed as ZIP, ~5 GB total for full national extract)
- **Update frequency:** Daily

**Citation:**

```
Kadaster (2026). Basisregistratie Adressen en Gebouwen (BAG) 2.0 Extract.
Kadaster, Dutch Land Registry and Mapping Agency.
https://www.kadaster.nl/zakelijk/producten/adressen-en-gebouwen/bag-2.0-extract
Retrieved: January 2026
```

**Required file:** `9999VBO08012026.zip` from the BAG extract

- VBO = Verblijfsobject (dwelling object / address object with usage purpose)

**Required fields:**

- `identificatie` - Unique object identifier (16 digits)
- `gebruiksdoel` - Usage purpose designation (functional category)
- `geometry` - Building footprint or address point (RD New projection, EPSG:28992)
- `status` - Object status (use only active records)

**Classification system:** Gebruiksdoel (usage purposes) include:

- `woonfunctie` - Residential function
- `winkelfunctie` - Shop/retail function
- `logiesfunctie` - Lodging/accommodation function
- `bijeenkomstfunctie` - Meeting/assembly function
- `gezondheidszorgfunctie` - Healthcare function
- `onderwijsfunctie` - Education function
- And others (see BAG documentation)

**Documentation:** <https://zakelijk.kadaster.nl/bag-2.0-extract>

**Characterisation usage:** Harmonised to 5 POI categories via usage purpose mapping (see `paper_data/code/config.py` and `paper_data/code/`).

---

## Reference Data Preparation

**Location:** Place reference datasets in `$T2E_DATA_DIR/validation/`

**Preparation script:** `paper_data/code/s00_prepare_reference_data.py`

This script:

1. Loads SIRENE Parquet/CSV and filters to active establishments with coordinates
2. Extracts BAG VBO data from ZIP and filters to relevant usage purposes
3. Converts coordinates to EPSG:3035 (ETRS89 / LAEA Europe)
4. Spatially filters to city boundaries
5. Saves processed GeoPackage files: `sirene_france.gpkg`, `bag_netherlands.gpkg`

**Raw data:**

- `StockEtablissement_utf8.parquet` — SIRENE (the script reads parquet only; if you downloaded the ZIP, extract the CSV and convert it with `pd.read_csv(...).to_parquet(...)`)
- `lvbag-extract-nl/` (directory with ZIP files) — BAG

---

## Data Citation Guidelines

**For methods section:**

> The SOAR dataset was validated against official national registries: the French SIRENE business registry (INSEE, 2026) covering ~31 million establishments and the Netherlands BAG building registry (Kadaster, 2026) covering ~10 million buildings.

**For acknowledgments:**

> This research used data from INSEE (Institut National de la Statistique et des Études Économiques) and Kadaster (Dutch Land Registry and Mapping Agency).

**For data availability statement:**

> SIRENE data are publicly available from <https://www.data.gouv.fr/en/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/> under Open License 2.0. BAG data are publicly available from <https://www.kadaster.nl/zakelijk/producten/adressen-en-gebouwen/bag-2.0-extract> under CC0 1.0 Universal license.
