#!/usr/bin/env python3
"""Upload SOAR-EU dataset to Zenodo via the InvenioRDM REST API.

Reads ZENODO_TOKEN from .env (root of repository).
Reads ZENODO_RECORD_ID from .env (or pass --record-id).

Usage:
    # Dry run — shows what would be uploaded
    uv run python scripts/zenodo_upload.py --dry-run

    # Bundle city files by country into zips (Zenodo 100-file limit)
    uv run python scripts/zenodo_upload.py --bundle

    # Upload metadata only (no files)
    uv run python scripts/zenodo_upload.py --metadata-only

    # Resume an interrupted upload (skips already-completed files)
    uv run python scripts/zenodo_upload.py --bundle --resume

    # Remove all existing files and re-upload
    uv run python scripts/zenodo_upload.py --bundle --replace
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

if "T2E_DATA_DIR" not in os.environ:
    raise OSError("T2E_DATA_DIR environment variable is not set. See .env.example.")
_DATA_DIR = Path(os.environ["T2E_DATA_DIR"])
BOUNDARIES_PATH = _DATA_DIR / "datasets" / "boundaries.gpkg"
PROCESSED_DIR = _DATA_DIR / "cities_data" / "processed"
COVERAGE_PATH = ROOT / "paper_data" / "outputs" / "completeness_coverage.csv"
_CSV_DIR = _DATA_DIR / "paper_data_outputs" / "csv"
SOURCE_COUNTS_PATH = _CSV_DIR / "building_source_counts.csv"
SOURCE_COUNTRY_PATH = _CSV_DIR / "building_source_by_country.csv"

ZENODO_API = "https://zenodo.org/api"

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "access": {
        "record": "public",
        "files": "public",
    },
    "files": {
        "enabled": True,
    },
    "metadata": {
        "title": ("SOAR-EU: Scalable, Open, Automatable, and Reproducible European Urban Dataset"),
        "description": (
            "<p>SOAR-EU provides pre-computed pedestrian-scale urban metrics "
            "for 626 European cities. It delivers over 100 pre-computed "
            "metrics per street segment at multiple spatial scales "
            "(200–9,600 m), covering network centrality, land-use "
            "accessibility and diversity, building morphology, green space "
            "proximity, and demographics.</p>"
            "<p>Each city is provided as a GeoPackage file (EPSG:3035) with "
            "three layers: streets (segment-level metrics), buildings "
            "(footprint morphometrics), and blocks (land-use block metrics "
            "with Spacematrix indicators). City files are individually "
            "ZIP-compressed and bundled by country.</p>"
            "<p>The deposit also includes a boundaries file mapping city "
            "identifiers to names and geometries, and a completeness "
            "coverage report documenting per-column non-null coverage for "
            "every attribute in every city.</p>"
            "<h4>Data sources</h4>"
            "<ul>"
            "<li><strong>GHS Urban Centre Database (GHS-UCDB) R2024A</strong> — "
            "European Commission Joint Research Centre. Urban centre boundary "
            "polygons. EC reuse policy (Decision 2011/833/EU).</li>"
            "<li><strong>Overture Maps Foundation — Transportation theme</strong> — "
            "Street networks (connectors and segments). Contains information "
            "from Overture Maps Foundation (overturemaps.org) and "
            "© OpenStreetMap contributors, made available under the "
            "Open Database License (ODbL).</li>"
            "<li><strong>Overture Maps Foundation — Buildings theme</strong> — "
            "Building footprints. Contains information from Overture Maps "
            "Foundation (overturemaps.org) and © OpenStreetMap contributors, "
            "made available under the Open Database License (ODbL).</li>"
            "<li><strong>Overture Maps Foundation — Places theme</strong> — "
            "Points of interest for land-use accessibility metrics. "
            "CDLA-Permissive-2.0.</li>"
            "<li><strong>Overture Maps Foundation — Base theme (infrastructure)</strong> — "
            "Transit stops, street furniture, parking. "
            "ODbL.</li>"
            "<li><strong>Copernicus Urban Atlas 2021</strong> — "
            "European Environment Agency. Land-cover/land-use blocks and "
            "green space polygons. EEA reuse policy (Regulation (EU) No 1159/2013).</li>"
            "<li><strong>Copernicus Street Tree Layer 2021</strong> — "
            "European Environment Agency. Tree canopy polygons. "
            "EEA reuse policy (Regulation (EU) No 1159/2013).</li>"
            "<li><strong>Copernicus Building Height 2012</strong> — "
            "European Environment Agency. Building height raster (10 m). "
            "EEA reuse policy (Regulation (EU) No 1159/2013).</li>"
            "<li><strong>Eurostat Census Grid 2021</strong> — "
            "Population, employment, age, nationality, and migration at "
            "1 km² resolution. EC reuse policy (Decision 2011/833/EU).</li>"
            "</ul>"
            "<h4>Processing</h4>"
            "<p>Processing pipeline: "
            "<a href='https://github.com/UCL/t2e-soar-eu'>github.com/UCL/t2e-soar-eu</a> "
            "(AGPL-3.0, v1.1.0). CRS: EPSG:3035 (ETRS89-LAEA Europe).</p>"
            "<p>Funded by the European Union's Horizon Europe Research and "
            "Innovation Programme under Grant Agreement No. 101078890.</p>"
        ),
        "version": "1.2.0",
        "publication_date": "2026-05-29",
        "resource_type": {"id": "dataset"},
        "creators": [
            {
                "person_or_org": {
                    "type": "personal",
                    "family_name": "Simons",
                    "given_name": "Gareth",
                    "identifiers": [
                        {"scheme": "orcid", "identifier": "0000-0003-3790-0638"},
                    ],
                },
                "affiliations": [
                    {"name": "Space Syntax Laboratory, UCL Bartlett School of Architecture"},
                ],
            },
            {
                "person_or_org": {
                    "type": "personal",
                    "family_name": "Karimi",
                    "given_name": "Kayvan",
                    "identifiers": [
                        {"scheme": "orcid", "identifier": "0000-0002-1461-2599"},
                    ],
                },
                "affiliations": [
                    {"name": "Space Syntax Laboratory, UCL Bartlett School of Architecture"},
                ],
            },
            {
                "person_or_org": {
                    "type": "personal",
                    "family_name": "Zhand",
                    "given_name": "Sepehr",
                    "identifiers": [
                        {"scheme": "orcid", "identifier": "0009-0003-8520-3551"},
                    ],
                },
                "affiliations": [
                    {"name": "Space Syntax Laboratory, UCL Bartlett School of Architecture"},
                ],
            },
        ],
        "publisher": "Zenodo",
        "rights": [
            {
                "title": {"en": "Open Data Commons Open Database License (ODbL) v1.0"},
                "description": {
                    "en": (
                        "Contains information from Overture Maps Foundation "
                        "(overturemaps.org), made available under the ODbL."
                    )
                },
                "link": "https://opendatacommons.org/licenses/odbl/1-0/",
            },
        ],
        "funding": [
            {
                "funder": {"name": "European Commission"},
                "award": {
                    "title": {"en": "TWIN2EXPAND"},
                    "number": "101078890",
                },
            },
        ],
    },
    "custom_fields": {
        "code:codeRepository": "https://github.com/UCL/t2e-soar-eu",
        "code:programmingLanguage": {"id": "python"},
        "code:developmentStatus": {"id": "active"},
        "code:runtimePlatform": "Python 3.12",
        "code:version": "v1.2.0",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_auth() -> tuple[str, str]:
    """Return (token, record_id) from env."""
    token = os.environ.get("ZENODO_TOKEN", "")
    record_id = os.environ.get("ZENODO_RECORD_ID", "")
    if not token:
        sys.exit("Error: ZENODO_TOKEN not set in .env")
    if not record_id:
        sys.exit("Error: ZENODO_RECORD_ID not set in .env")
    return token, record_id


def headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def list_remote_files(token: str, record_id: str) -> dict[str, dict]:
    """Return {filename: entry_dict} for files already in the draft."""
    r = requests.get(
        f"{ZENODO_API}/records/{record_id}/draft/files",
        headers=headers(token),
    )
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    data = r.json()
    entries = data.get("entries", data.get("contents", []))
    return {e["key"]: e for e in entries}


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    backoff: float = 5.0,
    **kwargs,
) -> requests.Response:
    """Make an HTTP request with retry on 5xx / connection errors."""
    for attempt in range(max_retries):
        try:
            r = requests.request(method, url, **kwargs)
            if r.status_code < 500:
                return r
            label = f"{r.status_code}"
        except requests.ConnectionError as e:
            label = str(e)
            r = None  # type: ignore[assignment]
        wait = backoff * (2**attempt)
        print(f" [{label}, retry {attempt + 1}/{max_retries} in {wait:.0f}s]", end="", flush=True)
        time.sleep(wait)
    # Last attempt — let it raise normally
    if r is not None:
        r.raise_for_status()
    raise requests.ConnectionError(f"Failed after {max_retries} retries: {url}")


def delete_all_remote_files(token: str, record_id: str) -> int:
    """Delete all files from the Zenodo draft. Returns count of deleted files."""
    existing = list_remote_files(token, record_id)
    if not existing:
        print("  No remote files to delete.")
        return 0
    hdrs = headers(token)
    deleted = 0
    for name in sorted(existing):
        print(f"  Deleting: {name}...", end="", flush=True)
        r = _request_with_retry(
            "DELETE",
            f"{ZENODO_API}/records/{record_id}/draft/files/{name}",
            headers=hdrs,
        )
        r.raise_for_status()
        print(" done")
        deleted += 1
    print(f"  Deleted {deleted} files from draft.")
    return deleted


def upload_file(
    token: str,
    record_id: str,
    filepath: Path,
    remote_name: str,
    *,
    skip_existing: bool = False,
    existing_files: dict[str, dict] | None = None,
) -> bool:
    """Upload a single file using the 3-step InvenioRDM API.

    Returns True if uploaded, False if skipped.
    """
    hdrs = headers(token)

    # Check if already uploaded
    if skip_existing and existing_files and remote_name in existing_files:
        entry = existing_files[remote_name]
        if entry.get("status") == "completed":
            remote_size = entry.get("size", 0)
            local_size = filepath.stat().st_size
            if remote_size == local_size:
                print(f"  Skip (already uploaded): {remote_name}")
                return False

    file_size = filepath.stat().st_size
    size_mb = file_size / (1024 * 1024)

    # Step 1: Initialize
    r = _request_with_retry(
        "POST",
        f"{ZENODO_API}/records/{record_id}/draft/files",
        json=[{"key": remote_name}],
        headers={**hdrs, "Content-Type": "application/json"},
    )
    if r.status_code == 400 and "already exists" in r.text.lower():
        # Delete and re-init
        _request_with_retry(
            "DELETE",
            f"{ZENODO_API}/records/{record_id}/draft/files/{remote_name}",
            headers=hdrs,
        )
        r = _request_with_retry(
            "POST",
            f"{ZENODO_API}/records/{record_id}/draft/files",
            json=[{"key": remote_name}],
            headers={**hdrs, "Content-Type": "application/json"},
        )
    r.raise_for_status()

    # Step 2: Upload content (streamed)
    print(f"  Uploading: {remote_name} ({size_mb:.1f} MB)...", end="", flush=True)
    t0 = time.time()
    with open(filepath, "rb") as f:
        r = _request_with_retry(
            "PUT",
            f"{ZENODO_API}/records/{record_id}/draft/files/{remote_name}/content",
            data=f,
            headers={**hdrs, "Content-Type": "application/octet-stream"},
        )
    r.raise_for_status()
    elapsed = time.time() - t0
    speed = size_mb / elapsed if elapsed > 0 else 0
    print(f" done ({elapsed:.0f}s, {speed:.1f} MB/s)")

    # Step 3: Commit
    r = _request_with_retry(
        "POST",
        f"{ZENODO_API}/records/{record_id}/draft/files/{remote_name}/commit",
        headers=hdrs,
    )
    r.raise_for_status()
    return True


def update_metadata(token: str, record_id: str) -> None:
    """Update draft metadata."""
    r = _request_with_retry(
        "PUT",
        f"{ZENODO_API}/records/{record_id}/draft",
        json=METADATA,
        headers={**headers(token), "Content-Type": "application/json"},
    )
    if not r.ok:
        print(f"Metadata update failed ({r.status_code}):")
        print(r.text[:2000])
        r.raise_for_status()
    print(f"Metadata updated for record {record_id}")


# ---------------------------------------------------------------------------
# Bundling
# ---------------------------------------------------------------------------


def load_country_mapping() -> dict[str, str]:
    """Return {bounds_fid: country} from boundaries.gpkg."""
    import pyogrio

    df = pyogrio.read_dataframe(
        BOUNDARIES_PATH,
        columns=["bounds_fid", "country"],
        read_geometry=False,
    )
    return dict(zip(df["bounds_fid"].astype(str), df["country"], strict=False))


def bundle_by_country(output_dir: Path) -> list[Path]:
    """Create one ZIP per country containing that country's city files.

    Returns list of created ZIP paths.
    """
    fid_to_country = load_country_mapping()

    # Group files by country
    country_files: dict[str, list[Path]] = {}
    for gpkg_zip in sorted(PROCESSED_DIR.glob("metrics_*.gpkg.zip")):
        fid = gpkg_zip.stem.replace("metrics_", "").replace(".gpkg", "")
        country = fid_to_country.get(fid, "Unknown")
        # Sanitise country name for filenames
        safe_country = country.replace(" ", "_").replace("/", "_")
        country_files.setdefault(safe_country, []).append(gpkg_zip)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = []

    for country, files in sorted(country_files.items()):
        bundle_path = output_dir / f"soar_eu_{country}.zip"
        if bundle_path.exists():
            print(f"  Bundle exists, skipping: {bundle_path.name}")
            bundles.append(bundle_path)
            continue

        print(
            f"  Bundling {country}: {len(files)} cities...",
            end="",
            flush=True,
        )
        # Use zip -j to store files flat (no directory structure)
        cmd = ["zip", "-j", "-0", str(bundle_path)]  # -0 = store only (no compression, already compressed)
        cmd.extend(str(f) for f in files)
        subprocess.run(cmd, check=True, capture_output=True)
        size_mb = bundle_path.stat().st_size / (1024 * 1024)
        print(f" {size_mb:.0f} MB")
        bundles.append(bundle_path)

    return bundles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload SOAR-EU dataset to Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading.",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Bundle city files by country (required for Zenodo 100-file limit).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=_DATA_DIR / "zenodo_bundles",
        help="Directory for country bundles (default: $T2E_DATA_DIR/zenodo_bundles).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Update metadata only, don't upload files.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete all existing files from the draft before uploading.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already uploaded (by name and size).",
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help="Override ZENODO_RECORD_ID from .env.",
    )
    args = parser.parse_args(argv)

    token, record_id = get_auth()
    if args.record_id:
        record_id = args.record_id

    # --- Metadata ---
    if not args.dry_run:
        print("Updating metadata...")
        try:
            update_metadata(token, record_id)
        except requests.HTTPError as e:
            print(f"Warning: metadata update failed: {e}")
            print(f"Response: {e.response.text if e.response else 'N/A'}")
            print("Continuing with file upload...")

    if args.metadata_only:
        return 0

    # --- Collect files to upload ---
    upload_list: list[tuple[Path, str]] = []  # (local_path, remote_name)

    # Always include boundaries, coverage, and source composition
    upload_list.append((BOUNDARIES_PATH, "boundaries.gpkg"))
    upload_list.append((COVERAGE_PATH, "completeness_coverage.csv"))
    upload_list.append((SOURCE_COUNTS_PATH, "building_source_counts.csv"))
    upload_list.append((SOURCE_COUNTRY_PATH, "building_source_by_country.csv"))

    if args.bundle:
        print(f"\nBundling city files by country into {args.bundle_dir}...")
        if args.dry_run:
            fid_to_country = load_country_mapping()
            countries = set(fid_to_country.values())
            print(f"  Would create {len(countries)} country bundles")
            for c in sorted(countries):
                safe = c.replace(" ", "_").replace("/", "_")
                n = sum(1 for v in fid_to_country.values() if v == c)
                print(f"    soar_eu_{safe}.zip ({n} cities)")
        else:
            bundles = bundle_by_country(args.bundle_dir)
            for b in bundles:
                upload_list.append((b, b.name))
    else:
        # Upload individual city files (only works if quota > 100 files)
        city_files = sorted(PROCESSED_DIR.glob("metrics_*.gpkg.zip"))
        print(
            f"\nWARNING: Uploading {len(city_files)} individual files. "
            f"Zenodo default limit is 100 files per record. "
            f"Use --bundle to bundle by country instead."
        )
        for f in city_files:
            upload_list.append((f, f.name))

    # --- Summary ---
    total_size = sum(p.stat().st_size for p, _ in upload_list if p.exists())
    print(f"\nUpload plan: {len(upload_list)} files, {total_size / (1024**3):.1f} GB")

    if args.dry_run:
        for path, name in upload_list:
            size = path.stat().st_size / (1024 * 1024) if path.exists() else 0
            exists = "EXISTS" if path.exists() else "MISSING"
            print(f"  {name:50s} {size:8.1f} MB  [{exists}]")
        print("\nDry run complete. Add --bundle to proceed.")
        return 0

    # --- Replace: delete existing files first ---
    if args.replace:
        print("\nDeleting all existing files from draft...")
        delete_all_remote_files(token, record_id)

    # --- Upload ---
    existing = {}
    if args.resume:
        print("Checking already-uploaded files...")
        existing = list_remote_files(token, record_id)
        print(f"  {len(existing)} files already in draft")

    uploaded = 0
    skipped = 0
    failed = 0

    for i, (path, name) in enumerate(upload_list, 1):
        print(f"[{i}/{len(upload_list)}]", end="")
        if not path.exists():
            print(f"  MISSING: {path}")
            failed += 1
            continue

        try:
            did_upload = upload_file(
                token,
                record_id,
                path,
                name,
                skip_existing=args.resume,
                existing_files=existing,
            )
            if did_upload:
                uploaded += 1
            else:
                skipped += 1
        except requests.HTTPError as e:
            print(f"  FAILED: {name}: {e}")
            if e.response is not None:
                print(f"    {e.response.text[:200]}")
            failed += 1
            # Brief pause before continuing
            time.sleep(2)

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
