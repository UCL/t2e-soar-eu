from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

WORKING_CRS = 3035
DISTANCES_LU = [200, 400, 800, 1200, 1600]


def _empty_places_gdf() -> gpd.GeoDataFrame:
    geometry = gpd.GeoSeries([], crs=f"EPSG:{WORKING_CRS}")  # type: ignore[arg-type]
    return gpd.GeoDataFrame(  # type: ignore[call-arg]
        {"merged_cats": pd.Series(dtype="string")},
        geometry=geometry,
        crs=f"EPSG:{WORKING_CRS}",
    )


def _match_sirene_ape(series: pd.Series, codes: list[str]) -> pd.Series:
    series = series.astype("string")
    mask = pd.Series(False, index=series.index)
    for code in codes:
        # Convention used elsewhere in repo: 2-digit codes act as prefixes (e.g., "47" = retail)
        if code.endswith("*") or len(code) <= 2:
            prefix = code.rstrip("*")
            mask |= series.str.startswith(prefix, na=False)
        else:
            mask |= series == code
    return mask


def map_sirene_to_registry_places(
    sirene_gpkg: Path,
    bounds_geom,
    sirene_ape_mapping: dict[str, list[str]],
    categories: Iterable[str],
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(sirene_gpkg, layer="establishments").to_crs(WORKING_CRS)
    gdf = gdf[gdf.intersects(bounds_geom.buffer(2000))].copy()

    cat_list = list(categories)
    out_frames: list[gpd.GeoDataFrame] = []
    for cat in cat_list:
        codes = sirene_ape_mapping.get(cat)
        if not codes:
            continue
        mask = _match_sirene_ape(gdf.get("APE_code"), codes)
        sub = gdf.loc[mask, ["geometry"]].copy()
        sub["merged_cats"] = cat
        out_frames.append(sub)

    if not out_frames:
        return _empty_places_gdf()

    places = gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs=f"EPSG:{WORKING_CRS}")  # type: ignore[call-arg]
    places = places.set_geometry("geometry", inplace=False)
    return places


def map_bag_to_registry_places(
    bag_gpkg: Path,
    bounds_geom,
    bag_usage_mapping: dict[str, list[str]],
    categories: Iterable[str],
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(bag_gpkg, layer="buildings").to_crs(WORKING_CRS)
    gdf = gdf[gdf.intersects(bounds_geom.buffer(2000))].copy()

    # Ensure points.
    if not gdf.empty and gdf.geometry.geom_type.iloc[0] != "Point":
        gdf["geometry"] = gdf.geometry.representative_point()
        gdf = gdf.set_geometry("geometry")

    usage = gdf.get("gebruiksdoel")
    if usage is None:
        return _empty_places_gdf()

    usage = usage.astype("string")

    cat_list = list(categories)
    out_frames: list[gpd.GeoDataFrame] = []
    for cat in cat_list:
        purposes = bag_usage_mapping.get(cat)
        if not purposes:
            continue
        mask = pd.Series(False, index=usage.index)
        for purpose in purposes:
            mask |= usage.str.lower().str.contains(str(purpose).lower(), na=False)
        sub = gdf.loc[mask, ["geometry"]].copy()
        sub["merged_cats"] = cat
        out_frames.append(sub)

    if not out_frames:
        return _empty_places_gdf()

    places = gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs=f"EPSG:{WORKING_CRS}")  # type: ignore[call-arg]
    places = places.set_geometry("geometry", inplace=False)
    return places


def extract_accessibility_columns(df: pd.DataFrame, categories: Iterable[str]) -> list[str]:
    cols: list[str] = []
    for cat in categories:
        for d in DISTANCES_LU:
            for suffix in (f"cc_{cat}_{d}_nw", f"cc_{cat}_{d}_wt", f"cc_{cat}_nearest_max_{d}"):
                if suffix in df.columns:
                    cols.append(suffix)
    # Always keep ids, live flag, and coordinates (for spatial block bootstrap).
    base = [c for c in ["bounds_fid", "node_id", "live", "x", "y"] if c in df.columns]
    return base + cols


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rho via pandas rank + numpy corrcoef.

    Lightweight alternative to safe_spearman that avoids the scipy import.
    Use safe_spearman when you need scipy's ConstantInputWarning handling;
    use this version in tight loops or when scipy is not available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int, largest: bool = True) -> float:
    """Fraction of top-k (or bottom-k) items that agree between two arrays."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < max(10, k):
        return float("nan")
    a2 = a[ok]
    b2 = b[ok]
    a_order = np.argsort(a2)
    b_order = np.argsort(b2)
    if largest:
        a_idx = set(a_order[-k:])
        b_idx = set(b_order[-k:])
    else:
        a_idx = set(a_order[:k])
        b_idx = set(b_order[:k])
    return len(a_idx & b_idx) / k


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rho with NaN-safety; returns NaN if undefined."""
    import warnings

    from scipy.stats import ConstantInputWarning, spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho, _p = spearmanr(a[ok], b[ok])
    return float(rho)


def safe_kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Kendall tau-b with NaN-safety; returns NaN if undefined."""
    import warnings

    from scipy.stats import ConstantInputWarning, kendalltau

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        tau, _p = kendalltau(a[ok], b[ok])
    return float(tau)


def assign_spatial_blocks(x: np.ndarray, y: np.ndarray, block_size_m: float = 1600.0) -> np.ndarray:
    """Assign nodes to square grid cells for spatial block bootstrap.

    Parameters
    ----------
    x, y : array-like
        Coordinates in a projected CRS (metres), e.g. EPSG:3035.
    block_size_m : float
        Side length of each grid cell in metres.

    Returns
    -------
    block_ids : np.ndarray of int
        Integer label for each node's grid cell.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = (x // block_size_m).astype(np.int64)
    iy = (y // block_size_m).astype(np.int64)
    keys = np.column_stack([ix, iy])
    _, block_ids = np.unique(keys, axis=0, return_inverse=True)
    return block_ids


def spatial_block_bootstrap_ci(
    stat_fn,
    *,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap CI by resampling spatial blocks with replacement.

    Parameters
    ----------
    stat_fn : callable(np.ndarray) -> float
        Receives an index array into the original data and returns a scalar
        statistic (e.g. Spearman rho computed on the resampled subset).
    block_ids : np.ndarray of int
        Block label for each observation (from ``assign_spatial_blocks``).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (default 0.05 → 95 % CI).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
    """
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    if n_blocks < 5:
        return (float("nan"), float("nan"))

    # Pre-compute per-block index arrays for efficiency.
    block_indices = {int(b): np.where(block_ids == b)[0] for b in unique_blocks}

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_indices[int(b)] for b in sampled])
        stats[i] = stat_fn(idx)

    stats = stats[np.isfinite(stats)]
    if stats.size < 10:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))
