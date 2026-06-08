#!/usr/bin/env python3
"""Point A: City-level summaries for between-city comparison.

This script implements Level 1 of the workflow doc: it reduces node-level
accessibility metrics into per-city summaries for Overture vs registry.

Inputs: the per-city node parquet outputs from `s01_run_reference_replication.py`.
Outputs:
- pointa_accessibility_city_summaries.csv

Notes
-----
- Counts are summarised as median(log1p(value)) across live nodes.
- Nearest is summarised as median(nearest_distance_m) across live nodes.
- For decision-style nearest checks, we also summarise the share of nodes with
    nearest <= tolerance (default tolerances: 50/100/200/400/800/1200/1600m).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from config import OUTPUT_DIR
from pointa_constants import DEFAULT_CATEGORIES, NEAREST_TOLERANCES_M
from pointa_utils import DISTANCES_LU


def _nanmedian(x: np.ndarray) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float(np.median(x[ok]))


def _median_log1p(x: np.ndarray) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float(np.median(np.log1p(x[ok])))


def _q25_log1p(x: np.ndarray) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float(np.percentile(np.log1p(x[ok]), 25))


def _q75_log1p(x: np.ndarray) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float(np.percentile(np.log1p(x[ok]), 75))


def _p90_log1p(x: np.ndarray) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float(np.percentile(np.log1p(x[ok]), 90))


def _p90_p50_spread_log1p(x: np.ndarray) -> float:
    """Tail spread of log1p(x): p90 - p50.

    Intuition: captures how heavy the tail is relative to the typical value,
    while being less sensitive to extreme outliers than max-like summaries.
    """
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    v = np.log1p(x[ok])
    p90 = float(np.percentile(v, 90))
    p50 = float(np.percentile(v, 50))
    return p90 - p50


def _top10_median_log1p(x: np.ndarray) -> float:
    """Median of the top 10% of log1p(x).

    This is a more stable tail-intensity summary than a single percentile.
    """
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    v = np.log1p(x[ok])
    n = int(v.size)
    k = max(1, int(np.ceil(0.10 * n)))
    # Take the largest k values, then median.
    topk = np.partition(v, n - k)[n - k :]
    return float(np.median(topk))


def _share_leq(x: np.ndarray, threshold: float) -> float:
    x = x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan")
    return float((x[ok] <= float(threshold)).mean())


def main() -> int:
    in_dir = OUTPUT_DIR / "replication"
    categories = list(DEFAULT_CATEGORIES)
    nearest_max_m = 1600
    nearest_tolerances_m = list(NEAREST_TOLERANCES_M)
    out_city_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_city_summaries.csv"
    out_city_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_city_csv.exists():
        print(f"✓ Output already exists, skipping: {out_city_csv.name}")
        return 0

    city_rows: list[dict] = []

    overture_files = sorted(in_dir.glob("pointa_nodes_overture_*.parquet"))
    if not overture_files:
        raise FileNotFoundError(f"No overture parquet files found in {in_dir}")

    for ovt_path in overture_files:
        bounds_fid = ovt_path.stem.split("_")[-1]
        reg_path = in_dir / f"pointa_nodes_registry_{bounds_fid}.parquet"
        if not reg_path.exists():
            continue

        try:
            ovt = pd.read_parquet(ovt_path)
            reg = pd.read_parquet(reg_path)
        except Exception as exc:
            print(f"⚠ Skipping bounds_fid={bounds_fid}: failed to read parquet pair: {exc}")
            continue

        key_cols = [c for c in ["bounds_fid", "node_id"] if c in ovt.columns and c in reg.columns]
        merged = ovt.merge(reg, on=key_cols, suffixes=("_ovt", "_reg"), how="inner")
        if "live_ovt" in merged.columns:
            merged = merged[merged["live_ovt"]].copy()

        n_nodes = int(len(merged))

        for cat in categories:
            for d in DISTANCES_LU:
                col_nw_ovt = f"cc_{cat}_{d}_nw_ovt"
                col_nw_reg = f"cc_{cat}_{d}_nw_reg"
                col_wt_ovt = f"cc_{cat}_{d}_wt_ovt"
                col_wt_reg = f"cc_{cat}_{d}_wt_reg"

                if col_nw_ovt in merged.columns and col_nw_reg in merged.columns:
                    ovt_nw = merged[col_nw_ovt].to_numpy()
                    reg_nw = merged[col_nw_reg].to_numpy()
                    for metric_name, fn in [
                        ("count_nw_q25_log1p", _q25_log1p),
                        ("count_nw_median_log1p", _median_log1p),
                        ("count_nw_q75_log1p", _q75_log1p),
                        ("count_nw_p90_log1p", _p90_log1p),
                        ("count_nw_p90_p50_spread_log1p", _p90_p50_spread_log1p),
                        ("count_nw_top10_median_log1p", _top10_median_log1p),
                    ]:
                        city_rows.append(
                            {
                                "bounds_fid": bounds_fid,
                                "category": cat,
                                "distance_m": d,
                                "metric": metric_name,
                                "source": "ovt",
                                "value": fn(ovt_nw),
                                "n_nodes": n_nodes,
                            }
                        )
                        city_rows.append(
                            {
                                "bounds_fid": bounds_fid,
                                "category": cat,
                                "distance_m": d,
                                "metric": metric_name,
                                "source": "reg",
                                "value": fn(reg_nw),
                                "n_nodes": n_nodes,
                            }
                        )

                if col_wt_ovt in merged.columns and col_wt_reg in merged.columns:
                    ovt_wt = merged[col_wt_ovt].to_numpy()
                    reg_wt = merged[col_wt_reg].to_numpy()
                    for metric_name, fn in [
                        ("count_wt_q25_log1p", _q25_log1p),
                        ("count_wt_median_log1p", _median_log1p),
                        ("count_wt_q75_log1p", _q75_log1p),
                        ("count_wt_p90_log1p", _p90_log1p),
                        ("count_wt_p90_p50_spread_log1p", _p90_p50_spread_log1p),
                        ("count_wt_top10_median_log1p", _top10_median_log1p),
                    ]:
                        city_rows.append(
                            {
                                "bounds_fid": bounds_fid,
                                "category": cat,
                                "distance_m": d,
                                "metric": metric_name,
                                "source": "ovt",
                                "value": fn(ovt_wt),
                                "n_nodes": n_nodes,
                            }
                        )
                        city_rows.append(
                            {
                                "bounds_fid": bounds_fid,
                                "category": cat,
                                "distance_m": d,
                                "metric": metric_name,
                                "source": "reg",
                                "value": fn(reg_wt),
                                "n_nodes": n_nodes,
                            }
                        )

            # Nearest (only produced at nearest_max_m in current pipeline).
            near_col_ovt = f"cc_{cat}_nearest_max_{int(nearest_max_m)}_ovt"
            near_col_reg = f"cc_{cat}_nearest_max_{int(nearest_max_m)}_reg"
            if near_col_ovt in merged.columns and near_col_reg in merged.columns:
                a = merged[near_col_ovt].to_numpy(dtype=float)
                b = merged[near_col_reg].to_numpy(dtype=float)

                city_rows.append(
                    {
                        "bounds_fid": bounds_fid,
                        "category": cat,
                        "distance_m": int(nearest_max_m),
                        "metric": "nearest_median_m",
                        "source": "ovt",
                        "value": _nanmedian(a),
                        "n_nodes": int(np.isfinite(a).sum()),
                    }
                )
                city_rows.append(
                    {
                        "bounds_fid": bounds_fid,
                        "category": cat,
                        "distance_m": int(nearest_max_m),
                        "metric": "nearest_median_m",
                        "source": "reg",
                        "value": _nanmedian(b),
                        "n_nodes": int(np.isfinite(b).sum()),
                    }
                )

                for tol in nearest_tolerances_m:
                    city_rows.append(
                        {
                            "bounds_fid": bounds_fid,
                            "category": cat,
                            "distance_m": int(nearest_max_m),
                            "metric": "nearest_within_share",
                            "tolerance_m": float(tol),
                            "source": "ovt",
                            "value": _share_leq(a, tol),
                            "n_nodes": int(np.isfinite(a).sum()),
                        }
                    )
                    city_rows.append(
                        {
                            "bounds_fid": bounds_fid,
                            "category": cat,
                            "distance_m": int(nearest_max_m),
                            "metric": "nearest_within_share",
                            "tolerance_m": float(tol),
                            "source": "reg",
                            "value": _share_leq(b, tol),
                            "n_nodes": int(np.isfinite(b).sum()),
                        }
                    )

    city_df = pd.DataFrame(city_rows)
    if city_df.empty:
        raise RuntimeError("No city summaries produced (no matching parquet pairs?).")

    city_df.to_csv(out_city_csv, index=False)
    print(f"✓ Wrote: {out_city_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
