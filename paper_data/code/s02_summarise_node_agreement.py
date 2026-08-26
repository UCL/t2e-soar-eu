#!/usr/bin/env python3
"""Point A: Compare node-level accessibility from Overture vs registry.

Reads the outputs from `s01_run_reference_replication.py` and produces a compact CSV summary
of within-city agreement metrics (node-level correlations + top-10% overlap).

This is deliberately lightweight: it is meant to validate that the workflow is
wired correctly before scaling to many reference cities.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from config import OUTPUT_DIR
from pointa_constants import (
    DEFAULT_CATEGORIES,
    NEAREST_TOLERANCES_M,
    SPATIAL_BLOCK_BOOT_ALPHA,
    SPATIAL_BLOCK_BOOT_N,
    SPATIAL_BLOCK_BOOT_SEED,
)
from pointa_utils import (
    DISTANCES_LU,
    assign_spatial_blocks,
    safe_kendall,
    safe_spearman,
    spatial_block_bootstrap_ci,
    spearman_rho,
    topk_overlap,
)


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)

    tp = float((yt & yp).sum())
    tn = float((~yt & ~yp).sum())
    fp = float((~yt & yp).sum())
    fn = float((yt & ~yp).sum())
    n = tp + tn + fp + fn
    if n == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    f1 = (
        (2 * precision * recall) / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> int:
    in_dir = OUTPUT_DIR / "replication"
    categories = list(DEFAULT_CATEGORIES)
    nearest_tolerances_m = list(NEAREST_TOLERANCES_M)
    out_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_node_agreement.csv"
    out_nearest_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_nearest_tolerance_agreement.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_nearest_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and out_nearest_csv.exists():
        print(f"✓ Both outputs already exist, skipping: {out_csv.name}, {out_nearest_csv.name}")
        return 0

    rows: list[dict] = []
    nearest_rows: list[dict] = []

    overture_files = sorted(in_dir.glob("pointa_nodes_overture_*.parquet"))
    if not overture_files:
        raise FileNotFoundError(f"No overture parquet files found in {in_dir}")

    n_cities = len(overture_files)
    for city_i, ovt_path in enumerate(overture_files, 1):
        bounds_fid = ovt_path.stem.split("_")[-1]
        reg_path = in_dir / f"pointa_nodes_registry_{bounds_fid}.parquet"
        if not reg_path.exists():
            continue
        print(f"[{city_i}/{n_cities}] fid={bounds_fid} …", end="", flush=True)

        try:
            ovt = pd.read_parquet(ovt_path)
            reg = pd.read_parquet(reg_path)
        except Exception as exc:  # parquet can be mid-write
            print(f"⚠ Skipping bounds_fid={bounds_fid}: failed to read parquet pair: {exc}")
            continue

        # Join by node_id (and bounds_fid as guardrail).
        key_cols = [c for c in ["bounds_fid", "node_id"] if c in ovt.columns and c in reg.columns]
        merged = ovt.merge(reg, on=key_cols, suffixes=("_ovt", "_reg"), how="inner")

        # Restrict to live nodes if available.
        if "live_ovt" in merged.columns:
            merged = merged[merged["live_ovt"]].copy()

        # Spatial block bootstrap: coordinates for distance-adaptive blocks.
        has_coords = "x_ovt" in merged.columns and "y_ovt" in merged.columns
        coords_x = merged["x_ovt"].to_numpy(dtype=float) if has_coords else None
        coords_y = merged["y_ovt"].to_numpy(dtype=float) if has_coords else None

        # Per-city RNG for reproducible block bootstrap. zlib.crc32 is a
        # stable digest; Python's built-in hash() is salted per process.
        city_seed = SPATIAL_BLOCK_BOOT_SEED + zlib.crc32(str(bounds_fid).encode()) % (2**31)
        boot_rng = np.random.default_rng(city_seed)

        for _cat_i, cat in enumerate(categories):
            for d in DISTANCES_LU:
                # Distance-adaptive blocks: block size = catchment distance.
                block_ids = None
                n_blocks = 0
                if has_coords:
                    block_ids = assign_spatial_blocks(coords_x, coords_y, block_size_m=d)
                    n_blocks = len(np.unique(block_ids))

                print(f" {cat[:3]}@{d}", end="", flush=True)

                col_nw_ovt = f"cc_{cat}_{d}_nw_ovt"
                col_nw_reg = f"cc_{cat}_{d}_nw_reg"
                col_wt_ovt = f"cc_{cat}_{d}_wt_ovt"
                col_wt_reg = f"cc_{cat}_{d}_wt_reg"
                col_near_ovt = f"cc_{cat}_nearest_max_{d}_ovt"
                col_near_reg = f"cc_{cat}_nearest_max_{d}_reg"

                for metric_label, col_ovt, col_reg, transform in [
                    ("count_nw", col_nw_ovt, col_nw_reg, np.log1p),
                    ("count_wt", col_wt_ovt, col_wt_reg, np.log1p),
                    ("nearest", col_near_ovt, col_near_reg, None),
                ]:
                    if col_ovt not in merged.columns or col_reg not in merged.columns:
                        continue
                    a_raw = merged[col_ovt].to_numpy(dtype=float)
                    b_raw = merged[col_reg].to_numpy(dtype=float)
                    a = transform(a_raw) if transform is not None else a_raw
                    b = transform(b_raw) if transform is not None else b_raw

                    rho = safe_spearman(a, b)
                    tau = safe_kendall(a, b)
                    k = max(10, int(0.1 * len(a)))
                    top_overlap = topk_overlap(a, b, k=k, largest=True)

                    row: dict = {
                        "bounds_fid": bounds_fid,
                        "category": cat,
                        "distance_m": d,
                        "metric": metric_label,
                        "spearman_rho": rho,
                        "kendall_tau": tau,
                        "top10_overlap": top_overlap,
                        "n_nodes": int(np.isfinite(a).sum()) if metric_label == "nearest" else int(len(a)),
                        "n_blocks": n_blocks,
                    }

                    # Spatial block bootstrap CIs for Spearman rho and top-10% overlap.
                    if block_ids is not None and n_blocks >= 5:
                        rho_lo, rho_hi = spatial_block_bootstrap_ci(
                            lambda idx, _a=a, _b=b: spearman_rho(_a[idx], _b[idx]),
                            block_ids=block_ids,
                            n_boot=SPATIAL_BLOCK_BOOT_N,
                            alpha=SPATIAL_BLOCK_BOOT_ALPHA,
                            rng=boot_rng,
                        )
                        ovlp_lo, ovlp_hi = spatial_block_bootstrap_ci(
                            lambda idx, _a=a, _b=b: topk_overlap(
                                _a[idx], _b[idx], k=max(10, int(0.1 * len(idx))), largest=True
                            ),
                            block_ids=block_ids,
                            n_boot=SPATIAL_BLOCK_BOOT_N,
                            alpha=SPATIAL_BLOCK_BOOT_ALPHA,
                            rng=boot_rng,
                        )
                        row.update(
                            {
                                "spearman_rho_ci_low": rho_lo,
                                "spearman_rho_ci_high": rho_hi,
                                "top10_overlap_ci_low": ovlp_lo,
                                "top10_overlap_ci_high": ovlp_hi,
                            }
                        )

                    rows.append(row)

                    # Nearest: additional within-tolerance assessment.
                    if metric_label == "nearest":
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() >= 10:
                            a_ok = a[ok]
                            b_ok = b[ok]
                            for tol in nearest_tolerances_m:
                                y_ovt = a_ok <= tol
                                y_reg = b_ok <= tol
                                scores = _binary_scores(y_reg, y_ovt)
                                nearest_row: dict = {
                                    "bounds_fid": bounds_fid,
                                    "category": cat,
                                    "nearest_max_m": d,
                                    "tolerance_m": float(tol),
                                    "n_nodes": int(ok.sum()),
                                    "n_blocks": n_blocks,
                                    **scores,
                                }
                                # Block bootstrap CI on F1 for nearest-within-tolerance.
                                if block_ids is not None and n_blocks >= 5:
                                    ok_blocks = block_ids[ok]

                                    def _f1_fn(idx, _a=a_ok, _b=b_ok, _tol=tol):
                                        s = _binary_scores((_b[idx] <= _tol), (_a[idx] <= _tol))
                                        return s["f1"]

                                    f1_lo, f1_hi = spatial_block_bootstrap_ci(
                                        _f1_fn,
                                        block_ids=ok_blocks,
                                        n_boot=SPATIAL_BLOCK_BOOT_N,
                                        alpha=SPATIAL_BLOCK_BOOT_ALPHA,
                                        rng=boot_rng,
                                    )
                                    nearest_row.update({"f1_ci_low": f1_lo, "f1_ci_high": f1_hi})
                                nearest_rows.append(nearest_row)

        print(" ✓", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"✓ Wrote: {out_csv}")

    if nearest_rows:
        out_near = pd.DataFrame(nearest_rows)
        out_near.to_csv(out_nearest_csv, index=False)
        print(f"✓ Wrote: {out_nearest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
