#!/usr/bin/env python3
"""Point A: Convert agreement summaries into a Support Matrix.

Reads the summary CSVs produced by `s02_summarise_node_agreement.py` /
`s03_summarise_city_agreement.py` and emits:
- A long-form support matrix CSV with agreement scores and CI-based labels
- A compact per-category minima table (CSV + LaTeX)

Downstream scripts (`s10` figures and related manuscript tables) read the continuous scores
from the CSV; the internal label column is used only for the sensitivity sweep.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import OUTPUT_DIR, PAPER_DATA_DIR, TABLE_DIR
from pointa_utils import spearman_rho


def _coerce_float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _load_thresholds(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("thresholds json must be an object")
    return data


def _thresholds_from_json(data: dict) -> dict[str, float]:
    dist_sp = data.get("distribution_spearman", {}) if isinstance(data.get("distribution_spearman", {}), dict) else {}
    hot = data.get("hotspot_top10", {}) if isinstance(data.get("hotspot_top10", {}), dict) else {}
    near = data.get("nearest_within_f1", {}) if isinstance(data.get("nearest_within_f1", {}), dict) else {}
    return {
        "dist_spearman_supported": _coerce_float(dist_sp, "supported_ge", 0.8),
        "dist_spearman_caution": _coerce_float(dist_sp, "caution_ge", 0.6),
        "hotspot_supported": _coerce_float(hot, "supported_ge", 0.6),
        "hotspot_caution": _coerce_float(hot, "caution_ge", 0.4),
        "nearest_f1_supported": _coerce_float(near, "supported_ge", 0.7),
        "nearest_f1_caution": _coerce_float(near, "caution_ge", 0.6),
    }


def _ci_from_json(data: dict) -> dict[str, float | int | bool]:
    ci = data.get("ci", {}) if isinstance(data.get("ci", {}), dict) else {}
    out: dict[str, float | int | bool] = {}
    if "alpha" in ci:
        out["alpha"] = _coerce_float(ci, "alpha", 0.05)
    if "n" in ci:
        try:
            out["n"] = int(ci.get("n"))
        except Exception:
            out["n"] = 2000
    if "seed" in ci:
        try:
            out["seed"] = int(ci.get("seed"))
        except Exception:
            out["seed"] = 123
    if "label_by_ci" in ci:
        out["label_by_ci"] = bool(ci.get("label_by_ci"))
    return out


def _label(score: float, *, supported_ge: float, caution_ge: float) -> str:
    if score is None or not np.isfinite(score):
        return "insufficient_data"
    if float(score) >= float(supported_ge):
        return "supported"
    if float(score) >= float(caution_ge):
        return "caution"
    return "not_supported"


def _label_ci(
    *,
    score: float,
    ci_low: float,
    ci_high: float,
    supported_ge: float,
    caution_ge: float,
) -> str:
    """Label based on confidence bounds.

    Policy:
    - supported: even the lower bound clears the supported cut-point
    - not_supported: even the upper bound fails to clear the caution cut-point
    - caution: everything in-between (including wide/uncertain intervals)
    """
    if score is None or not np.isfinite(float(score)):
        return "insufficient_data"
    if not np.isfinite(float(ci_low)) or not np.isfinite(float(ci_high)):
        return _label(float(score), supported_ge=supported_ge, caution_ge=caution_ge)
    if float(ci_low) >= float(supported_ge):
        return "supported"
    if float(ci_high) < float(caution_ge):
        return "not_supported"
    return "caution"


def _bootstrap_ci(
    values: np.ndarray,
    *,
    stat_fn,
    n: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or n <= 0:
        return (float("nan"), float("nan"))
    stats = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        samp = rng.choice(v, size=v.size, replace=True)
        stats[i] = float(stat_fn(samp))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5 or n <= 0:
        return (float("nan"), float("nan"))
    stats = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        idx = rng.integers(0, x.size, size=x.size)
        stats[i] = spearman_rho(x[idx], y[idx])
    stats = stats[np.isfinite(stats)]
    if stats.size < 10:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def _min_supported(x: pd.DataFrame, key_col: str, score_col: str, *, supported_ge: float) -> float:
    if x.empty:
        return float("nan")
    sub = x[np.isfinite(x[score_col].astype(float))].copy()
    if sub.empty:
        return float("nan")
    sub["_ok"] = sub[score_col].astype(float) >= float(supported_ge)
    sub = sub[sub["_ok"]]
    if sub.empty:
        return float("nan")
    return float(sub[key_col].min())


def _write_latex_table(df: pd.DataFrame, out_path: Path, float_cols: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2 = df.copy()
    for c in float_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").round(0).astype("Int64")
    out_path.write_text(df2.to_latex(index=False, escape=True))


# ---------------------------------------------------------------------------
# Panel functions — each returns list[dict] of support-matrix rows.
# ---------------------------------------------------------------------------


def _make_row(
    *,
    category,
    dimension,
    value,
    metric_family,
    decision_type,
    score,
    ci_low,
    ci_high,
    n_cities,
    supported_ge,
    caution_ge,
    label_by_ci,
    estimate_type="ci",
) -> dict:
    label = (
        _label_ci(
            score=float(score),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            supported_ge=float(supported_ge),
            caution_ge=float(caution_ge),
        )
        if label_by_ci
        else _label(float(score), supported_ge=float(supported_ge), caution_ge=float(caution_ge))
    )
    return {
        "category": category,
        "dimension": dimension,
        "value": float(value),
        "metric_family": metric_family,
        "decision_type": decision_type,
        "estimate_type": estimate_type,
        "score": float(score) if score is not None else float("nan"),
        "ci_low": float(ci_low) if ci_low is not None else float("nan"),
        "ci_high": float(ci_high) if ci_high is not None else float("nan"),
        "label": label,
        "n_cities": int(n_cities) if n_cities is not None else 0,
    }


def _panel_hotspot_within(
    node_raw: pd.DataFrame,
    *,
    label_by_ci: bool,
    supported_ge: float,
    caution_ge: float,
    **_kw,
) -> list[dict]:
    """Within-city hotspot support: median of per-city block-bootstrap CIs."""
    rows: list[dict] = []
    if node_raw.empty:
        return rows
    for metric in ["count_nw", "count_wt"]:
        sub = node_raw[node_raw["metric"] == metric]
        if sub.empty:
            continue
        for (category, distance_m), g in sub.groupby(["category", "distance_m"], dropna=False):
            score, ci_low, ci_high = _median_block_ci(g, "top10_overlap")
            rows.append(
                _make_row(
                    category=category,
                    dimension="distance_m",
                    value=distance_m,
                    metric_family=metric,
                    decision_type="hotspot_top10",
                    score=score,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    n_cities=g["bounds_fid"].nunique(),
                    supported_ge=supported_ge,
                    caution_ge=caution_ge,
                    label_by_ci=label_by_ci,
                )
            )
    return rows


def _median_block_ci(g: pd.DataFrame, score_col: str) -> tuple[float, float, float]:
    """Extract median score and median block-bootstrap CI bounds from s02 output."""
    vals = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
    score = float(np.nanmedian(vals)) if np.isfinite(np.nanmedian(vals)) else float("nan")
    lo_col = f"{score_col}_ci_low"
    hi_col = f"{score_col}_ci_high"
    if lo_col in g.columns and hi_col in g.columns:
        lo = pd.to_numeric(g[lo_col], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(g[hi_col], errors="coerce").to_numpy(dtype=float)
        ci_low = float(np.nanmedian(lo)) if np.any(np.isfinite(lo)) else float("nan")
        ci_high = float(np.nanmedian(hi)) if np.any(np.isfinite(hi)) else float("nan")
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return score, ci_low, ci_high


def _panel_distribution_within(
    node_raw: pd.DataFrame,
    *,
    label_by_ci: bool,
    supported_ge: float,
    caution_ge: float,
    **_kw,
) -> list[dict]:
    """Within-city distribution agreement: median of per-city block-bootstrap CIs."""
    rows: list[dict] = []
    if node_raw.empty:
        return rows
    for metric in ["count_nw", "count_wt"]:
        sub = node_raw[node_raw["metric"] == metric]
        if sub.empty:
            continue
        for (category, distance_m), g in sub.groupby(["category", "distance_m"], dropna=False):
            n_cities = g["bounds_fid"].nunique()
            # Spearman ρ — use block bootstrap CIs from s02 directly.
            score, ci_low, ci_high = _median_block_ci(g, "spearman_rho")
            rows.append(
                _make_row(
                    category=category,
                    dimension="distance_m",
                    value=distance_m,
                    metric_family=metric,
                    decision_type="distribution_spearman",
                    score=score,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    n_cities=n_cities,
                    supported_ge=supported_ge,
                    caution_ge=caution_ge,
                    label_by_ci=label_by_ci,
                )
            )
            # Kendall τ (point estimate only — no block bootstrap CI).
            if "kendall_tau" in g.columns:
                tau_vals = pd.to_numeric(g["kendall_tau"], errors="coerce").to_numpy(dtype=float)
                tau_score = float(np.nanmedian(tau_vals)) if np.isfinite(np.nanmedian(tau_vals)) else float("nan")
                rows.append(
                    _make_row(
                        category=category,
                        dimension="distance_m",
                        value=distance_m,
                        metric_family=metric,
                        decision_type="distribution_kendall",
                        score=tau_score,
                        ci_low=float("nan"),
                        ci_high=float("nan"),
                        n_cities=n_cities,
                        supported_ge=supported_ge,
                        caution_ge=caution_ge,
                        label_by_ci=False,
                    )
                )
    return rows


def _panel_benchmark_between(
    city: pd.DataFrame,
    *,
    ci_n: int,
    ci_alpha: float,
    rng: np.random.Generator,
    label_by_ci: bool,
    nearest_f1_supported: float,
    nearest_f1_caution: float,
) -> list[dict]:
    """Between-city benchmarking: nearest-share agreement (harmonic-mean city F1)."""
    rows: list[dict] = []
    if city.empty:
        return rows

    # Between-city nearest: harmonic mean of city-level shares (city F1).
    # For each city, compute H(share_ovt, share_reg) = 2*p*q / (p+q).
    sub = city[(city["metric"] == "nearest_within_share") & (city["source"].isin(["ovt", "reg"]))]
    if not sub.empty:
        for (category, tolerance_m), g in sub.groupby(["category", "tolerance_m"], dropna=False):
            piv = g.pivot_table(index="bounds_fid", columns="source", values="value", aggfunc="mean")
            if "ovt" not in piv.columns or "reg" not in piv.columns:
                continue
            x = piv["ovt"].to_numpy(dtype=float)
            y = piv["reg"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            xm, ym = x[mask], y[mask]
            if xm.size < 3:
                continue
            denom = xm + ym
            city_f1 = np.where(denom > 0, 2.0 * xm * ym / denom, 0.0)
            score = float(np.nanmedian(city_f1))
            ci_low, ci_high = _bootstrap_ci(
                city_f1,
                stat_fn=lambda a: float(np.nanmedian(a)),
                n=ci_n,
                alpha=ci_alpha,
                rng=rng,
            )
            rows.append(
                _make_row(
                    category=category,
                    dimension="tolerance_m",
                    value=tolerance_m,
                    metric_family="nearest",
                    decision_type="benchmark_nearest",
                    score=score,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    n_cities=int(mask.sum()),
                    supported_ge=nearest_f1_supported,
                    caution_ge=nearest_f1_caution,
                    label_by_ci=label_by_ci,
                )
            )

    return rows


def _panel_nearest_f1(
    nearest_raw: pd.DataFrame,
    *,
    label_by_ci: bool,
    supported_ge: float,
    caution_ge: float,
    **_kw,
) -> list[dict]:
    """Within-city nearest F1: median of per-city block-bootstrap CIs."""
    rows: list[dict] = []
    if nearest_raw.empty:
        return rows
    for (category, tolerance_m), g in nearest_raw.groupby(["category", "tolerance_m"], dropna=False):
        score, ci_low, ci_high = _median_block_ci(g, "f1")
        rows.append(
            _make_row(
                category=category,
                dimension="tolerance_m",
                value=tolerance_m,
                metric_family="nearest",
                decision_type="nearest_within_f1",
                score=score,
                ci_low=ci_low,
                ci_high=ci_high,
                n_cities=g["bounds_fid"].nunique(),
                supported_ge=supported_ge,
                caution_ge=caution_ge,
                label_by_ci=label_by_ci,
            )
        )
    return rows


def main() -> int:
    # Paths.
    node_raw_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_node_agreement.csv"
    nearest_raw_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_nearest_tolerance_agreement.csv"
    city_summaries_csv = OUTPUT_DIR / "csv" / "pointa_accessibility_city_summaries.csv"
    out_support_matrix_csv = OUTPUT_DIR / "csv" / "pointa_support_matrix.csv"
    out_minima_csv = OUTPUT_DIR / "csv" / "pointa_support_minima.csv"
    out_minima_tex = TABLE_DIR / "table_pointa_support_minima.tex"
    thresholds_json = PAPER_DATA_DIR / "code" / "pointa_thresholds.json"
    out_thresholds_used_json = OUTPUT_DIR / "csv" / "pointa_support_thresholds_used.json"

    # Defaults.
    ci_alpha = 0.05
    ci_n = 2000
    ci_seed = 123
    label_by_ci = True
    dist_spearman_supported = 0.8
    dist_spearman_caution = 0.6
    hotspot_supported = 0.6
    hotspot_caution = 0.4
    nearest_f1_supported = 0.7
    nearest_f1_caution = 0.6

    # If a frozen thresholds JSON exists, use it.
    frozen_thresholds: dict | None = None
    if thresholds_json.exists():
        try:
            frozen_thresholds = _load_thresholds(thresholds_json)
            thr = _thresholds_from_json(frozen_thresholds)
            ci_policy = _ci_from_json(frozen_thresholds)
            dist_spearman_supported = thr["dist_spearman_supported"]
            dist_spearman_caution = thr["dist_spearman_caution"]
            hotspot_supported = thr["hotspot_supported"]
            hotspot_caution = thr["hotspot_caution"]
            nearest_f1_supported = thr["nearest_f1_supported"]
            nearest_f1_caution = thr["nearest_f1_caution"]
            if "alpha" in ci_policy:
                ci_alpha = float(ci_policy["alpha"])
            if "n" in ci_policy:
                ci_n = int(ci_policy["n"])
            if "seed" in ci_policy:
                ci_seed = int(ci_policy["seed"])
            if "label_by_ci" in ci_policy:
                label_by_ci = bool(ci_policy["label_by_ci"])
            print(f"✓ Using frozen thresholds from: {thresholds_json}")
        except Exception as exc:
            print(f"⚠ Failed to load thresholds JSON ({thresholds_json}): {exc}")

    have_raw = node_raw_csv.exists() and nearest_raw_csv.exists() and city_summaries_csv.exists()
    if not have_raw:
        print("Support matrix: missing required CSVs; run s2/s3 first.")
        return 0

    if out_support_matrix_csv.exists() and out_minima_csv.exists() and out_thresholds_used_json.exists():
        print(f"✓ Support matrix outputs already exist, skipping: {out_support_matrix_csv.name}, {out_minima_csv.name}")
        return 0

    rng = np.random.default_rng(ci_seed)

    node_raw = pd.read_csv(node_raw_csv)
    nearest_raw = pd.read_csv(nearest_raw_csv)
    city = pd.read_csv(city_summaries_csv)

    # Record thresholds used for this run.
    used = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "thresholds": {
            "distribution_spearman": {
                "supported_ge": dist_spearman_supported,
                "caution_ge": dist_spearman_caution,
            },
            "hotspot_top10": {
                "supported_ge": hotspot_supported,
                "caution_ge": hotspot_caution,
            },
            "nearest_within_f1": {
                "supported_ge": nearest_f1_supported,
                "caution_ge": nearest_f1_caution,
            },
        },
        "tolerances_m": sorted({float(x) for x in nearest_raw.get("tolerance_m", []) if pd.notnull(x)}),
        "distances_m": sorted({float(x) for x in node_raw.get("distance_m", []) if pd.notnull(x)}),
        "thresholds_source": str(thresholds_json) if frozen_thresholds is not None else "defaults",
        "ci": {
            "alpha": ci_alpha,
            "n": ci_n,
            "seed": ci_seed,
            "label_by_ci": label_by_ci,
        },
    }
    out_thresholds_used_json.parent.mkdir(parents=True, exist_ok=True)
    out_thresholds_used_json.write_text(json.dumps(used, indent=2))
    print(f"✓ Wrote: {out_thresholds_used_json}")

    # Build support matrix from panel functions.
    ci_kw = dict(ci_n=ci_n, ci_alpha=ci_alpha, rng=rng, label_by_ci=label_by_ci)

    rows: list[dict] = []
    rows.extend(_panel_hotspot_within(node_raw, **ci_kw, supported_ge=hotspot_supported, caution_ge=hotspot_caution))
    rows.extend(
        _panel_distribution_within(
            node_raw, **ci_kw, supported_ge=dist_spearman_supported, caution_ge=dist_spearman_caution
        )
    )
    rows.extend(
        _panel_benchmark_between(
            city,
            **ci_kw,
            nearest_f1_supported=nearest_f1_supported,
            nearest_f1_caution=nearest_f1_caution,
        )
    )
    rows.extend(
        _panel_nearest_f1(nearest_raw, **ci_kw, supported_ge=nearest_f1_supported, caution_ge=nearest_f1_caution)
    )
    support = pd.DataFrame(rows)
    out_support_matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    support.to_csv(out_support_matrix_csv, index=False)
    print(f"✓ Wrote: {out_support_matrix_csv}")

    # Compact minima table: minimum distance/tolerance per category at reference threshold.
    minima_rows: list[dict] = []
    categories = sorted(set(support["category"].dropna().astype(str)))

    for cat in categories:
        row: dict = {"category": cat}

        for mf in ["count_nw", "count_wt"]:
            sub = support[
                (support["category"] == cat)
                & (support["metric_family"] == mf)
                & (support["decision_type"] == "hotspot_top10")
                & (support["dimension"] == "distance_m")
            ]
            row[f"min_supported_hotspot_{mf}_m"] = _min_supported(sub, "value", "score", supported_ge=hotspot_supported)

        sub = support[
            (support["category"] == cat)
            & (support["decision_type"] == "nearest_within_f1")
            & (support["dimension"] == "tolerance_m")
        ]
        row["min_supported_nearest_f1_tol_m"] = _min_supported(sub, "value", "score", supported_ge=nearest_f1_supported)

        sub = support[
            (support["category"] == cat)
            & (support["decision_type"] == "benchmark_nearest")
            & (support["dimension"] == "tolerance_m")
        ]
        row["min_supported_benchmark_nearest_tol_m"] = _min_supported(
            sub, "value", "score", supported_ge=nearest_f1_supported
        )

        minima_rows.append(row)

    minima = pd.DataFrame(minima_rows)
    out_minima_csv.parent.mkdir(parents=True, exist_ok=True)
    minima.to_csv(out_minima_csv, index=False)
    print(f"✓ Wrote: {out_minima_csv}")

    float_cols = [c for c in minima.columns if c != "category"]
    _write_latex_table(minima, out_minima_tex, float_cols=float_cols)
    print(f"✓ Wrote: {out_minima_tex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
