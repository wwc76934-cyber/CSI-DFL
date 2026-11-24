"""Statistical audit helpers: Newey-West t-stat, paired circular block bootstrap, PSR/DSR (N=3 & N=10), CSCV PBO."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew


EULER_GAMMA = 0.5772156649015329


def newey_west_t_stat(series: np.ndarray, lag: int = 4) -> tuple[float, float, float]:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 5:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    e = x - mean
    var_nw = float(np.dot(e, e) / t)
    max_lag = min(lag, t - 1)
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1)
        var_nw += 2.0 * w * float(np.dot(e[k:], e[:-k]) / t)
    se = math.sqrt(max(var_nw, 1e-18) / t)
    t_stat = mean / se if se > 0 else np.nan
    p_val = float(2.0 * (1.0 - norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else np.nan
    return mean, float(t_stat), p_val


def annualized_ir_from_returns(active_returns: np.ndarray, periods_per_year: int = 252) -> float:
    x = np.asarray(active_returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = float(x.std(ddof=1))
    return float(math.sqrt(periods_per_year) * x.mean() / sd) if sd > 0 else np.nan


def paired_return_test(
    ret_a: np.ndarray,
    ret_b: np.ndarray,
    bench_ret: np.ndarray | None = None,
    lag: int = 4,
    n_boot: int = 2000,
    block_size: int = 10,
    periods_per_year: int = 252,
    seed: int = 42,
) -> dict[str, float]:
    """Paired daily return difference test (A - B) with Newey-West t/p and circular block-bootstrap 95% CIs."""
    a = np.asarray(ret_a, dtype=np.float64)
    b = np.asarray(ret_b, dtype=np.float64)
    if bench_ret is not None:
        bm = np.asarray(bench_ret, dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(bm)
        a, b, bm = a[mask], b[mask], bm[mask]
    else:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        bm = None

    diff = a - b
    n = len(diff)
    if n < 20:
        return {
            "days": n,
            "mean_daily_bps": np.nan,
            "ann_diff": np.nan,
            "te_diff": np.nan,
            "ir_diff": np.nan,
            "nw_tstat": np.nan,
            "nw_pval": np.nan,
            "ann_diff_ci_lo": np.nan,
            "ann_diff_ci_hi": np.nan,
            "ir_gap": np.nan,
            "ir_gap_ci_lo": np.nan,
            "ir_gap_ci_hi": np.nan,
        }

    mean_d, t_nw, p_nw = newey_west_t_stat(diff, lag=lag)
    te_diff = float(diff.std(ddof=1) * math.sqrt(periods_per_year))
    ir_diff = annualized_ir_from_returns(diff, periods_per_year)

    if bm is not None:
        act_a = a - bm
        act_b = b - bm
        ir_gap = annualized_ir_from_returns(act_a, periods_per_year) - annualized_ir_from_returns(act_b, periods_per_year)
    else:
        act_a, act_b, ir_gap = None, None, np.nan

    rng = np.random.default_rng(seed)
    blk = max(1, min(int(block_size), n // 2))
    n_blocks = int(math.ceil(n / blk))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(blk)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(n_boot, -1)[:, :n] % n

    boot_diff = diff[idx]
    boot_ann = boot_diff.mean(axis=1) * periods_per_year
    ann_lo, ann_hi = np.percentile(boot_ann, [2.5, 97.5])

    if act_a is not None and act_b is not None:
        ba = act_a[idx]
        bb = act_b[idx]
        sd_a = ba.std(axis=1, ddof=1)
        sd_b = bb.std(axis=1, ddof=1)
        ir_a = np.where(sd_a > 1e-12, math.sqrt(periods_per_year) * ba.mean(axis=1) / sd_a, np.nan)
        ir_b = np.where(sd_b > 1e-12, math.sqrt(periods_per_year) * bb.mean(axis=1) / sd_b, np.nan)
        boot_ir_gap = ir_a - ir_b
        valid_gap = boot_ir_gap[np.isfinite(boot_ir_gap)]
        ir_lo, ir_hi = np.percentile(valid_gap, [2.5, 97.5]) if len(valid_gap) else (np.nan, np.nan)
    else:
        ir_lo, ir_hi = np.nan, np.nan

    return {
        "days": int(n),
        "mean_daily_bps": float(mean_d * 10000.0),
        "ann_diff": float(mean_d * periods_per_year),
        "te_diff": te_diff,
        "ir_diff": float(ir_diff),
        "nw_tstat": float(t_nw),
        "nw_pval": float(p_nw),
        "ann_diff_ci_lo": float(ann_lo),
        "ann_diff_ci_hi": float(ann_hi),
        "ir_gap": float(ir_gap),
        "ir_gap_ci_lo": float(ir_lo),
        "ir_gap_ci_hi": float(ir_hi),
    }


def probabilistic_sharpe_ratio(
    active_returns: np.ndarray,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float, float, float]:
    x = np.asarray(active_returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return np.nan, np.nan, np.nan, np.nan
    sr = float(x.mean() / sd)
    sr0 = float(benchmark_sr / math.sqrt(periods_per_year))
    sk = float(skew(x, bias=False))
    kt = float(kurtosis(x, fisher=False, bias=False))
    denom_sq = 1.0 - sk * sr + ((kt - 1.0) / 4.0) * (sr ** 2)
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(max(denom_sq, 1e-12))
    return sr, float(norm.cdf(z)), sk, kt - 3.0


def deflated_sharpe_ratio(
    active_returns_matrix: np.ndarray,
    target_col: int,
    periods_per_year: int = 252,
    n_trials_override: int | None = None,
) -> tuple[float, float]:
    mat = np.asarray(active_returns_matrix, dtype=np.float64)
    _, n_cols = mat.shape
    n_trials = int(n_cols if n_trials_override is None else max(n_cols, n_trials_override))
    if n_trials <= 1:
        _, psr, _, _ = probabilistic_sharpe_ratio(mat[:, target_col], 0.0, periods_per_year)
        return 0.0, psr

    srs = []
    for j in range(n_cols):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        sd = col.std(ddof=1)
        srs.append(col.mean() / sd if sd > 0 else 0.0)
    sr_std = math.sqrt(max(float(np.var(srs, ddof=1)), 1e-12)) if len(srs) > 1 else 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0_ann = sr_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2) * math.sqrt(periods_per_year)
    _, dsr, _, _ = probabilistic_sharpe_ratio(mat[:, target_col], sr0_ann, periods_per_year)
    return float(sr0_ann), float(dsr)


def cscv_pbo(
    active_returns_matrix: np.ndarray,
    n_blocks: int = 16,
    max_splits: int = 500,
    seed: int = 42,
) -> tuple[float, float, int]:
    mat = np.asarray(active_returns_matrix, dtype=np.float64)
    n_obs, n_strat = mat.shape
    if n_strat < 2 or n_obs < n_blocks * 5:
        return np.nan, np.nan, 0
    block_size = n_obs // n_blocks
    blocks = mat[: block_size * n_blocks].reshape(n_blocks, block_size, n_strat)

    all_combos = list(combinations(range(n_blocks), n_blocks // 2))
    if len(all_combos) > max_splits:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_combos), size=max_splits, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    logits, oos_pcts = [], []
    all_set = set(range(n_blocks))
    for is_idx in combos:
        oos_idx = sorted(all_set - set(is_idx))
        is_ret = blocks[list(is_idx)].reshape(-1, n_strat)
        oos_ret = blocks[oos_idx].reshape(-1, n_strat)
        is_sd = is_ret.std(axis=0, ddof=1)
        is_sr = np.where(is_sd > 0, is_ret.mean(axis=0) / is_sd, -np.inf)
        best_j = int(np.argmax(is_sr))

        oos_sd = oos_ret.std(axis=0, ddof=1)
        oos_sr = np.where(oos_sd > 0, oos_ret.mean(axis=0) / oos_sd, -np.inf)
        order = np.argsort(oos_sr, kind="mergesort")
        ranks = np.empty(n_strat, dtype=np.float64)
        ranks[order] = np.arange(1, n_strat + 1, dtype=np.float64)
        omega = ranks[best_j] / (n_strat + 1.0)
        oos_pcts.append(omega)
        logits.append(math.log(omega / (1.0 - omega)))

    return float(np.mean(np.array(logits) <= 0.0)), float(np.median(oos_pcts)), len(combos)
