"""Statistical audit helpers: Newey-West t-stat, PSR/DSR (N=3), and CSCV PBO."""

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
) -> tuple[float, float]:
    mat = np.asarray(active_returns_matrix, dtype=np.float64)
    _, n_trials = mat.shape
    if n_trials <= 1:
        _, psr, _, _ = probabilistic_sharpe_ratio(mat[:, target_col], 0.0, periods_per_year)
        return 0.0, psr

    srs = []
    for j in range(n_trials):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        sd = col.std(ddof=1)
        srs.append(col.mean() / sd if sd > 0 else 0.0)
    sr_std = math.sqrt(max(float(np.var(srs, ddof=1)), 1e-12))
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
