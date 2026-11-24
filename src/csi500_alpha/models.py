"""Model ladder: B0_FloatMV_Anchor, S0_EqualWeight, S0_LowTurn_Band, S1_RollingRidge, S2_WhiteboxDFL."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .factors import FACTORS
from .projection import (
    DifferentiableIEProjection,
    build_constraint_matrix_numpy,
    compute_active_bounds_numpy,
    precompute_eq_operator_numpy,
    project_portfolio_numpy,
)


@dataclass
class WeekRecord:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    codes: list[str]
    x: np.ndarray              # (N, F) float32
    industry_codes: np.ndarray # (N,) str
    anchor: np.ndarray         # (N,) float32
    beta: np.ndarray           # (N,) float32
    size_z: np.ndarray         # (N,) float32
    A: np.ndarray              # (M, N) float32
    B_eq: np.ndarray           # (M, N) float32
    lo: np.ndarray             # (N,) float32
    hi: np.ndarray             # (N,) float32
    target: np.ndarray         # (N,) float32
    valid_target: np.ndarray   # (N,) bool
    stock_ret: np.ndarray      # (N,) float32 open-to-next-rebalance-open return
    excess_ret: np.ndarray     # (N,) float32 stock_ret - anchor_return_hold
    xx: np.ndarray             # (F, F) float64 cross-moment for Ridge
    xy: np.ndarray             # (F,) float64 cross-moment for Ridge
    has_mature_target: bool


def prepare_week_records(
    panel: pd.DataFrame,
    cfg: dict[str, Any],
    factor_cols: list[str] | None = None,
) -> list[WeekRecord]:
    p_cfg = cfg["portfolio"]
    active_limit = float(p_cfg["active_limit"])
    single_cap = float(p_cfg["single_stock_cap"])
    eps = float(p_cfg["dykstra_eps"])
    f_cols = list(FACTORS if factor_cols is None else factor_cols)

    records: list[WeekRecord] = []
    for day, grp in panel.groupby("signal_date", sort=True):
        g = grp.sort_values("code").reset_index(drop=True)
        if len(g) < 100:
            continue
        x = np.nan_to_num(g[f_cols].to_numpy(dtype=np.float32), nan=0.0)
        ind_codes = g["industry"].fillna("Other").astype(str).to_numpy()
        anchor = g["weight_float_mv"].to_numpy(dtype=np.float64)
        anchor = np.maximum(anchor, 0.0)
        anchor = (anchor / anchor.sum()).astype(np.float32)

        beta = g["BETA_120"].fillna(1.0).to_numpy(dtype=np.float32)
        size_z = g["size_z"].fillna(0.0).to_numpy(dtype=np.float32)
        A = build_constraint_matrix_numpy(ind_codes, beta, size_z)
        B_eq = precompute_eq_operator_numpy(A, eps=eps)
        lo, hi = compute_active_bounds_numpy(anchor, active_limit=active_limit, single_cap=single_cap)

        target_raw = g["risk_adjusted_target"].to_numpy(dtype=np.float32)
        valid_target = np.isfinite(target_raw)
        target_clean = np.where(valid_target, target_raw, 0.0).astype(np.float32)

        sret_raw = g["stock_return_5d"].to_numpy(dtype=np.float32)
        sret_clean = np.where(np.isfinite(sret_raw), np.clip(sret_raw, -0.5, 1.5), 0.0).astype(np.float32)

        bench_col = "anchor_return_hold" if "anchor_return_hold" in g.columns else "h00905_return_5d"
        excess_raw = (g["stock_return_5d"] - g[bench_col]).to_numpy(dtype=np.float32)
        excess_clean = np.where(np.isfinite(excess_raw), excess_raw, 0.0).astype(np.float32)

        nv = max(1, int(valid_target.sum()))
        xv = x[valid_target].astype(np.float64)
        yv = target_clean[valid_target].astype(np.float64)
        xc = xv - xv.mean(axis=0) if len(xv) else xv
        xx = (xc.T @ xc) / nv
        xy = (xc.T @ yv) / nv

        exit_vals = g["exit_date"].dropna()
        exit_dt = pd.Timestamp(exit_vals.iat[0]) if not exit_vals.empty else None
        has_mature = exit_dt is not None and int(valid_target.sum()) >= 100

        records.append(
            WeekRecord(
                signal_date=pd.Timestamp(day),
                entry_date=pd.Timestamp(g["entry_date"].iat[0]),
                exit_date=exit_dt,
                codes=g["code"].astype(str).tolist(),
                x=x,
                industry_codes=ind_codes,
                anchor=anchor,
                beta=beta,
                size_z=size_z,
                A=A,
                B_eq=B_eq,
                lo=lo,
                hi=hi,
                target=target_clean,
                valid_target=valid_target,
                stock_ret=sret_clean,
                excess_ret=excess_clean,
                xx=xx,
                xy=xy,
                has_mature_target=has_mature,
            )
        )
    return records


def zscore_np(arr: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    sd = x.std(ddof=0)
    return ((x - x.mean()) / sd if sd > 1e-12 else x * 0.0).astype(np.float32)


class WhiteboxDFL(nn.Module):
    """Whitebox DFL: convex combination of F factors + DifferentiableIEProjection."""

    def __init__(self, num_factors: int = 10, gamma: float = 0.005, train_steps: int = 12, eps: float = 1e-7) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_factors, dtype=torch.float32))
        self.gamma = gamma
        self.proj = DifferentiableIEProjection(num_steps=train_steps, eps=eps)

    def factor_weights(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def score_cross_section(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        theta = self.factor_weights()
        raw = torch.matmul(x, theta)
        if valid_mask is None:
            mean = raw.mean(dim=-1, keepdim=True)
            sd = raw.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-6)
            return (raw - mean) / sd
        cnt = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (raw * valid_mask).sum(dim=-1, keepdim=True) / cnt
        var = (((raw - mean) ** 2) * valid_mask).sum(dim=-1, keepdim=True) / cnt
        return ((raw - mean) / torch.sqrt(var.clamp(min=1e-12))) * valid_mask

    def forward_portfolio(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        B_eq: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_z = self.score_cross_section(x, valid_mask=valid_mask)
        dw0 = self.gamma * score_z
        dw = self.proj(dw0, A, lo, hi, B_eq=B_eq, num_steps=num_steps)
        return dw, score_z, self.factor_weights()


def _pack_chunk_tensors(
    chunk_weeks: list[WeekRecord],
    code_to_u: dict[str, int],
    u_count: int,
) -> dict[str, Any]:
    t_len = len(chunk_weeks)
    n_max = max(len(w.codes) for w in chunk_weeks)
    m_max = max(w.A.shape[0] for w in chunk_weeks)
    f_dim = chunk_weeks[0].x.shape[1]

    x = np.zeros((t_len, n_max, f_dim), dtype=np.float32)
    anchor = np.zeros((t_len, n_max), dtype=np.float32)
    A = np.zeros((t_len, m_max, n_max), dtype=np.float32)
    B_eq = np.zeros((t_len, m_max, n_max), dtype=np.float32)
    lo = np.zeros((t_len, n_max), dtype=np.float32)
    hi = np.zeros((t_len, n_max), dtype=np.float32)
    target = np.zeros((t_len, n_max), dtype=np.float32)
    valid_target = np.zeros((t_len, n_max), dtype=np.float32)
    stock_ret = np.zeros((t_len, n_max), dtype=np.float32)
    excess_ret = np.zeros((t_len, n_max), dtype=np.float32)
    valid_mask = np.zeros((t_len, n_max), dtype=np.float32)
    u_idx = np.full((t_len, n_max), u_count, dtype=np.int64)

    for t, w in enumerate(chunk_weeks):
        n = len(w.codes)
        m = w.A.shape[0]
        x[t, :n, :] = w.x
        anchor[t, :n] = w.anchor
        A[t, :m, :n] = w.A
        B_eq[t, :m, :n] = w.B_eq
        lo[t, :n] = w.lo
        hi[t, :n] = w.hi
        target[t, :n] = w.target
        valid_target[t, :n] = w.valid_target.astype(np.float32)
        stock_ret[t, :n] = w.stock_ret
        excess_ret[t, :n] = w.excess_ret
        valid_mask[t, :n] = 1.0
        u_idx[t, :n] = [code_to_u[c] for c in w.codes]

    return {
        "x": torch.from_numpy(x),
        "anchor": torch.from_numpy(anchor),
        "A": torch.from_numpy(A),
        "B_eq": torch.from_numpy(B_eq),
        "lo": torch.from_numpy(lo),
        "hi": torch.from_numpy(hi),
        "target": torch.from_numpy(target),
        "valid_target": torch.from_numpy(valid_target),
        "stock_ret": torch.from_numpy(stock_ret),
        "excess_ret": torch.from_numpy(excess_ret),
        "valid_mask": torch.from_numpy(valid_mask),
        "u_idx": torch.from_numpy(u_idx),
        "u_count": u_count,
    }


def _batch_pearson_ic(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    p_mean = (pred * mask).sum(dim=1, keepdim=True) / cnt
    t_mean = (target * mask).sum(dim=1, keepdim=True) / cnt
    dp = (pred - p_mean) * mask
    dt = (target - t_mean) * mask
    cov = (dp * dt).sum(dim=1)
    denom = torch.sqrt((dp * dp).sum(dim=1).clamp(min=1e-12) * (dt * dt).sum(dim=1).clamp(min=1e-12))
    return (cov / denom).mean()


def _split_chunks(train_weeks: list[WeekRecord], chunk_size: int) -> list[dict[str, Any]]:
    union_codes = sorted({c for w in train_weeks for c in w.codes})
    code_to_u = {c: i for i, c in enumerate(union_codes)}
    u_count = len(union_codes)

    n = len(train_weeks)
    starts = list(range(0, n, chunk_size))
    if len(starts) > 1 and (n - starts[-1]) < 8:
        starts.pop()
    chunks = []
    for i, s in enumerate(starts):
        e = n if i == len(starts) - 1 else starts[i + 1]
        chunks.append(_pack_chunk_tensors(train_weeks[s:e], code_to_u, u_count))
    return chunks


def _train_whitebox_dfl(
    chunks: list[dict[str, Any]],
    num_factors: int,
    gamma: float,
    train_steps: int,
    eps: float,
    epochs: int,
    lr: float,
    prior_lambda: float,
    train_fee: float,
    aux_ic_w: float,
    seed: int,
) -> WhiteboxDFL:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = WhiteboxDFL(num_factors=num_factors, gamma=gamma, train_steps=train_steps, eps=eps)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    prior_w = torch.full((num_factors,), 1.0 / num_factors, dtype=torch.float32)

    net.train()
    for _ in range(epochs):
        prev_chunk_drifted: torch.Tensor | None = None
        for ch in chunks:
            opt.zero_grad()
            dw, score_z, theta = net.forward_portfolio(
                ch["x"], ch["A"], ch["lo"], ch["hi"],
                B_eq=ch["B_eq"], valid_mask=ch["valid_mask"], num_steps=train_steps,
            )
            t_len = dw.shape[0]
            u_cnt = ch["u_count"]

            # Full portfolio weights w_t = anchor_t + dw_t and drifted full-portfolio turnover
            w_t = (ch["anchor"] + dw) * ch["valid_mask"]
            w_full = torch.zeros((t_len, u_cnt + 1), dtype=dw.dtype)
            w_full = w_full.scatter_add(1, ch["u_idx"], w_t)[:, :u_cnt]

            ret_full = torch.zeros((t_len, u_cnt + 1), dtype=dw.dtype)
            ret_full = ret_full.scatter_add(1, ch["u_idx"], ch["stock_ret"] * ch["valid_mask"])[:, :u_cnt]

            w_grown = w_full * (1.0 + ret_full)
            w_drifted = w_grown / w_grown.sum(dim=1, keepdim=True).clamp(min=1e-6)

            if t_len > 1:
                turn_inner = torch.abs(w_full[1:] - w_drifted[:-1]).sum(dim=1)
            else:
                turn_inner = torch.zeros(0, dtype=dw.dtype)

            if prev_chunk_drifted is not None:
                turn_first = torch.abs(w_full[0] - prev_chunk_drifted).sum().unsqueeze(0)
            else:
                turn_first = torch.zeros(1, dtype=dw.dtype)

            turnover = torch.cat([turn_first, turn_inner], dim=0)
            tc = train_fee * turnover

            active_ret = (dw * ch["excess_ret"] * ch["valid_mask"]).sum(dim=1)
            net_ret = active_ret - tc
            net_ir = net_ret.mean() / (net_ret.std(unbiased=False) + 1e-4)
            ic = _batch_pearson_ic(score_z, ch["target"], ch["valid_target"])
            prior_pen = prior_lambda * ((theta - prior_w) ** 2).sum()
            loss = -net_ir + aux_ic_w * (-ic) + prior_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            prev_chunk_drifted = w_drifted[-1].detach()
        sched.step()
    return net


def _drift_weight_map(prev_w: dict[str, float], prev_ret: dict[str, float]) -> dict[str, float]:
    if not prev_w:
        return {}
    grown = {c: max(0.0, w * (1.0 + prev_ret.get(c, 0.0))) for c, w in prev_w.items()}
    tot = sum(grown.values())
    if tot <= 1e-12:
        return {}
    return {c: v / tot for c, v in grown.items() if v > 1e-12}


def run_all_models(
    weeks: list[WeekRecord],
    cfg: dict[str, Any],
    factor_cols: list[str] | None = None,
    seed_override: int | None = None,
    strategies_override: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run B0, S0, S0_LowTurn_Band, S1_RollingRidge, and S2_WhiteboxDFL causally over eval_start..dev_end."""
    eval_start = pd.Timestamp(cfg["dates"]["eval_start"])
    dev_end = pd.Timestamp(cfg["dates"]["dev_end"])
    p_cfg = cfg["portfolio"]
    m_cfg = cfg["models"]

    f_cols = list(FACTORS if factor_cols is None else factor_cols)
    active_strategies = list(m_cfg["evaluated_strategies"] if strategies_override is None else strategies_override)

    gamma = float(p_cfg["proj_gamma"])
    active_limit = float(p_cfg["active_limit"])
    single_cap = float(p_cfg["single_stock_cap"])
    inf_steps = int(p_cfg["inference_dykstra_steps"])
    train_steps = int(p_cfg["train_dykstra_steps"])
    eps = float(p_cfg["dykstra_eps"])
    ema_alpha = float(p_cfg.get("low_turn_ema_alpha", 0.65))

    window = int(m_cfg["rolling_window_weeks"])
    min_weeks = int(m_cfg["min_train_weeks"])
    ridge_l2 = float(m_cfg["ridge_l2"])
    dfl_refit = int(m_cfg["dfl_refit_weeks"])
    dfl_epochs = int(m_cfg["dfl_epochs"])
    dfl_lr = float(m_cfg["dfl_lr"])
    dfl_prior_lambda = float(m_cfg["dfl_prior_lambda"])
    dfl_train_fee = float(m_cfg["dfl_train_fee"])
    dfl_aux_ic = float(m_cfg["dfl_aux_ic_weight"])
    dfl_chunk = int(m_cfg["dfl_chunk_weeks"])
    dfl_seed = int(m_cfg["dfl_seed"] if seed_override is None else seed_override)
    torch.set_num_threads(int(m_cfg["torch_threads"]))

    n_fac = len(f_cols)
    eye_fac = ridge_l2 * np.eye(n_fac, dtype=np.float64)

    outputs: dict[str, list[pd.DataFrame]] = {s: [] for s in active_strategies}
    audit_rows, ridge_weights_rows, dfl_weights_rows, turn_rec_rows = [], [], [], []
    dfl_net: WhiteboxDFL | None = None
    eval_count = 0

    prev_s0_lowturn_w: dict[str, float] = {}
    prev_s0_lowturn_ret: dict[str, float] = {}
    prev_dfl_w: dict[str, float] = {}
    prev_dfl_dw: dict[str, float] = {}
    prev_dfl_ret: dict[str, float] = {}

    for idx, current in enumerate(weeks):
        if not (eval_start <= current.signal_date <= dev_end):
            continue

        train = [w for w in weeks[:idx] if w.has_mature_target and w.exit_date is not None and w.exit_date <= current.signal_date][-window:]
        if len(train) < min_weeks:
            continue
        latest_exit = max(w.exit_date for w in train if w.exit_date is not None)
        if latest_exit > current.signal_date:
            raise ValueError(f"Unmatured label leaked into training on {current.signal_date.date()}")

        # --- B0: Pure FloatMV Anchor ---
        if "B0_FloatMV_Anchor" in outputs:
            z0 = np.zeros(len(current.codes), dtype=np.float32)
            a0 = current.anchor.astype(np.float64)
            outputs["B0_FloatMV_Anchor"].append(_week_df(current, z0, np.zeros_like(a0), a0))

        # --- S0: Equal Weight (1/F each) ---
        s0 = zscore_np(current.x.mean(axis=1))
        dw0, w0 = project_portfolio_numpy(
            s0, current.anchor, current.industry_codes, current.beta, current.size_z,
            gamma=gamma, active_limit=active_limit, single_cap=single_cap, num_steps=inf_steps, eps=eps,
        )
        if "S0_EqualWeight" in outputs:
            outputs["S0_EqualWeight"].append(_week_df(current, s0, dw0, w0))

        # --- S0_LowTurn_Band: Equal Weight + Rule-Based Weight Smoothing (Zero Learned Parameters) ---
        if "S0_LowTurn_Band" in outputs:
            if not prev_s0_lowturn_w:
                w_lt = w0.copy()
            else:
                drifted = _drift_weight_map(prev_s0_lowturn_w, prev_s0_lowturn_ret)
                prev_vec = np.array([drifted.get(c, 0.0) for c in current.codes], dtype=np.float64)
                prev_sum = prev_vec.sum()
                if prev_sum > 1e-6:
                    prev_vec = prev_vec / prev_sum
                    w_lt = ema_alpha * w0 + (1.0 - ema_alpha) * prev_vec
                else:
                    w_lt = w0.copy()
                w_lt = np.clip(w_lt, 0.0, np.maximum(single_cap, current.anchor.astype(np.float64)))
                w_lt = w_lt / w_lt.sum()
            dw_lt = w_lt - current.anchor.astype(np.float64)
            outputs["S0_LowTurn_Band"].append(_week_df(current, s0, dw_lt, w_lt))
            prev_s0_lowturn_w = dict(zip(current.codes, w_lt.tolist()))
            prev_s0_lowturn_ret = dict(zip(current.codes, current.stock_ret.tolist()))

        # --- S1: Rolling Ridge ---
        if "S1_RollingRidge" in outputs:
            gram = np.mean([w.xx for w in train], axis=0) + eye_fac
            rhs = np.mean([w.xy for w in train], axis=0)
            raw_coef = np.linalg.solve(gram, rhs)
            pos_coef = np.maximum(raw_coef, 0.0)
            ridge_w = pos_coef / pos_coef.sum() if pos_coef.sum() > 1e-12 else np.full(n_fac, 1.0 / n_fac)
            s1 = zscore_np(current.x @ ridge_w.astype(np.float32))
            dw1, w1 = project_portfolio_numpy(
                s1, current.anchor, current.industry_codes, current.beta, current.size_z,
                gamma=gamma, active_limit=active_limit, single_cap=single_cap, num_steps=inf_steps, eps=eps,
            )
            outputs["S1_RollingRidge"].append(_week_df(current, s1, dw1, w1))
            ridge_weights_rows.append({"signal_date": current.signal_date, **dict(zip(f_cols, ridge_w))})

        # --- S2: Whitebox DFL ---
        if "S2_WhiteboxDFL" in outputs:
            if dfl_net is None or eval_count % dfl_refit == 0:
                chunks = _split_chunks(train, dfl_chunk)
                dfl_net = _train_whitebox_dfl(
                    chunks, n_fac, gamma, train_steps, eps,
                    dfl_epochs, dfl_lr, dfl_prior_lambda, dfl_train_fee, dfl_aux_ic, dfl_seed,
                )

            with torch.no_grad():
                dfl_net.eval()
                dw2_t, s2_t, theta_t = dfl_net.forward_portfolio(
                    torch.from_numpy(current.x),
                    torch.from_numpy(current.A),
                    torch.from_numpy(current.lo),
                    torch.from_numpy(current.hi),
                    B_eq=torch.from_numpy(current.B_eq),
                    num_steps=inf_steps,
                )
                dw2 = dw2_t.numpy().astype(np.float64)
                w2 = np.clip(current.anchor.astype(np.float64) + dw2, 0.0, np.maximum(single_cap, current.anchor))
                w2 = w2 / w2.sum()
                dw2 = w2 - current.anchor.astype(np.float64)
                s2 = s2_t.numpy().astype(np.float32)
                dfl_w = theta_t.numpy()

            outputs["S2_WhiteboxDFL"].append(_week_df(current, s2, dw2, w2))
            dfl_weights_rows.append({"signal_date": current.signal_date, **dict(zip(f_cols, dfl_w))})

            # Turnover reconciliation diagnostics for DFL
            cur_w_map = dict(zip(current.codes, w2.tolist()))
            cur_dw_map = dict(zip(current.codes, dw2.tolist()))
            if prev_dfl_w:
                drifted_dfl = _drift_weight_map(prev_dfl_w, prev_dfl_ret)
                all_c = set(cur_w_map) | set(drifted_dfl)
                full_drifted_turn = sum(abs(cur_w_map.get(c, 0.0) - drifted_dfl.get(c, 0.0)) for c in all_c)
                all_c_raw = set(cur_w_map) | set(prev_dfl_w)
                target_undrifted_turn = sum(abs(cur_w_map.get(c, 0.0) - prev_dfl_w.get(c, 0.0)) for c in all_c_raw)
                all_c_dw = set(cur_dw_map) | set(prev_dfl_dw)
                proxy_dw_turn = sum(abs(cur_dw_map.get(c, 0.0) - prev_dfl_dw.get(c, 0.0)) for c in all_c_dw)
                turn_rec_rows.append({
                    "signal_date": current.signal_date,
                    "entry_date": current.entry_date,
                    "proxy_active_delta_turn": proxy_dw_turn,
                    "target_undrifted_turn": target_undrifted_turn,
                    "full_drifted_turn": full_drifted_turn,
                })
            prev_dfl_w = cur_w_map
            prev_dfl_dw = cur_dw_map
            prev_dfl_ret = dict(zip(current.codes, current.stock_ret.tolist()))

        audit_rows.append({
            "signal_date": current.signal_date,
            "train_weeks": len(train),
            "oldest_train_signal": train[0].signal_date,
            "latest_train_exit": latest_exit,
        })
        eval_count += 1
        if verbose and eval_count % 100 == 0:
            print(f"[models] Predicted {eval_count} weeks through {current.signal_date.date()}", flush=True)

    preds = {name: pd.concat(frames, ignore_index=True) for name, frames in outputs.items() if frames}
    return (
        preds,
        pd.DataFrame(audit_rows),
        pd.DataFrame(ridge_weights_rows),
        pd.DataFrame(dfl_weights_rows),
        pd.DataFrame(turn_rec_rows),
    )


def _week_df(w: WeekRecord, score: np.ndarray, dw: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": w.signal_date,
        "entry_date": w.entry_date,
        "exit_date": w.exit_date if w.exit_date is not None else pd.NaT,
        "code": w.codes,
        "industry": w.industry_codes,
        "anchor_weight": w.anchor,
        "active_weight": dw.astype(np.float32),
        "target_weight": weight.astype(np.float32),
        "score": score.astype(np.float32),
        "beta": w.beta,
        "size_z": w.size_z,
        "stock_return_5d": np.where(w.valid_target, w.excess_ret, np.nan),
        "risk_adjusted_target": np.where(w.valid_target, w.target, np.nan),
    })
