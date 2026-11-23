"""Pure-PyTorch and NumPy Differentiable Index-Enhancement Projection Layer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_constraint_matrix_numpy(
    industry_codes: np.ndarray,
    beta: np.ndarray,
    size_z: np.ndarray,
    min_industry_size: int = 2,
) -> np.ndarray:
    """Build row-normalized equality constraint matrix A (M x N) for A @ dw = 0."""
    n = len(beta)
    rows: list[np.ndarray] = [np.ones(n, dtype=np.float64)]

    unique_inds, counts = np.unique(industry_codes, return_counts=True)
    eligible = [(ind, int(cnt)) for ind, cnt in zip(unique_inds, counts) if cnt >= min_industry_size]
    if len(eligible) > 1:
        total_eligible = sum(cnt for _, cnt in eligible)
        if total_eligible == n:
            eligible.sort(key=lambda pair: (-pair[1], str(pair[0])))
            eligible = eligible[1:]
        for ind, _ in eligible:
            rows.append((industry_codes == ind).astype(np.float64))

    beta_clean = np.nan_to_num(beta.astype(np.float64), nan=1.0)
    size_clean = np.nan_to_num(size_z.astype(np.float64), nan=0.0)
    rows.append(beta_clean - beta_clean.mean())
    rows.append(size_clean - size_clean.mean())

    mat = np.stack(rows, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    mat = mat[valid] / norms[valid]
    return mat.astype(np.float32)


def precompute_eq_operator_numpy(A: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Precompute B_eq = (A @ A.T + eps * I)^{-1} @ A in float64."""
    A64 = np.asarray(A, dtype=np.float64)
    m = A64.shape[0]
    gram = A64 @ A64.T + eps * np.eye(m, dtype=np.float64)
    return np.linalg.solve(gram, A64).astype(np.float32)


def compute_active_bounds_numpy(
    anchor: np.ndarray,
    active_limit: float = 0.015,
    single_cap: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute box bounds [lo, hi] for active weights dw so 0 is always feasible."""
    effective_cap = np.maximum(single_cap, anchor)
    lo = np.maximum(-anchor, -active_limit)
    hi = np.minimum(effective_cap - anchor, active_limit)
    hi = np.maximum(hi, 0.0)
    lo = np.minimum(lo, 0.0)
    return lo.astype(np.float32), hi.astype(np.float32)


class DifferentiableIEProjection(nn.Module):
    """Unrolled differentiable Dykstra projection onto {A @ dw = 0} intersect [lo, hi]."""

    def __init__(self, num_steps: int = 30, eps: float = 1e-7) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.eps = eps

    def forward(
        self,
        dw0: torch.Tensor,
        A: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        B_eq: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        steps = self.num_steps if num_steps is None else num_steps
        batched = dw0.dim() == 2

        if B_eq is None:
            m = A.shape[-2]
            eye = torch.eye(m, dtype=A.dtype, device=A.device)
            if batched:
                gram = torch.bmm(A, A.transpose(-1, -2)) + self.eps * eye.unsqueeze(0)
                B_eq = torch.linalg.solve(gram, A)
            else:
                gram = A @ A.T + self.eps * eye
                B_eq = torch.linalg.solve(gram, A)

        if batched:
            AT = A.transpose(-1, -2)

            def proj_eq(v: torch.Tensor) -> torch.Tensor:
                bv = torch.bmm(B_eq, v.unsqueeze(-1))
                return v - torch.bmm(AT, bv).squeeze(-1)
        else:
            def proj_eq(v: torch.Tensor) -> torch.Tensor:
                return v - A.T @ (B_eq @ v)

        x = dw0
        p = torch.zeros_like(dw0)
        q = torch.zeros_like(dw0)

        for _ in range(steps):
            y = torch.clamp(x + p, min=lo, max=hi)
            p = x + p - y
            x = proj_eq(y + q)
            q = y + q - x

        return torch.clamp(x + p, min=lo, max=hi)


def project_portfolio_numpy(
    score: np.ndarray,
    anchor: np.ndarray,
    industry_codes: np.ndarray,
    beta: np.ndarray,
    size_z: np.ndarray,
    gamma: float = 0.005,
    active_limit: float = 0.015,
    single_cap: float = 0.04,
    num_steps: int = 30,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy Dykstra projection sharing the exact feasible set."""
    a = np.asarray(anchor, dtype=np.float64)
    a = a / a.sum()
    A = build_constraint_matrix_numpy(industry_codes, beta, size_z).astype(np.float64)
    lo, hi = compute_active_bounds_numpy(a, active_limit=active_limit, single_cap=single_cap)
    lo64, hi64 = lo.astype(np.float64), hi.astype(np.float64)

    dw0 = gamma * np.nan_to_num(score.astype(np.float64), nan=0.0)
    m = A.shape[0]
    gram = A @ A.T + eps * np.eye(m, dtype=np.float64)
    B_eq = np.linalg.solve(gram, A)

    x = dw0.copy()
    p = np.zeros_like(x)
    q = np.zeros_like(x)

    for _ in range(num_steps):
        y = np.clip(x + p, lo64, hi64)
        p = x + p - y
        v = y + q
        x = v - A.T @ (B_eq @ v)
        q = v - x

    dw = np.clip(x + p, lo64, hi64)
    w = a + dw
    w = np.clip(w, 0.0, np.maximum(single_cap, a))
    w = w / w.sum()
    return w - a, w
