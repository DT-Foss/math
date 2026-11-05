from __future__ import annotations
import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def numerical_rank(x: np.ndarray, rtol: float = 1e-10) -> int:
    s = np.linalg.svd(x, compute_uv=False)
    if not len(s) or s[0] == 0:
        return 0
    return int(np.sum(s > s[0] * rtol))


def top_rank_energy(x: np.ndarray, rank: int) -> float:
    s = np.linalg.svd(x, compute_uv=False)
    e = s * s
    if e.sum() == 0:
        return 1.0
    return float(e[:rank].sum() / e.sum())


def rank_barrier(X: np.ndarray, A: np.ndarray, B: np.ndarray, fn=silu) -> dict:
    delta = A @ B.T
    out_delta = fn(X + delta) - fn(X)
    r = A.shape[1]
    return {
        "input_rank": int(r),
        "output_numerical_rank": numerical_rank(out_delta),
        "energy_in_input_rank": top_rank_energy(out_delta, r),
        "energy_in_2x_rank": top_rank_energy(out_delta, min(2*r, min(out_delta.shape))),
    }
