from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def full_normalized(scores: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable row-wise softmax-weighted value sum and log partition."""
    m = np.max(scores, axis=1, keepdims=True)
    ex = np.exp(scores - m)
    z = ex.sum(axis=1, keepdims=True)
    weights = ex / z
    out = weights @ values
    logz = np.log(z[:, 0]) + m[:, 0]
    return out, logz


@dataclass
class NormalizedState:
    scores: np.ndarray
    values: np.ndarray
    output: np.ndarray
    logz: np.ndarray

    @classmethod
    def from_full(cls, scores: np.ndarray, values: np.ndarray) -> "NormalizedState":
        out, logz = full_normalized(scores, values)
        return cls(scores.copy(), values.copy(), out, logz)


def topk_miss_probability(n: int, edits: int, k: int) -> float:
    """Probability that all k cached candidates are edited, sampling without replacement."""
    if k <= 0:
        return 1.0
    if edits < k:
        return 0.0
    p = 1.0
    for j in range(k):
        p *= (edits - j) / (n - j)
    return p


def minimum_topk(n: int, edits: int, expected_fallback_rows: float = 1e-3) -> int:
    """Smallest k with n*P(all top-k edited) <= expected_fallback_rows."""
    for k in range(1, min(n, edits) + 2):
        if n * topk_miss_probability(n, edits, k) <= expected_fallback_rows:
            return k
    return min(n, edits + 1)


def mass_gated_update(
    state: NormalizedState,
    columns: np.ndarray,
    new_scores_u: np.ndarray,
    new_values_u: np.ndarray | None = None,
    cancellation_floor: float = 2.0 ** -12,
) -> dict:
    """
    Exact algebraic update on well-conditioned rows; recompute only cancellation rows.

    Let O be the old normalized weighted sum and L the old log partition. For edited U,
      a = exp(S_U-L), b = exp(S'_U-L), q = 1-sum(a)+sum(b)
      O' = (O-aV_U+bV'_U)/q, L'=L+log(q).
    Rows with tiny q are rebased by a full stable row recomputation.
    """
    cols = np.asarray(columns, dtype=np.int64)
    if new_scores_u.shape != (state.scores.shape[0], len(cols)):
        raise ValueError("new_scores_u shape mismatch")
    old_scores_u = state.scores[:, cols].copy()
    old_values_u = state.values[cols].copy()
    if new_values_u is None:
        new_values_u = old_values_u
    new_values_u = np.asarray(new_values_u)
    if new_values_u.shape != old_values_u.shape:
        raise ValueError("new_values_u shape mismatch")

    a = np.exp(old_scores_u - state.logz[:, None])
    b = np.exp(new_scores_u - state.logz[:, None])
    q = 1.0 - a.sum(axis=1) + b.sum(axis=1)
    fast = np.isfinite(q) & (q > cancellation_floor)

    old_contrib = a @ old_values_u
    new_contrib = b @ new_values_u
    candidate = (state.output - old_contrib + new_contrib) / q[:, None]
    new_logz = state.logz + np.log(np.where(fast, q, 1.0))

    state.scores[:, cols] = new_scores_u
    state.values[cols] = new_values_u
    state.output[fast] = candidate[fast]
    state.logz[fast] = new_logz[fast]

    fallback_rows = np.flatnonzero(~fast)
    if len(fallback_rows):
        s = state.scores[fallback_rows]
        m = np.max(s, axis=1, keepdims=True)
        ex = np.exp(s - m)
        z = ex.sum(axis=1, keepdims=True)
        state.output[fallback_rows] = (ex / z) @ state.values
        state.logz[fallback_rows] = np.log(z[:, 0]) + m[:, 0]

    return {
        "fast_rows": int(fast.sum()),
        "fallback_rows": int((~fast).sum()),
        "min_q": float(np.min(q)),
    }
