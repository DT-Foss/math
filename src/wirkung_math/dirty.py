from __future__ import annotations
import numpy as np


def compose_stencils(*kernels: np.ndarray) -> np.ndarray:
    """Compose linear 1-D local operators into one equivalent stencil."""
    if not kernels:
        return np.array([1.0])
    out = np.asarray(kernels[0], dtype=float)
    for k in kernels[1:]:
        out = np.convolve(out, np.asarray(k, dtype=float))
    return out


def dirty_cone(indices: np.ndarray, length: int, radius: int, layers: int = 1) -> np.ndarray:
    """Exact dependency cone for a stack of radius-r local operators."""
    reach = radius * layers
    mask = np.zeros(length, dtype=bool)
    for i in np.asarray(indices, dtype=int):
        lo, hi = max(0, i - reach), min(length, i + reach + 1)
        mask[lo:hi] = True
    return np.flatnonzero(mask)
