from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def crossover_rank(edits: int, width: int) -> float:
    """Rank below which sparse×low-rank uses fewer multiplies than a direct sparse delta."""
    e, d = float(edits), float(width)
    return (e * d) / (e + d)


def effective_edits(edits: int, width: int, rank: int) -> float:
    """Equivalent direct-sparse edit count for n*r*(e+d) work."""
    return rank * (edits + width) / width


def sparse_lowrank_update(E_u: np.ndarray, A: np.ndarray, B: np.ndarray) -> "FactorDelta":
    """Exact E[:,U] @ (A @ B.T), returned without materializing the dense result."""
    if E_u.shape[1] != A.shape[0]:
        raise ValueError("E_u columns must match A rows")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same rank")
    C = E_u @ A
    return FactorDelta(C, B.copy())


@dataclass(frozen=True)
class FactorDelta:
    """Represents ΔX = left @ right.T without dense materialization."""
    left: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("factors must be matrices")
        if self.left.shape[1] != self.right.shape[1]:
            raise ValueError("factor ranks must match")

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.left.shape[0]), int(self.right.shape[0]))

    def materialize(self) -> np.ndarray:
        return self.left @ self.right.T

    def materialize_block(self, rows, cols) -> np.ndarray:
        """Materialize only a requested output block."""
        return self.left[rows] @ self.right[cols].T

    def compose_right(self, W: np.ndarray) -> "FactorDelta":
        """Propagate ΔX through X -> XW exactly: Δ(XW)=left @ (W.T@right).T."""
        if W.shape[0] != self.right.shape[0]:
            raise ValueError("W input width must match delta width")
        return FactorDelta(self.left, W.T @ self.right)

    def scaled_rows(self, scale: np.ndarray) -> "FactorDelta":
        if scale.ndim != 1 or scale.shape[0] != self.left.shape[0]:
            raise ValueError("row scale length mismatch")
        return FactorDelta(self.left * scale[:, None], self.right)

    def add(self, other: "FactorDelta") -> "FactorDelta":
        """Exact factor concatenation: ABᵀ + CDᵀ = [A C][B D]ᵀ."""
        if self.shape != other.shape:
            raise ValueError("shape mismatch")
        return FactorDelta(
            np.concatenate([self.left, other.left], axis=1),
            np.concatenate([self.right, other.right], axis=1),
        )

    def compact(self, rtol: float = 1e-12) -> "FactorDelta":
        """Rank-reveal and recompress factors without materializing the large matrix."""
        if self.rank == 0:
            return self
        ql, rl = np.linalg.qr(self.left, mode="reduced")
        qr, rr = np.linalg.qr(self.right, mode="reduced")
        core = rl @ rr.T
        u, s, vt = np.linalg.svd(core, full_matrices=False)
        if not len(s) or s[0] == 0:
            return FactorDelta(self.left[:, :0], self.right[:, :0])
        keep = s > s[0] * rtol
        root = np.sqrt(s[keep])
        left = (ql @ u[:, keep]) * root[None, :]
        right = (qr @ vt[keep, :].T) * root[None, :]
        return FactorDelta(left, right)
