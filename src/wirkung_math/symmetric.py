from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SymmetricLowRankDelta:
    """Represents ΔA = U C U^T with a small symmetric core C."""
    U: np.ndarray
    C: np.ndarray

    def __post_init__(self):
        if self.U.ndim != 2 or self.C.ndim != 2:
            raise ValueError("U and C must be matrices")
        if self.C.shape != (self.U.shape[1],self.U.shape[1]):
            raise ValueError("core shape mismatch")
        if not np.allclose(self.C,self.C.T,rtol=1e-12,atol=1e-12):
            raise ValueError("C must be symmetric")

    @property
    def rank_bound(self) -> int:
        return int(self.U.shape[1])

    def materialize(self) -> np.ndarray:
        return self.U @ self.C @ self.U.T

    @classmethod
    def from_gram_replacement(cls, old_rows: np.ndarray, new_rows: np.ndarray) -> "SymmetricLowRankDelta":
        old=np.asarray(old_rows,dtype=np.float64)
        new=np.asarray(new_rows,dtype=np.float64)
        if old.shape != new.shape:
            raise ValueError("shape mismatch")
        e=old.shape[0]
        U=np.concatenate([new.T,old.T],axis=1)
        C=np.diag(np.concatenate([np.ones(e),-np.ones(e)]))
        return cls(U,C)

    def compact(self, rtol: float=1e-12) -> "SymmetricLowRankDelta":
        """Rank reveal via QR + eigendecomposition of only the small core."""
        if self.rank_bound == 0:
            return self
        q,r=np.linalg.qr(self.U,mode="reduced")
        core=r @ self.C @ r.T
        core=(core+core.T)*0.5
        w,v=np.linalg.eigh(core)
        scale=float(np.max(np.abs(w))) if len(w) else 0.0
        if scale == 0:
            return SymmetricLowRankDelta(self.U[:,:0],np.zeros((0,0)))
        keep=np.abs(w) > scale*rtol
        root=np.sqrt(np.abs(w[keep]))
        U2=(q @ v[:,keep]) * root[None,:]
        C2=np.diag(np.sign(w[keep]))
        return SymmetricLowRankDelta(U2,C2)


@dataclass
class WoodburyCache:
    """
    Shared low-rank update state for solve, inverse action and logdet.

    Uses I + C U^T A^-1 U, never C^-1, so singular/indefinite small cores
    remain representable.
    """
    A_inv: np.ndarray
    delta: SymmetricLowRankDelta
    AU: np.ndarray
    middle: np.ndarray

    @classmethod
    def prepare(cls, A_inv: np.ndarray, delta: SymmetricLowRankDelta) -> "WoodburyCache":
        A_inv=np.asarray(A_inv,dtype=np.float64)
        U=delta.U; C=delta.C
        AU=A_inv @ U
        middle=np.eye(delta.rank_bound) + C @ (U.T @ AU)
        return cls(A_inv,delta,AU,middle)

    def apply_inverse(self, rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim == 1
        B=rhs[:,None] if vector else rhs
        z=self.A_inv @ B
        small=self.delta.C @ (self.delta.U.T @ z)
        corr=self.AU @ np.linalg.solve(self.middle,small)
        out=z-corr
        return out[:,0] if vector else out

    def materialize_inverse(self) -> np.ndarray:
        return self.apply_inverse(np.eye(self.A_inv.shape[0]))

    def logdet_update(self, base_sign: float, base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.middle)
        return float(base_sign*sign), float(base_logabsdet+logabs)

    def quadratic_form(self, rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim == 1
        B=rhs[:,None] if vector else rhs
        solved=self.apply_inverse(B)
        q=np.sum(B*solved,axis=0)
        return q[0] if vector else q
