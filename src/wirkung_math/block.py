from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BlockAppendCache:
    """
    Append a k×k block to A using the Schur complement.

    A_new = [[A, B],
             [B^T, D]]
    """
    A_inv: np.ndarray
    B: np.ndarray
    D: np.ndarray
    AinvB: np.ndarray
    schur: np.ndarray

    @classmethod
    def prepare(cls, A_inv: np.ndarray, B: np.ndarray, D: np.ndarray) -> "BlockAppendCache":
        A_inv=np.asarray(A_inv,dtype=np.float64)
        B=np.asarray(B,dtype=np.float64)
        D=np.asarray(D,dtype=np.float64)
        AinvB=A_inv@B
        schur=D-B.T@AinvB
        return cls(A_inv,B,D,AinvB,schur)

    def apply_inverse(self, rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim==1
        R=rhs[:,None] if vector else rhs
        n=self.A_inv.shape[0]
        top=R[:n]; bottom=R[n:]
        z=self.A_inv@top
        y=np.linalg.solve(self.schur,bottom-self.B.T@z)
        x=z-self.AinvB@y
        out=np.vstack([x,y])
        return out[:,0] if vector else out

    def materialize_inverse(self) -> np.ndarray:
        n=self.A_inv.shape[0]+self.D.shape[0]
        return self.apply_inverse(np.eye(n))

    def logdet_update(self, base_sign: float, base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.schur)
        return float(base_sign*sign),float(base_logabsdet+logabs)


def delete_trailing_block_from_inverse(full_inv: np.ndarray, k: int) -> np.ndarray:
    """Recover inverse of the retained leading principal block after deleting k trailing rows/cols."""
    full=np.asarray(full_inv,dtype=np.float64)
    if k<=0 or k>=full.shape[0]:
        raise ValueError("invalid k")
    P=full[:-k,:-k]
    Q=full[:-k,-k:]
    R=full[-k:,-k:]
    return P-Q@np.linalg.solve(R,Q.T)
