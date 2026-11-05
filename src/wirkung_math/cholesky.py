from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def cholupdate_lower(L: np.ndarray, x: np.ndarray, sign: int=1) -> np.ndarray:
    """Rank-1 update/downdate of lower Cholesky L for A + sign*x*x^T."""
    if sign not in (-1,1):
        raise ValueError("sign must be +1 or -1")
    L=np.asarray(L,dtype=np.float64).copy()
    x=np.asarray(x,dtype=np.float64).copy()
    n=L.shape[0]
    if L.shape != (n,n) or x.shape != (n,):
        raise ValueError("shape mismatch")
    for k in range(n):
        lkk=L[k,k]
        value=lkk*lkk + sign*x[k]*x[k]
        if value <= 0:
            raise np.linalg.LinAlgError("downdate would destroy positive definiteness")
        r=math.sqrt(value)
        c=r/lkk
        s=x[k]/lkk
        L[k,k]=r
        if k+1<n:
            col=(L[k+1:,k] + sign*s*x[k+1:]) / c
            L[k+1:,k]=col
            x[k+1:]=c*x[k+1:] - s*col
    return L


@dataclass
class CholeskyState:
    """SPD matrix state supporting row-replacement Gram updates without refactorization."""
    L: np.ndarray

    @classmethod
    def from_matrix(cls, A: np.ndarray) -> "CholeskyState":
        return cls(np.linalg.cholesky(np.asarray(A,dtype=np.float64)))

    def update_vector(self, x: np.ndarray) -> None:
        self.L=cholupdate_lower(self.L,x,+1)

    def downdate_vector(self, x: np.ndarray) -> None:
        self.L=cholupdate_lower(self.L,x,-1)

    def replace_gram_rows(self, old_rows: np.ndarray, new_rows: np.ndarray) -> None:
        old=np.asarray(old_rows,dtype=np.float64)
        new=np.asarray(new_rows,dtype=np.float64)
        if old.shape != new.shape or old.shape[1] != self.L.shape[0]:
            raise ValueError("shape mismatch")
        for row in old:
            self.downdate_vector(row)
        for row in new:
            self.update_vector(row)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        y=np.linalg.solve(self.L,rhs)
        return np.linalg.solve(self.L.T,y)

    @property
    def logdet(self) -> float:
        return float(2.0*np.log(np.diag(self.L)).sum())

    def matrix(self) -> np.ndarray:
        return self.L@self.L.T
