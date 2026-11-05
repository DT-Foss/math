from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EncodedTemporalDelta:
    coefficients:np.ndarray
    residual_indices:np.ndarray
    residual_values:np.ndarray
    captured_energy:float


@dataclass(frozen=True)
class TemporalDeltaBasis:
    """
    Layer-specific basis learned from token-to-token deltas, not full activations.

    The residual is retained exactly. Acceleration is used only when the residual
    itself is sparse/cheap; otherwise the planner routes to another exact form.
    """
    basis:np.ndarray
    singular_values:np.ndarray

    @classmethod
    def fit(cls,deltas:np.ndarray,rank:int)->"TemporalDeltaBasis":
        D=np.asarray(deltas,dtype=np.float64)
        if D.ndim!=2 or not 0<rank<=min(D.shape):
            raise ValueError("invalid delta matrix/rank")
        _,s,Vh=np.linalg.svd(D,full_matrices=False)
        B=Vh[:rank].T
        return cls(B,s[:rank])

    @property
    def width(self)->int:
        return int(self.basis.shape[0])

    @property
    def rank(self)->int:
        return int(self.basis.shape[1])

    def encode(self,delta:np.ndarray,*,residual_atol:float=0.0)->EncodedTemporalDelta:
        x=np.asarray(delta,dtype=np.float64)
        if x.shape!=(self.width,):
            raise ValueError("delta shape mismatch")
        c=self.basis.T@x
        residual=x-self.basis@c
        idx=np.flatnonzero(np.abs(residual)>residual_atol)
        total=float(np.dot(x,x))
        captured=float(np.dot(c,c))
        return EncodedTemporalDelta(
            c,idx,residual[idx],
            1.0 if total==0.0 else min(1.0,captured/total),
        )

    def batch_encode(self,deltas:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        D=np.asarray(deltas,dtype=np.float64)
        if D.ndim!=2 or D.shape[1]!=self.width:
            raise ValueError("delta matrix shape mismatch")
        coefficients=D@self.basis
        residual=D-coefficients@self.basis.T
        return coefficients,residual


def exact_batch_factorization(
    delta_matrix:np.ndarray,*,rtol:float=0.0
)->tuple[np.ndarray,np.ndarray]:
    """Exact numerical low-rank factors delta_X = A B^T for lane/speculative batches."""
    D=np.asarray(delta_matrix,dtype=np.float64)
    if D.ndim!=2:
        raise ValueError("delta_matrix must be two-dimensional")
    U,s,Vh=np.linalg.svd(D,full_matrices=False)
    if len(s)==0:
        return np.empty((D.shape[0],0)),np.empty((D.shape[1],0))
    threshold=(rtol*s[0] if rtol>0 else np.finfo(np.float64).eps*max(D.shape)*s[0])
    rank=int(np.count_nonzero(s>threshold))
    root=np.sqrt(s[:rank])
    A=U[:,:rank]*root[None,:]
    B=Vh[:rank].T*root[None,:]
    return A,B
