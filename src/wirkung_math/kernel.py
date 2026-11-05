from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def rbf_rows(points: np.ndarray, reference: np.ndarray, gamma: float) -> np.ndarray:
    points=np.asarray(points,dtype=np.float64)
    reference=np.asarray(reference,dtype=np.float64)
    p2=np.sum(points*points,axis=1)[:,None]
    r2=np.sum(reference*reference,axis=1)[None,:]
    dist=np.maximum(p2+r2-2.0*(points@reference.T),0.0)
    return np.exp(-gamma*dist)


def rbf_kernel(x: np.ndarray, gamma: float) -> np.ndarray:
    return rbf_rows(x,x,gamma)


@dataclass(frozen=True)
class RowColumnDelta:
    """
    Symmetric matrix change supported only on selected rows/columns.

    row_delta = ΔA[indices,:]. The full ΔA has rank at most 2e.
    """
    indices: np.ndarray
    row_delta: np.ndarray

    def __post_init__(self):
        idx=np.asarray(self.indices,dtype=np.int64)
        R=np.asarray(self.row_delta,dtype=np.float64)
        if idx.ndim != 1 or R.ndim != 2 or R.shape[0] != len(idx):
            raise ValueError("shape mismatch")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("indices must be unique")
        block=R[:,idx]
        if not np.allclose(block,block.T,rtol=1e-10,atol=1e-10):
            raise ValueError("changed principal block must be symmetric")
        object.__setattr__(self,"indices",idx)
        object.__setattr__(self,"row_delta",R)

    @property
    def rank_bound(self) -> int:
        return 2*len(self.indices)

    def materialize(self) -> np.ndarray:
        n=self.row_delta.shape[1]; e=len(self.indices)
        S=np.zeros((n,e),dtype=np.float64)
        S[self.indices,np.arange(e)]=1.0
        B=self.row_delta[:,self.indices]
        return S@self.row_delta + self.row_delta.T@S.T - S@B@S.T

    def prepare_woodbury(self, A_inv: np.ndarray) -> "RowColumnWoodbury":
        return RowColumnWoodbury.prepare(A_inv,self)


@dataclass
class RowColumnWoodbury:
    """
    Woodbury cache for RowColumnDelta without materializing the n×e selector S.
    """
    A_inv: np.ndarray
    delta: RowColumnDelta
    AF: np.ndarray
    C: np.ndarray
    middle: np.ndarray

    @classmethod
    def prepare(cls, A_inv: np.ndarray, delta: RowColumnDelta) -> "RowColumnWoodbury":
        A_inv=np.asarray(A_inv,dtype=np.float64)
        idx=delta.indices; R=delta.row_delta; e=len(idx)
        AS=A_inv[:,idx]
        AR=A_inv @ R.T
        AF=np.concatenate([AS,AR],axis=1)
        top_left=A_inv[np.ix_(idx,idx)]
        top_right=AR[idx,:]
        gram=np.block([[top_left,top_right],[top_right.T,R@AR]])
        B=R[:,idx]
        C=np.block([[-B,np.eye(e)],[np.eye(e),np.zeros((e,e))]])
        middle=np.eye(2*e) + C@gram
        return cls(A_inv,delta,AF,C,middle)

    def _ft(self, z: np.ndarray) -> np.ndarray:
        return np.concatenate([z[self.delta.indices,:], self.delta.row_delta@z],axis=0)

    def apply_inverse(self, rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim == 1
        B=rhs[:,None] if vector else rhs
        z=self.A_inv @ B
        corr=self.AF @ np.linalg.solve(self.middle,self.C@self._ft(z))
        out=z-corr
        return out[:,0] if vector else out

    def logdet_update(self, base_sign: float, base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.middle)
        return float(base_sign*sign),float(base_logabsdet+logabs)


def rbf_replacement_delta(
    x: np.ndarray, indices: np.ndarray, new_points: np.ndarray, gamma: float
) -> RowColumnDelta:
    """Exact row/column delta from replacing a small set of points."""
    x=np.asarray(x,dtype=np.float64)
    idx=np.asarray(indices,dtype=np.int64)
    new=np.asarray(new_points,dtype=np.float64)
    old_rows=rbf_rows(x[idx],x,gamma)
    new_rows=rbf_rows(new,x,gamma)
    new_rows[:,idx]=rbf_kernel(new,gamma)
    return RowColumnDelta(idx,new_rows-old_rows)


def rbf_single_point_sparse_feature_update(
    old_sqdist: np.ndarray,
    x_features: np.ndarray,
    old_point_features: np.ndarray,
    feature_delta: np.ndarray,
    point_index: int,
    gamma: float,
) -> np.ndarray:
    """
    Update one full RBF row in O(n*s) when only s coordinates of the point changed.

    x_features contains only those s feature columns for every reference point.
    old_sqdist is the cached full squared-distance row before the edit.
    """
    old_sqdist=np.asarray(old_sqdist,dtype=np.float64)
    xf=np.asarray(x_features,dtype=np.float64)
    p=np.asarray(old_point_features,dtype=np.float64)
    delta=np.asarray(feature_delta,dtype=np.float64)
    if xf.shape[1] != len(p) or p.shape != delta.shape:
        raise ValueError("feature shape mismatch")
    newdist=old_sqdist + 2.0*((p[None,:]-xf)@delta) + float(delta@delta)
    newdist=np.maximum(newdist,0.0)
    newdist[int(point_index)]=0.0
    return np.exp(-gamma*newdist)


@dataclass
class RowColumnSolverWoodbury:
    """Row/column Woodbury over an arbitrary cached symmetric base solver."""
    solve_base: object
    delta: RowColumnDelta
    AF: np.ndarray
    C: np.ndarray
    middle: np.ndarray

    @classmethod
    def prepare(cls,solve_base,delta: RowColumnDelta) -> "RowColumnSolverWoodbury":
        idx=delta.indices; R=delta.row_delta; e=len(idx); n=R.shape[1]
        S=np.zeros((n,e),dtype=np.float64)
        S[idx,np.arange(e)]=1.0
        AS=np.asarray(solve_base(S),dtype=np.float64)
        AR=np.asarray(solve_base(R.T),dtype=np.float64)
        AF=np.concatenate([AS,AR],axis=1)
        top_left=AS[idx,:]
        top_right=AR[idx,:]
        bottom_left=R@AS
        bottom_right=R@AR
        gram=np.block([[top_left,top_right],[bottom_left,bottom_right]])
        B=R[:,idx]
        C=np.block([[-B,np.eye(e)],[np.eye(e),np.zeros((e,e))]])
        middle=np.eye(2*e)+C@gram
        return cls(solve_base,delta,AF,C,middle)

    def _ft(self,z: np.ndarray) -> np.ndarray:
        return np.concatenate([z[self.delta.indices,:],self.delta.row_delta@z],axis=0)

    def apply_inverse(self,rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim==1
        B=rhs[:,None] if vector else rhs
        z=np.asarray(self.solve_base(B),dtype=np.float64)
        out=z-self.AF@np.linalg.solve(self.middle,self.C@self._ft(z))
        return out[:,0] if vector else out

    def logdet_update(self,base_sign: float,base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.middle)
        return float(base_sign*sign),float(base_logabsdet+logabs)
