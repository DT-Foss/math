from __future__ import annotations
import numpy as np
from .symmetric import SymmetricLowRankDelta


def midpoint_difference(old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    old=np.asarray(old,dtype=np.float64)
    new=np.asarray(new,dtype=np.float64)
    if old.shape!=new.shape:
        raise ValueError("shape mismatch")
    return new-old,new+old


def replacement_statistics_delta(
    old: np.ndarray, new: np.ndarray
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Exact row-replacement deltas using midpoint/difference algebra.

      D = new-old
      P = new+old

      Delta S = sum D
      Delta Q = sum (D * P)
      Delta G = 1/2 (D^T P + P^T D)

    The Gram update uses one GEMM plus symmetrization instead of
    new.T@new - old.T@old.
    """
    D,P=midpoint_difference(old,new)
    ds=D.sum(axis=0)
    dq=np.einsum("ij,ij->j",D,P,optimize=False)
    m=D.T@P
    dg=(m+m.T)*0.5
    return ds,dq,dg


def replacement_gram_delta(old: np.ndarray,new: np.ndarray) -> SymmetricLowRankDelta:
    """
    Keep the Gram replacement unmaterialized.

      Delta G = 1/2(D^T P + P^T D)

    with U=[D^T, P^T] and small symmetric core
      C=[[0, .5 I],[.5 I, 0]].

    Nominal rank <= 2e.
    """
    D,P=midpoint_difference(old,new)
    e=D.shape[0]
    U=np.concatenate([D.T,P.T],axis=1)
    C=np.zeros((2*e,2*e),dtype=np.float64)
    C[:e,e:]=0.5*np.eye(e)
    C[e:,:e]=0.5*np.eye(e)
    return SymmetricLowRankDelta(U,C)


def replacement_cross_delta(
    old_x: np.ndarray,new_x: np.ndarray,
    old_y: np.ndarray,new_y: np.ndarray,
) -> np.ndarray:
    """
    Exact cross-statistic replacement:
      Delta(X^T Y) = 1/2(Dx^T Py + Px^T Dy).
    """
    Dx,Px=midpoint_difference(old_x,new_x)
    Dy,Py=midpoint_difference(old_y,new_y)
    return 0.5*(Dx.T@Py + Px.T@Dy)


def lowrank_replacement_statistics(
    A: np.ndarray,
    B: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,SymmetricLowRankDelta]:
    """
    Exact replacement statistics when D = new-old = A B^T has rank r.

    Let P = new+old and C=A^T P. Then:
      Delta S = B (A^T 1)
      Delta Q_j = <B[j,:], C[:,j]>
      Delta G = 1/2(B C + C^T B^T), rank <= 2r.

    The dense Gram delta is never required.
    """
    A=np.asarray(A,dtype=np.float64)
    B=np.asarray(B,dtype=np.float64)
    old=np.asarray(old,dtype=np.float64)
    new=np.asarray(new,dtype=np.float64)
    if old.shape!=new.shape or A.shape[0]!=old.shape[0] or B.shape[0]!=old.shape[1] or A.shape[1]!=B.shape[1]:
        raise ValueError("shape mismatch")
    P=new+old
    Csmall=A.T@P
    ds=B@A.sum(axis=0)
    dq=np.einsum("jr,rj->j",B,Csmall,optimize=False)
    r=A.shape[1]
    U=np.concatenate([B,Csmall.T],axis=1)
    core=np.zeros((2*r,2*r),dtype=np.float64)
    core[:r,r:]=0.5*np.eye(r)
    core[r:,:r]=0.5*np.eye(r)
    return ds,dq,SymmetricLowRankDelta(U,core)
