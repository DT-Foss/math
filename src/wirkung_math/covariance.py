from __future__ import annotations
import numpy as np
from .symmetric import SymmetricLowRankDelta


def covariance_replacement_delta(
    count:int,
    old_mean:np.ndarray,
    delta_sum:np.ndarray,
    gram_delta:SymmetricLowRankDelta,
) -> SymmetricLowRankDelta:
    """
    Exact population-covariance update factor for fixed row count.

      C = G/n - mu mu^T
      dmu = dS/n
      dC = dG/n - mu dmu^T - dmu mu^T - dmu dmu^T.

    If rank(dG)<=k then rank(dC)<=k+2.
    """
    if count<=0:
        raise ValueError("count must be positive")
    mu=np.asarray(old_mean,dtype=np.float64)
    ds=np.asarray(delta_sum,dtype=np.float64)
    if mu.shape!=ds.shape or mu.ndim!=1 or gram_delta.U.shape[0]!=len(mu):
        raise ValueError("shape mismatch")
    dmu=ds/float(count)

    U1=gram_delta.U
    C1=gram_delta.C/float(count)

    U2=np.stack([mu,dmu],axis=1)
    C2=np.array([[0.0,-1.0],[-1.0,-1.0]],dtype=np.float64)

    U=np.concatenate([U1,U2],axis=1)
    C=np.zeros((C1.shape[0]+2,C1.shape[1]+2),dtype=np.float64)
    C[:C1.shape[0],:C1.shape[1]]=C1
    C[-2:,-2:]=C2
    return SymmetricLowRankDelta(U,C)
