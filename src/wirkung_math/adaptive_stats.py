from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .native_rank1 import NativeRank1StatsState
from .replacement import midpoint_difference

@dataclass
class AdaptiveStatsState:
    """
    Persistent sum/sumsq/Gram state with edit-rank-aware execution.

    e=1: native in-place symmetric midpoint update.
    e>1: one D.T@P GEMM + symmetrization, no old/new Gram pair.
    """
    sum:np.ndarray
    sumsq:np.ndarray
    gram:np.ndarray
    native_rank1:bool=True

    def __post_init__(self):
        self.sum=np.ascontiguousarray(self.sum,dtype=np.float64)
        self.sumsq=np.ascontiguousarray(self.sumsq,dtype=np.float64)
        self.gram=np.ascontiguousarray(self.gram,dtype=np.float64)
        self._native=None
        if self.native_rank1:
            try:
                self._native=NativeRank1StatsState(self.sum,self.sumsq,self.gram)
                # keep public references to the arrays owned by native state
                self.sum=self._native.sum
                self.sumsq=self._native.sumsq
                self.gram=self._native.gram
            except Exception:
                self._native=None

    def replace_rows(self,old:np.ndarray,new:np.ndarray)->str:
        old=np.asarray(old,dtype=np.float64)
        new=np.asarray(new,dtype=np.float64)
        if old.ndim==1:
            old=old[None,:];new=new[None,:]
        if old.shape!=new.shape or old.shape[1]!=len(self.sum):
            raise ValueError("shape mismatch")
        if old.shape[0]==1 and self._native is not None:
            self._native.update(old[0],new[0])
            return "native_rank1"

        D,P=midpoint_difference(old,new)
        self.sum += D.sum(axis=0)
        self.sumsq += np.einsum("ij,ij->j",D,P,optimize=False)
        M=D.T@P
        self.gram += (M+M.T)*0.5
        return "midpoint_gemm"
