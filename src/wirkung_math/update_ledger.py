from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from .woodbury import GeneralLowRankUpdate,SolverWoodburyCache


@dataclass
class LowRankUpdateLedger:
    """
    Coalesce a stream of A <- A + U_i V_i^T updates against one base solver.

    Instead of nesting Woodbury caches, concatenate factors and periodically
    rank-reveal the combined update. The current matrix is
      A_current = A_base + U_all V_all^T.
    """
    solve_base:object
    U_parts:list[np.ndarray]=field(default_factory=list)
    V_parts:list[np.ndarray]=field(default_factory=list)

    @property
    def nominal_rank(self)->int:
        return sum(x.shape[1] for x in self.U_parts)

    def append(self,U:np.ndarray,V:np.ndarray)->None:
        U=np.asarray(U,dtype=np.float64);V=np.asarray(V,dtype=np.float64)
        if U.shape!=V.shape or U.ndim!=2:
            raise ValueError("U and V must have same n×r shape")
        self.U_parts.append(U);self.V_parts.append(V)

    def combined(self)->GeneralLowRankUpdate:
        if not self.U_parts:
            raise ValueError("empty ledger")
        return GeneralLowRankUpdate(np.concatenate(self.U_parts,axis=1),
                                    np.concatenate(self.V_parts,axis=1))

    def compact(self,rtol:float=1e-12)->None:
        """
        Exact numerical rank reveal of U V^T through reduced QR + small SVD.
        """
        upd=self.combined()
        qu,ru=np.linalg.qr(upd.U,mode="reduced")
        qv,rv=np.linalg.qr(upd.V,mode="reduced")
        core=ru@rv.T
        a,s,bt=np.linalg.svd(core,full_matrices=False)
        if not len(s) or s[0]==0:
            self.U_parts=[];self.V_parts=[];return
        keep=s>s[0]*rtol
        root=np.sqrt(s[keep])
        U=(qu@a[:,keep])*root[None,:]
        V=(qv@bt[keep,:].T)*root[None,:]
        self.U_parts=[U];self.V_parts=[V]

    def cache(self)->SolverWoodburyCache:
        return SolverWoodburyCache.prepare(self.solve_base,self.combined())

    def solve(self,rhs:np.ndarray)->np.ndarray:
        return self.cache().apply_inverse(rhs)


@dataclass
class IncrementalWoodburyLedger:
    """
    Streaming A <- A + U_i V_i^T state with an incrementally maintained Woodbury core.

    Old A^-1 U columns are never recomputed. Appending a q-rank update needs only
    solve_base(U_new), O(n*r*q) cross terms, and a q×q Schur update of M^-1 where

      M = I + V^T A_base^-1 U.

    Solve then uses one base solve plus the accumulated compact correction.
    """
    solve_base:object
    U:np.ndarray|None=None
    V:np.ndarray|None=None
    AU:np.ndarray|None=None
    M_inv:np.ndarray|None=None
    core_sign:float=1.0
    core_logabsdet:float=0.0

    @property
    def rank(self)->int:
        return 0 if self.U is None else int(self.U.shape[1])

    def append(self,U_new:np.ndarray,V_new:np.ndarray)->None:
        U_new=np.asarray(U_new,dtype=np.float64)
        V_new=np.asarray(V_new,dtype=np.float64)
        if U_new.shape!=V_new.shape or U_new.ndim!=2:
            raise ValueError("U and V must have same n×r shape")
        AU_new=np.asarray(self.solve_base(U_new),dtype=np.float64)
        q=U_new.shape[1]

        if self.U is None:
            D=np.eye(q)+V_new.T@AU_new
            self.U=U_new.copy();self.V=V_new.copy();self.AU=AU_new
            self.M_inv=np.linalg.inv(D)
            sign,logabs=np.linalg.slogdet(D)
            self.core_sign=float(sign);self.core_logabsdet=float(logabs)
            return

        assert self.V is not None and self.AU is not None and self.M_inv is not None
        A=self.V.T@AU_new
        B=V_new.T@self.AU
        D=np.eye(q)+V_new.T@AU_new
        S=D-B@self.M_inv@A
        S_inv=np.linalg.inv(S)
        sign_s,log_s=np.linalg.slogdet(S)
        self.core_sign*=float(sign_s)
        self.core_logabsdet+=float(log_s)

        MI=self.M_inv
        MIA=MI@A
        BMI=B@MI
        top_left=MI+MIA@S_inv@BMI
        top_right=-MIA@S_inv
        bottom_left=-S_inv@BMI
        bottom_right=S_inv
        self.M_inv=np.block([[top_left,top_right],[bottom_left,bottom_right]])
        self.U=np.concatenate([self.U,U_new],axis=1)
        self.V=np.concatenate([self.V,V_new],axis=1)
        self.AU=np.concatenate([self.AU,AU_new],axis=1)

    def solve(self,rhs:np.ndarray)->np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        if self.U is None:
            return np.asarray(self.solve_base(rhs),dtype=np.float64)
        assert self.V is not None and self.AU is not None and self.M_inv is not None
        vector=rhs.ndim==1
        B=rhs[:,None] if vector else rhs
        z=np.asarray(self.solve_base(B),dtype=np.float64)
        out=z-self.AU@(self.M_inv@(self.V.T@z))
        return out[:,0] if vector else out

    def logdet_update(self,base_sign:float,base_logabsdet:float)->tuple[float,float]:
        return float(base_sign*self.core_sign),float(base_logabsdet+self.core_logabsdet)

    def materialize_update(self)->np.ndarray:
        if self.U is None:
            raise ValueError("empty ledger")
        assert self.V is not None
        return self.U@self.V.T
