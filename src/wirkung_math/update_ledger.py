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
