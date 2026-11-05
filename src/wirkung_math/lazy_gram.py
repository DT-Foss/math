from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .update_ledger import IncrementalWoodburyLedger

@dataclass
class LazyGramState:
    """
    Gram/scatter matrix maintained as base + low-rank replacement ledger.

    Row replacement:
      G' = G - X_old^T X_old + X_new^T X_new
         = G + U V^T

    The dense d×d Gram matrix is not rewritten unless explicitly requested.
    Solve/logdet consume the same streaming Woodbury state; selected blocks can
    be materialized directly from the factors.
    """
    base_gram:np.ndarray
    solve_base:object
    base_sign:float
    base_logabsdet:float
    ledger:IncrementalWoodburyLedger

    @classmethod
    def from_gram(cls,gram:np.ndarray,solve_base=None):
        G=np.asarray(gram,dtype=np.float64).copy()
        if solve_base is None:
            solve_base=lambda b:np.linalg.solve(G,b)
        sign,logabs=np.linalg.slogdet(G)
        return cls(G,solve_base,float(sign),float(logabs),IncrementalWoodburyLedger(solve_base))

    def replace_rows(self,old_rows:np.ndarray,new_rows:np.ndarray)->None:
        old=np.asarray(old_rows,dtype=np.float64)
        new=np.asarray(new_rows,dtype=np.float64)
        if old.shape!=new.shape or old.ndim!=2 or old.shape[1]!=self.base_gram.shape[0]:
            raise ValueError("row shape mismatch")
        U=np.concatenate([new.T,old.T],axis=1)
        V=np.concatenate([new.T,-old.T],axis=1)
        self.ledger.append(U,V)

    @property
    def rank_bound(self)->int:
        return self.ledger.rank

    def solve(self,rhs:np.ndarray)->np.ndarray:
        return self.ledger.solve(rhs)

    @property
    def slogdet(self)->tuple[float,float]:
        return self.ledger.logdet_update(self.base_sign,self.base_logabsdet)

    def block(self,rows:np.ndarray,cols:np.ndarray)->np.ndarray:
        I=np.asarray(rows,dtype=np.int64);J=np.asarray(cols,dtype=np.int64)
        out=self.base_gram[np.ix_(I,J)].copy()
        if self.ledger.U is not None:
            assert self.ledger.V is not None
            out+=self.ledger.U[I]@self.ledger.V[J].T
        return out

    def materialize(self)->np.ndarray:
        if self.ledger.U is None:
            return self.base_gram.copy()
        return self.base_gram+self.ledger.materialize_update()
