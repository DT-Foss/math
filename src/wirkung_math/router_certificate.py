from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _topk(x:np.ndarray,k:int)->np.ndarray:
    idx=np.argpartition(x,-k)[-k:]
    return idx[np.argsort(x[idx])[::-1]]


@dataclass
class CertifiedTopKRouterState:
    """
    Exact MoE/top-k router with interval certificates.

    Non-selected logits are not recomputed while their upper bounds remain
    below the exact lower bound of the selected set. Sparse activation changes
    require only the changed router columns and exact updates for selected
    experts. A failed certificate triggers one full refresh.
    """
    weight:np.ndarray
    x:np.ndarray
    logits:np.ndarray
    lower:np.ndarray
    upper:np.ndarray
    selected:np.ndarray
    k:int
    full_refreshes:int=0
    certified_updates:int=0

    @classmethod
    def from_input(cls,weight:np.ndarray,x:np.ndarray,k:int)->"CertifiedTopKRouterState":
        W=np.asarray(weight)
        x=np.asarray(x).copy()
        if W.ndim!=2 or x.shape!=(W.shape[1],) or not 0<k<=W.shape[0]:
            raise ValueError("shape/k mismatch")
        z=W@x
        sel=_topk(z,k)
        return cls(W,x,z.copy(),z.copy(),z.copy(),sel,int(k))

    def _outside_mask(self)->np.ndarray:
        mask=np.ones(len(self.logits),dtype=bool)
        mask[self.selected]=False
        return mask

    def update_sparse(
        self,indices:np.ndarray,new_values:np.ndarray
    )->tuple[np.ndarray,np.ndarray,bool]:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values)
        if new.shape!=(len(idx),):
            raise ValueError("new_values shape mismatch")
        delta=new-self.x[idx]
        self.x[idx]=new

        # Exact Cauchy bounds for every expert restricted to changed columns.
        delta_norm=float(np.linalg.norm(delta))
        bounds=np.linalg.norm(self.weight[:,idx],axis=1)*delta_norm
        self.lower-=bounds
        self.upper+=bounds

        # Previous winners are cheap to update exactly.
        exact_selected=self.logits[self.selected]+self.weight[np.ix_(self.selected,idx)]@delta
        self.logits[self.selected]=exact_selected
        self.lower[self.selected]=exact_selected
        self.upper[self.selected]=exact_selected

        outside=self._outside_mask()
        certified=(
            not np.any(outside)
            or float(np.min(exact_selected))>float(np.max(self.upper[outside]))
        )
        if certified:
            self.certified_updates+=1
        else:
            self.logits=self.weight@self.x
            self.lower=self.logits.copy()
            self.upper=self.logits.copy()
            self.selected=_topk(self.logits,self.k)
            exact_selected=self.logits[self.selected]
            self.full_refreshes+=1

        weights=np.exp(exact_selected-np.max(exact_selected))
        weights=weights/weights.sum()
        return self.selected.copy(),weights,bool(certified)

    @property
    def certificate_margin(self)->float:
        outside=self._outside_mask()
        if not np.any(outside):
            return float("inf")
        return float(np.min(self.lower[self.selected])-np.max(self.upper[outside]))
