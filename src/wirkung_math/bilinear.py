from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def full_normalized_bilinear(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float|None=None
) -> tuple[np.ndarray,np.ndarray]:
    q=np.asarray(q); k=np.asarray(k); v=np.asarray(v)
    if scale is None:
        scale=1.0/np.sqrt(q.shape[1])
    scores=(q@k.T)*scale
    m=scores.max(axis=1,keepdims=True)
    ex=np.exp(scores-m)
    z=ex.sum(axis=1,keepdims=True)
    return (ex/z)@v, np.log(z[:,0])+m[:,0]


@dataclass
class DynamicNormalizedBilinearState:
    """
    Stateful normalized bilinear map.

    Replacing e rows changes:
      - e full score rows because Q_U changed;
      - e score columns for all clean rows because K_U changed;
      - e value rows because V_U changed.

    Clean rows use an exact normalized sufficient-statistic replacement;
    dirty query rows are recomputed in full.
    """
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    output: np.ndarray
    logz: np.ndarray
    scale: float

    @classmethod
    def from_qkv(
        cls,q: np.ndarray,k: np.ndarray,v: np.ndarray,scale: float|None=None
    ) -> "DynamicNormalizedBilinearState":
        q=np.asarray(q).copy(); k=np.asarray(k).copy(); v=np.asarray(v).copy()
        if scale is None:
            scale=1.0/np.sqrt(q.shape[1])
        out,logz=full_normalized_bilinear(q,k,v,scale)
        return cls(q,k,v,out,logz,float(scale))

    def replace_rows(
        self,
        indices: np.ndarray,
        new_q: np.ndarray,
        new_k: np.ndarray,
        new_v: np.ndarray,
        cancellation_floor: float=2.0**-12,
    ) -> dict:
        idx=np.asarray(indices,dtype=np.int64)
        if len(np.unique(idx)) != len(idx):
            raise ValueError("indices must be unique")
        new_q=np.asarray(new_q); new_k=np.asarray(new_k); new_v=np.asarray(new_v)
        if new_q.shape != self.q[idx].shape or new_k.shape != self.k[idx].shape or new_v.shape != self.v[idx].shape:
            raise ValueError("replacement shape mismatch")

        mask=np.ones(len(self.q),dtype=bool)
        mask[idx]=False
        clean=np.flatnonzero(mask)

        old_k=self.k[idx].copy()
        old_v=self.v[idx].copy()

        old_scores=(self.q[clean]@old_k.T)*self.scale
        new_scores=(self.q[clean]@new_k.T)*self.scale
        a=np.exp(old_scores-self.logz[clean,None])
        b=np.exp(new_scores-self.logz[clean,None])
        qmass=1.0-a.sum(axis=1)+b.sum(axis=1)
        fast=np.isfinite(qmass)&(qmass>cancellation_floor)

        fast_rows=clean[fast]
        if len(fast_rows):
            af=a[fast]; bf=b[fast]; qf=qmass[fast]
            self.output[fast_rows]=(
                self.output[fast_rows]-af@old_v+bf@new_v
            )/qf[:,None]
            self.logz[fast_rows]+=np.log(qf)

        self.q[idx]=new_q
        self.k[idx]=new_k
        self.v[idx]=new_v

        fallback=clean[~fast]
        full_rows=np.concatenate([idx,fallback])
        if len(full_rows):
            scores=(self.q[full_rows]@self.k.T)*self.scale
            m=scores.max(axis=1,keepdims=True)
            ex=np.exp(scores-m)
            z=ex.sum(axis=1,keepdims=True)
            self.output[full_rows]=(ex/z)@self.v
            self.logz[full_rows]=np.log(z[:,0])+m[:,0]

        return {
            "recomputed_query_rows":int(len(idx)),
            "fallback_clean_rows":int(len(fallback)),
            "incremental_clean_rows":int(fast.sum()),
        }
