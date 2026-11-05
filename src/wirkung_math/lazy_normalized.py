from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .bilinear import full_normalized_bilinear


@dataclass(frozen=True)
class KVEdit:
    indices: np.ndarray
    old_k: np.ndarray
    new_k: np.ndarray
    old_v: np.ndarray
    new_v: np.ndarray


@dataclass
class LazyNormalizedKVState:
    """
    Versioned normalized bilinear state with deferred row reconciliation.

    K/V edits are appended in O(e*d) without touching all query rows. A query
    row applies only the edits since its own version. Rows that have not been
    requested incur no work. Query rows can be invalidated independently when
    Q changes.

    This composes streaming state with consumer-aware routing.
    """
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    output: np.ndarray
    logz: np.ndarray
    scale: float
    row_version: np.ndarray
    valid: np.ndarray
    edits: list[KVEdit]

    @classmethod
    def from_qkv(cls,q:np.ndarray,k:np.ndarray,v:np.ndarray,scale:float|None=None):
        q=np.asarray(q).copy();k=np.asarray(k).copy();v=np.asarray(v).copy()
        if scale is None:
            scale=1.0/math.sqrt(q.shape[1])
        out,logz=full_normalized_bilinear(q,k,v,scale)
        n=len(q)
        return cls(q,k,v,out,logz,float(scale),
                   np.zeros(n,dtype=np.int64),np.ones(n,dtype=bool),[])

    @property
    def version(self)->int:
        return len(self.edits)

    def append_kv_update(self,indices:np.ndarray,new_k:np.ndarray,new_v:np.ndarray)->None:
        idx=np.asarray(indices,dtype=np.int64)
        nk=np.asarray(new_k);nv=np.asarray(new_v)
        if nk.shape!=self.k[idx].shape or nv.shape!=self.v[idx].shape:
            raise ValueError("replacement shape mismatch")
        self.edits.append(KVEdit(idx.copy(),self.k[idx].copy(),nk.copy(),
                                 self.v[idx].copy(),nv.copy()))
        self.k[idx]=nk;self.v[idx]=nv

    def update_query_rows(self,indices:np.ndarray,new_q:np.ndarray)->None:
        idx=np.asarray(indices,dtype=np.int64)
        nq=np.asarray(new_q)
        if nq.shape!=self.q[idx].shape:
            raise ValueError("query shape mismatch")
        self.q[idx]=nq
        self.valid[idx]=False

    def _full_row(self,i:int)->None:
        scores=(self.q[i]@self.k.T)*self.scale
        m=float(np.max(scores))
        ex=np.exp(scores-m)
        z=float(np.sum(ex))
        self.output[i]=(ex@self.v)/z
        self.logz[i]=math.log(z)+m
        self.row_version[i]=self.version
        self.valid[i]=True

    def reconcile(self,rows:np.ndarray,cancellation_floor:float=2.0**-12)->dict:
        rows=np.unique(np.asarray(rows,dtype=np.int64))
        current=self.version
        full_rows=0
        applied_batches=0
        for i0 in rows:
            i=int(i0)
            if not self.valid[i]:
                self._full_row(i);full_rows+=1;continue

            pos=int(self.row_version[i])
            fallback=False
            while pos<current:
                edit=self.edits[pos]
                old_scores=(self.q[i]@edit.old_k.T)*self.scale
                new_scores=(self.q[i]@edit.new_k.T)*self.scale
                a=np.exp(old_scores-self.logz[i])
                b=np.exp(new_scores-self.logz[i])
                mass=1.0-float(np.sum(a))+float(np.sum(b))
                if not np.isfinite(mass) or mass<=cancellation_floor:
                    fallback=True;break
                self.output[i]=(self.output[i]-a@edit.old_v+b@edit.new_v)/mass
                self.logz[i]+=math.log(mass)
                pos+=1;applied_batches+=1

            if fallback:
                self._full_row(i);full_rows+=1
            else:
                self.row_version[i]=current

        return {
            "rows":int(len(rows)),
            "applied_edit_batches":int(applied_batches),
            "full_rows":int(full_rows),
            "current_version":int(current),
        }

    def query(self,rows:np.ndarray,cancellation_floor:float=2.0**-12)->np.ndarray:
        rows=np.asarray(rows,dtype=np.int64)
        self.reconcile(rows,cancellation_floor)
        return self.output[rows]

    def compact_log(self,max_entries:int)->None:
        """
        Bound edit-log memory exactly.

        Rows older than the retained suffix are invalidated. Their next query
        performs one exact full-row reconstruction from current K/V instead of
        requiring discarded historical edits.
        """
        if max_entries<0:
            raise ValueError("max_entries must be nonnegative")
        drop=max(0,len(self.edits)-max_entries)
        if drop==0:return
        stale=self.row_version<drop
        self.valid[stale]=False
        self.row_version[~stale]-=drop
        self.row_version[stale]=0
        self.edits=self.edits[drop:]
