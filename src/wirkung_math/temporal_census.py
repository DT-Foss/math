from __future__ import annotations
from collections import Counter
from dataclasses import dataclass,asdict
import json
import numpy as np


def support_for_energy(x:np.ndarray,fraction:float=0.999)->int:
    a=np.square(np.asarray(x,dtype=np.float64).ravel())
    total=float(a.sum())
    if total==0.0:
        return 0
    order=np.sort(a)[::-1]
    return int(np.searchsorted(np.cumsum(order),fraction*total,side="left")+1)


def effective_rank(x:np.ndarray,fraction:float=0.999)->int:
    X=np.asarray(x,dtype=np.float64)
    if X.ndim!=2:
        raise ValueError("effective_rank expects a matrix")
    s=np.linalg.svd(X,compute_uv=False)
    return support_for_energy(s,fraction)


@dataclass(frozen=True)
class TemporalDeltaRecord:
    width:int
    coordinate_support:int
    coordinate_fraction:float
    block_size:int
    changed_blocks:int
    block_fraction:float
    l2_norm:float
    relative_l2:float
    top99_support:int
    top999_support:int
    basis_top99:int|None=None
    basis_top999:int|None=None
    basis_residual_fraction:float|None=None
    code_hamming:int|None=None
    code_hamming_fraction:float|None=None


def measure_temporal_delta(
    previous:np.ndarray,
    current:np.ndarray,
    *,
    atol:float=0.0,
    block_size:int=32,
    basis:np.ndarray|None=None,
    previous_codes:np.ndarray|None=None,
    current_codes:np.ndarray|None=None,
)->TemporalDeltaRecord:
    prev=np.asarray(previous)
    cur=np.asarray(current)
    if prev.shape!=cur.shape or prev.ndim!=1:
        raise ValueError("previous/current must be equal-length vectors")
    delta=np.asarray(cur-prev,dtype=np.float64)
    support=np.flatnonzero(np.abs(delta)>atol)
    width=len(delta)
    blocks=np.unique(support//block_size) if len(support) else np.empty(0,dtype=np.int64)
    total_blocks=(width+block_size-1)//block_size
    norm=float(np.linalg.norm(delta))
    base=float(np.linalg.norm(cur))

    b99=b999=None
    residual_fraction=None
    if basis is not None:
        B=np.asarray(basis,dtype=np.float64)
        if B.ndim!=2 or B.shape[0]!=width:
            raise ValueError("basis shape mismatch")
        coeff=B.T@delta
        recon=B@coeff
        b99=support_for_energy(coeff,.99)
        b999=support_for_energy(coeff,.999)
        residual=float(np.linalg.norm(delta-recon))
        residual_fraction=0.0 if norm==0.0 else residual/norm

    hamming=hamming_fraction=None
    if previous_codes is not None or current_codes is not None:
        if previous_codes is None or current_codes is None:
            raise ValueError("both code arrays are required")
        pc=np.asarray(previous_codes)
        cc=np.asarray(current_codes)
        if pc.shape!=cc.shape:
            raise ValueError("code shape mismatch")
        hamming=int(np.count_nonzero(pc!=cc))
        hamming_fraction=hamming/max(1,pc.size)

    return TemporalDeltaRecord(
        width=width,
        coordinate_support=int(len(support)),
        coordinate_fraction=len(support)/max(1,width),
        block_size=int(block_size),
        changed_blocks=int(len(blocks)),
        block_fraction=len(blocks)/max(1,total_blocks),
        l2_norm=norm,
        relative_l2=0.0 if base==0.0 else norm/base,
        top99_support=support_for_energy(delta,.99),
        top999_support=support_for_energy(delta,.999),
        basis_top99=b99,
        basis_top999=b999,
        basis_residual_fraction=residual_fraction,
        code_hamming=hamming,
        code_hamming_fraction=hamming_fraction,
    )


class TemporalDeltaCensus:
    """Accumulate exact temporal-delta geometry for Origami/Beast layer traces."""
    def __init__(self)->None:
        self.records:dict[str,list[TemporalDeltaRecord]]={}
        self.support_patterns:dict[str,Counter[tuple[int,...]]]={}

    def add(
        self,name:str,previous:np.ndarray,current:np.ndarray,**kwargs
    )->TemporalDeltaRecord:
        rec=measure_temporal_delta(previous,current,**kwargs)
        self.records.setdefault(name,[]).append(rec)
        delta=np.asarray(current)-np.asarray(previous)
        support=tuple(np.flatnonzero(delta).tolist())
        self.support_patterns.setdefault(name,Counter())[support]+=1
        return rec

    def summary(self)->dict:
        out={}
        for name,rows in self.records.items():
            numeric={}
            keys=asdict(rows[0]).keys() if rows else ()
            for key in keys:
                values=[getattr(r,key) for r in rows if getattr(r,key) is not None]
                if values:
                    a=np.asarray(values,dtype=np.float64)
                    numeric[key]={
                        "median":float(np.median(a)),
                        "p90":float(np.percentile(a,90)),
                        "max":float(np.max(a)),
                    }
            patterns=self.support_patterns.get(name,Counter())
            numeric["most_common_support_patterns"]=patterns.most_common(16)
            numeric["steps"]=len(rows)
            out[name]=numeric
        return out

    def to_json(self)->str:
        return json.dumps(self.summary(),indent=2,sort_keys=True)
