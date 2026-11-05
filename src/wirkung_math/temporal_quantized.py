from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantizedVectorState:
    """
    Exact temporal state in the runtime's quantized arithmetic.

    `levels` may be one global codebook [L] or per-coordinate levels [D,L].
    Code equality turns a dense floating-point drift into exact zero work.
    """
    codes:np.ndarray
    levels:np.ndarray

    def __post_init__(self):
        self.codes=np.asarray(self.codes).copy()
        self.levels=np.asarray(self.levels)
        if self.codes.ndim!=1 or self.levels.ndim not in (1,2):
            raise ValueError("invalid code/level shape")
        if self.levels.ndim==2 and self.levels.shape[0]!=len(self.codes):
            raise ValueError("per-coordinate codebook width mismatch")

    def decode(self,indices:np.ndarray|None=None)->np.ndarray:
        if indices is None:
            idx=np.arange(len(self.codes),dtype=np.int64)
        else:
            idx=np.asarray(indices,dtype=np.int64)
        code=self.codes[idx].astype(np.int64)
        if self.levels.ndim==1:
            return self.levels[code]
        return self.levels[idx,code]

    def update_codes(
        self,new_codes:np.ndarray,indices:np.ndarray|None=None
    )->tuple[np.ndarray,np.ndarray]:
        if indices is None:
            idx=np.arange(len(self.codes),dtype=np.int64)
            incoming=np.asarray(new_codes)
            if incoming.shape!=self.codes.shape:
                raise ValueError("new code shape mismatch")
        else:
            idx=np.asarray(indices,dtype=np.int64)
            incoming=np.asarray(new_codes)
            if incoming.shape!=(len(idx),):
                raise ValueError("new code shape mismatch")
        old_codes=self.codes[idx].copy()
        changed_mask=old_codes!=incoming
        changed=idx[changed_mask]
        if not len(changed):
            return changed,np.empty(0,dtype=np.result_type(self.levels,float))
        old_values=self.decode(changed)
        self.codes[idx]=incoming
        new_values=self.decode(changed)
        return changed,new_values-old_values

    @staticmethod
    def bitplane_flips(old_codes:np.ndarray,new_codes:np.ndarray,bits:int)->int:
        old=np.asarray(old_codes,dtype=np.uint64)
        new=np.asarray(new_codes,dtype=np.uint64)
        xor=old^new
        total=0
        for bit in range(bits):
            total+=int(np.count_nonzero(xor&(np.uint64(1)<<np.uint64(bit))))
        return total


@dataclass
class QuantizedTemporalLinearState:
    """Exact W x update from changed activation codes only."""
    weight:np.ndarray
    vector:QuantizedVectorState
    y:np.ndarray

    @classmethod
    def from_codes(
        cls,weight:np.ndarray,codes:np.ndarray,levels:np.ndarray
    )->"QuantizedTemporalLinearState":
        W=np.asarray(weight)
        q=QuantizedVectorState(codes,levels)
        x=q.decode()
        if W.ndim!=2 or W.shape[1]!=len(x):
            raise ValueError("weight shape mismatch")
        return cls(W,q,W@x)

    def update_codes(
        self,new_codes:np.ndarray,indices:np.ndarray|None=None
    )->tuple[np.ndarray,np.ndarray]:
        changed,delta=self.vector.update_codes(new_codes,indices)
        if len(changed):
            self.y=self.y+self.weight[:,changed]@delta
        return self.y,changed
