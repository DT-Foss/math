from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .tensor_train import TensorTrain


@dataclass
class TensorTrainPartitionAttention:
    """
    Exact/controlled-rank compiled partition function over quantized key codes.

    For code tuple c=(c1..cr), define tensors

      H[c]   = sum_i exp(base_score_i-m) [code_i=c]
      M[c,v] = sum_i exp(base_score_i-m) value_i[v] [code_i=c].

    A temporal query coefficient alpha contracts each code mode with

      f_j[cj] = exp(alpha_j * level_j[cj]).

    Query cost is governed by TT ranks rather than context length or the number
    of occupied full code tuples. `rtol=0` preserves the numerical tensor rank;
    a positive tolerance yields a controlled compressed approximation.
    """
    levels:tuple[np.ndarray,...]
    mass_tt:TensorTrain
    value_tt:TensorTrain
    base_shift:float
    token_count:int
    value_width:int

    @classmethod
    def compile(
        cls,
        base_scores:np.ndarray,
        key_codes:np.ndarray,
        levels:list[np.ndarray]|tuple[np.ndarray,...],
        values:np.ndarray,
        *,
        rtol:float=0.0,
        max_rank:int|None=None,
    )->"TensorTrainPartitionAttention":
        s=np.asarray(base_scores,dtype=np.float64)
        codes=np.asarray(key_codes,dtype=np.int64)
        V=np.asarray(values,dtype=np.float64)
        lev=tuple(np.asarray(x,dtype=np.float64) for x in levels)
        if s.ndim!=1 or codes.ndim!=2 or V.ndim!=2 or len(s)!=len(codes) or len(s)!=len(V):
            raise ValueError("shape mismatch")
        if codes.shape[1]!=len(lev):
            raise ValueError("code width / level count mismatch")
        shape=tuple(len(x) for x in lev)
        for j,size in enumerate(shape):
            if np.any(codes[:,j]<0) or np.any(codes[:,j]>=size):
                raise ValueError("code outside level range")
        shift=float(np.max(s)) if len(s) else 0.0
        w=np.exp(s-shift)
        H=np.zeros(shape,dtype=np.float64)
        M=np.zeros(shape+(V.shape[1],),dtype=np.float64)
        index=tuple(codes[:,j] for j in range(codes.shape[1]))
        np.add.at(H,index,w)
        np.add.at(M,index+(slice(None),),w[:,None]*V)
        return cls(
            lev,
            TensorTrain.from_tensor(H,rtol=rtol,max_rank=max_rank),
            TensorTrain.from_tensor(M,rtol=rtol,max_rank=max_rank),
            shift,int(len(s)),int(V.shape[1]),
        )

    @property
    def code_width(self)->int:
        return len(self.levels)

    @property
    def storage_values(self)->int:
        return self.mass_tt.storage_values+self.value_tt.storage_values

    @property
    def dense_values(self)->int:
        return self.mass_tt.dense_values+self.value_tt.dense_values

    @property
    def compression_ratio(self)->float:
        return self.dense_values/max(1,self.storage_values)

    def query(self,alpha:np.ndarray)->tuple[np.ndarray,float]:
        a=np.asarray(alpha,dtype=np.float64)
        if a.shape!=(self.code_width,):
            raise ValueError("alpha shape mismatch")
        vectors=[]
        shifts=[]
        for coefficient,levels in zip(a,self.levels):
            raw=coefficient*levels
            shift=float(np.max(raw))
            vectors.append(np.exp(raw-shift))
            shifts.append(shift)
        denominator=float(self.mass_tt.contract_prefix(vectors))
        numerator=np.asarray(self.value_tt.contract_prefix(vectors),dtype=np.float64)
        output=numerator/denominator
        log_partition=self.base_shift+sum(shifts)+float(np.log(denominator))
        return output,log_partition

    def report(self)->dict:
        return {
            "tokens":self.token_count,
            "code_shape":[len(x) for x in self.levels],
            "mass_ranks":list(self.mass_tt.ranks),
            "value_ranks":list(self.value_tt.ranks),
            "storage_values":self.storage_values,
            "dense_values":self.dense_values,
            "compression_ratio":self.compression_ratio,
        }
