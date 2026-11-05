from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TensorTrain:
    """Dense-reference tensor-train compiler used by partition-function states."""
    cores:tuple[np.ndarray,...]
    shape:tuple[int,...]

    @classmethod
    def from_tensor(
        cls,tensor:np.ndarray,*,rtol:float=0.0,max_rank:int|None=None
    )->"TensorTrain":
        T=np.asarray(tensor,dtype=np.float64)
        shape=tuple(int(x) for x in T.shape)
        if not shape:
            raise ValueError("scalar tensor is not a tensor train")
        cores=[]
        rprev=1
        work=T
        for mode in range(len(shape)-1):
            matrix=work.reshape(rprev*shape[mode],-1)
            U,s,Vh=np.linalg.svd(matrix,full_matrices=False)
            if len(s)==0:
                rank=1
            elif rtol>0:
                rank=max(1,int(np.count_nonzero(s>s[0]*rtol)))
            else:
                # Exact numerical rank at machine precision.
                threshold=np.finfo(np.float64).eps*max(matrix.shape)*s[0]
                rank=max(1,int(np.count_nonzero(s>threshold)))
            if max_rank is not None:
                rank=min(rank,int(max_rank))
            core=U[:,:rank].reshape(rprev,shape[mode],rank)
            cores.append(core)
            work=(s[:rank,None]*Vh[:rank,:]).reshape((rank,)+shape[mode+1:])
            rprev=rank
        cores.append(work.reshape(rprev,shape[-1],1))
        return cls(tuple(cores),shape)

    @property
    def ranks(self)->tuple[int,...]:
        return (1,)+tuple(int(core.shape[2]) for core in self.cores)

    @property
    def storage_values(self)->int:
        return int(sum(core.size for core in self.cores))

    @property
    def dense_values(self)->int:
        return int(np.prod(self.shape,dtype=np.int64))

    @property
    def compression_ratio(self)->float:
        return self.dense_values/max(1,self.storage_values)

    def contract_prefix(self,vectors:list[np.ndarray]|tuple[np.ndarray,...]):
        """
        Contract vectors into the leading modes.

        This runtime needs either all modes (scalar H) or all but one mode
        (vector-valued M). Those two cases remain allocation-free.
        """
        count=len(vectors)
        if count not in (len(self.cores),len(self.cores)-1):
            raise ValueError("contract all modes or all but the final mode")
        state=np.ones(1,dtype=np.float64)
        for core,vector0 in zip(self.cores[:count],vectors):
            vector=np.asarray(vector0,dtype=np.float64)
            if vector.shape!=(core.shape[1],):
                raise ValueError("vector/mode shape mismatch")
            state=np.einsum("a,aib,i->b",state,core,vector,optimize=True)
        if count==len(self.cores):
            if state.shape!=(1,):
                raise RuntimeError("invalid terminal TT rank")
            return float(state[0])
        final=self.cores[-1]
        if final.shape[2]!=1:
            raise RuntimeError("invalid final TT rank")
        return np.einsum("a,aib->i",state,final,optimize=True)

    def reconstruct(self)->np.ndarray:
        value=self.cores[0]
        # [1,n1,r1] -> [n1,r1]
        value=value[0]
        for core in self.cores[1:]:
            value=np.einsum("...a,aib->...ib",value,core,optimize=True)
            value=value[...,0] if core.shape[2]==1 else value
        return np.asarray(value).reshape(self.shape)
