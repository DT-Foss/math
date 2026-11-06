from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def balanced_ternary_scales(max_abs:np.ndarray,stages:int)->np.ndarray:
    """
    Positive place values whose signed-digit range [-1,0,+1] spans max_abs.

    Sum_s w_s = max_abs, w_(s+1)=w_s/3.
    """
    m=np.asarray(max_abs,dtype=np.float64)
    if stages<1:
        raise ValueError("stages must be positive")
    denom=1.5*(1.0-3.0**(-stages))
    w0=np.maximum(m/denom,1e-30)
    return np.stack([w0/(3.0**s) for s in range(stages)])


def balanced_ternary_encode(x:np.ndarray,scales:np.ndarray)->np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    s=np.asarray(scales,dtype=np.float64)
    if s.ndim!=2 or x.shape!=(s.shape[1],):
        raise ValueError("shape mismatch")
    residual=x.copy()
    codes=np.empty_like(s,dtype=np.int8)
    for stage in range(len(s)):
        c=np.rint(residual/s[stage]).clip(-1,1).astype(np.int8)
        codes[stage]=c
        residual-=c*s[stage]
    return codes


@dataclass
class SignedDigitResponseLinearState:
    """
    High-precision temporal Linear as several ternary response-event planes.

    Each activation coordinate is represented as a sum of balanced signed
    digits. Every plane has only {-1,0,+1} codes, so block-state cardinality is
    3**block_size regardless of total effective precision.

    Each plane owns a deterministic integer response accumulator. Full and
    temporal execution are exactly identical in the compiled arithmetic.
    """
    response_q:tuple[np.ndarray,...]     # [input, output]
    response_scale:tuple[np.ndarray,...] # [output]
    activation_scales:np.ndarray         # [stage,input]
    block_size:int
    tables:tuple[np.ndarray,...]         # [block,3**block,output] int64
    states:np.ndarray|None=None          # [stage,block]
    accumulators:np.ndarray|None=None    # [stage,output]

    @classmethod
    def compile(
        cls,weight:np.ndarray,max_abs:np.ndarray,*,stages:int,
        block_size:int=2,response_qmax:int=32767,
    )->"SignedDigitResponseLinearState":
        W=np.asarray(weight,dtype=np.float64)
        m=np.asarray(max_abs,dtype=np.float64)
        if W.ndim!=2 or m.shape!=(W.shape[1],):
            raise ValueError("shape mismatch")
        scales=balanced_ternary_scales(m,stages)
        n,o=W.shape[1],W.shape[0]
        B=(n+block_size-1)//block_size
        K=3**block_size
        rqs=[];rss=[];tabs=[]
        for stage in range(stages):
            response=W*scales[stage][None,:]
            rs=np.maximum(np.max(np.abs(response),axis=1)/response_qmax,1e-30)
            rq=np.rint(response/rs[:,None]).clip(
                -response_qmax,response_qmax
            ).astype(np.int32).T
            table=np.zeros((B,K,o),dtype=np.int64)
            for b in range(B):
                start=b*block_size;stop=min(start+block_size,n)
                cols=rq[start:stop]
                for state in range(K):
                    z=state
                    code=np.empty(stop-start,dtype=np.int64)
                    for u in range(len(code)):
                        code[u]=(z%3)-1;z//=3
                    table[b,state]=code@cols
            rqs.append(rq);rss.append(rs);tabs.append(table)
        return cls(tuple(rqs),tuple(rss),scales,int(block_size),tuple(tabs))

    @property
    def stages(self)->int:
        return len(self.tables)

    @property
    def input_width(self)->int:
        return int(self.activation_scales.shape[1])

    @property
    def output_width(self)->int:
        return int(self.tables[0].shape[2])

    @property
    def effective_bits(self)->float:
        return float(np.log2(3.0**self.stages))

    @property
    def table_bytes(self)->int:
        return int(sum(x.nbytes for x in self.tables))

    def _block_states(self,codes:np.ndarray)->np.ndarray:
        if codes.shape!=(self.stages,self.input_width):
            raise ValueError("code shape mismatch")
        B=self.tables[0].shape[0]
        out=np.zeros((self.stages,B),dtype=np.int32)
        for s in range(self.stages):
            for b in range(B):
                state=0;mul=1
                for u in range(self.block_size):
                    i=b*self.block_size+u
                    digit=0 if i>=self.input_width else int(codes[s,i])
                    state+=(digit+1)*mul;mul*=3
                out[s,b]=state
        return out

    def full(self,x:np.ndarray)->np.ndarray:
        codes=balanced_ternary_encode(x,self.activation_scales)
        states=self._block_states(codes)
        acc=np.empty((self.stages,self.output_width),dtype=np.int64)
        idx=np.arange(states.shape[1])
        for s in range(self.stages):
            acc[s]=self.tables[s][idx,states[s]].sum(axis=0,dtype=np.int64)
        self.states=states
        self.accumulators=acc
        return sum(self.response_scale[s]*acc[s] for s in range(self.stages))

    def update(self,x:np.ndarray)->tuple[np.ndarray,int]:
        codes=balanced_ternary_encode(x,self.activation_scales)
        states=self._block_states(codes)
        if self.states is None or self.accumulators is None:
            return self.full(x),states.size
        changed=0
        for s in range(self.stages):
            dirty=np.flatnonzero(states[s]!=self.states[s])
            changed+=len(dirty)
            for b in dirty:
                self.accumulators[s]+=(
                    self.tables[s][b,states[s,b]]-
                    self.tables[s][b,self.states[s,b]]
                )
        self.states=states
        y=sum(
            self.response_scale[s]*self.accumulators[s]
            for s in range(self.stages)
        )
        return y,int(changed)

    def reset(self)->None:
        self.states=None;self.accumulators=None
