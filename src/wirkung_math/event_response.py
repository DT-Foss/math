from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _decode_state(code:int,base:int,width:int,qmax:int)->np.ndarray:
    out=np.empty(width,dtype=np.int64)
    z=int(code)
    for i in range(width):
        out[i]=(z%base)-qmax
        z//=base
    return out


@dataclass
class IntegerResponseLinearState:
    """
    Persistent Linear state in deterministic response-quantized arithmetic.

    Activation:
        x_i ~= scale_i * code_i, code_i in [-q,q]

    Compile the consumer response R_ji = W_ji * scale_i once, then quantize each
    output row with response_scale_j:
        R_ji ~= response_scale_j * response_q[i,j].

    Runtime accumulator:
        a_j = sum_i response_q[i,j] * code_i

    A block-state table stores each block's contribution. Temporal execution is
    therefore exactly:
        a' = a + table[b,new_state] - table[b,old_state]
    for blocks whose code state changed.

    No recurrent floating-point accumulation exists; full and temporal routes
    share the same integer semantics.
    """
    response_q:np.ndarray       # [input, output], integer
    response_scale:np.ndarray   # [output]
    activation_scale:np.ndarray # [input]
    qmax:int
    block_size:int
    table:np.ndarray            # [blocks, states, output], int64
    states:np.ndarray|None=None
    accumulator:np.ndarray|None=None

    @classmethod
    def compile(
        cls,
        weight:np.ndarray,
        activation_scale:np.ndarray,
        *,
        qmax:int,
        block_size:int=2,
        response_qmax:int=32767,
    )->"IntegerResponseLinearState":
        W=np.asarray(weight,dtype=np.float64)
        s=np.asarray(activation_scale,dtype=np.float64)
        if W.ndim!=2 or s.shape!=(W.shape[1],):
            raise ValueError("shape mismatch")
        if qmax<1 or block_size<1 or response_qmax<1:
            raise ValueError("invalid quantization/layout")
        response=W*s[None,:]
        row_scale=np.maximum(np.max(np.abs(response),axis=1)/response_qmax,1e-30)
        rq=np.rint(response/row_scale[:,None]).clip(-response_qmax,response_qmax).astype(np.int32).T
        n,o=rq.shape
        base=2*qmax+1
        states_per_block=base**block_size
        if states_per_block>65536:
            raise ValueError("block state space too large")
        blocks=(n+block_size-1)//block_size
        table=np.zeros((blocks,states_per_block,o),dtype=np.int64)
        for b in range(blocks):
            start=b*block_size
            stop=min(start+block_size,n)
            width=stop-start
            cols=rq[start:stop]
            for state in range(states_per_block):
                code=_decode_state(state,base,block_size,qmax)[:width]
                table[b,state]=code@cols
        return cls(rq,row_scale,s,int(qmax),int(block_size),table)

    @property
    def input_width(self)->int:
        return int(self.response_q.shape[0])

    @property
    def output_width(self)->int:
        return int(self.response_q.shape[1])

    @property
    def states_per_block(self)->int:
        return int(self.table.shape[1])

    @property
    def table_bytes(self)->int:
        return int(self.table.nbytes)

    def encode(self,x:np.ndarray)->np.ndarray:
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(self.input_width,):
            raise ValueError("input shape mismatch")
        return np.rint(x/self.activation_scale).clip(-self.qmax,self.qmax).astype(np.int16)

    def block_states(self,codes:np.ndarray)->np.ndarray:
        c=np.asarray(codes)
        if c.shape!=(self.input_width,):
            raise ValueError("code shape mismatch")
        base=2*self.qmax+1
        blocks=self.table.shape[0]
        states=np.zeros(blocks,dtype=np.int32)
        for b in range(blocks):
            code=0
            mul=1
            for u in range(self.block_size):
                i=b*self.block_size+u
                digit=self.qmax if i>=len(c) else int(c[i])+self.qmax
                code+=digit*mul
                mul*=base
            states[b]=code
        return states

    def full_codes(self,codes:np.ndarray)->np.ndarray:
        states=self.block_states(codes)
        acc=self.table[np.arange(len(states)),states].sum(axis=0,dtype=np.int64)
        self.states=states
        self.accumulator=acc
        return self.response_scale*acc

    def full(self,x:np.ndarray)->np.ndarray:
        return self.full_codes(self.encode(x))

    def update_codes(self,codes:np.ndarray)->tuple[np.ndarray,int]:
        states=self.block_states(codes)
        if self.states is None or self.accumulator is None:
            return self.full_codes(codes),len(states)
        changed=np.flatnonzero(states!=self.states)
        for b in changed:
            self.accumulator += self.table[b,states[b]]-self.table[b,self.states[b]]
        self.states=states
        return self.response_scale*self.accumulator,int(len(changed))

    def update(self,x:np.ndarray)->tuple[np.ndarray,int]:
        return self.update_codes(self.encode(x))

    def reset(self)->None:
        self.states=None
        self.accumulator=None
