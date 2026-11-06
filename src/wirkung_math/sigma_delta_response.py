from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SigmaDeltaResponseLinearState:
    """
    Deterministic error-feedback temporal Linear state.

    A prompt/prefill establishes an exact floating anchor x0 and y0=W x0+b.
    Subsequent states are represented only by integer level crossings:

        x_hat = x0 + step * code
        inc   = round((x - x_hat) / step)
        code' = code + inc

    Because the quantization residual stays in x-x_hat, it is automatically
    carried into the next step. Error therefore remains bounded by the local
    step instead of accumulating across time.

    The consumer response W*step is compiled once into per-output integer
    response pages. Runtime updates one int64 accumulator:

        acc' = acc + response_q^T inc

    Full reconstruction and temporal updates share exactly the same compiled
    arithmetic. Rebase simply chooses a new exact anchor.
    """
    weight:np.ndarray
    bias:np.ndarray
    step:np.ndarray
    response_q:np.ndarray
    response_scale:np.ndarray
    anchor:np.ndarray|None=None
    codes:np.ndarray|None=None
    accumulator:np.ndarray|None=None
    base_output:np.ndarray|None=None

    @classmethod
    def compile(
        cls,weight:np.ndarray,step:np.ndarray,bias:np.ndarray|None=None,
        *,response_qmax:int=32767,
    )->"SigmaDeltaResponseLinearState":
        W=np.asarray(weight,dtype=np.float64)
        h=np.asarray(step,dtype=np.float64)
        if W.ndim!=2 or h.shape!=(W.shape[1],):
            raise ValueError("shape mismatch")
        b=np.zeros(W.shape[0],dtype=np.float64) if bias is None else np.asarray(bias,dtype=np.float64)
        if b.shape!=(W.shape[0],):
            raise ValueError("bias shape mismatch")
        response=W*h[None,:]
        rs=np.maximum(np.max(np.abs(response),axis=1)/response_qmax,1e-30)
        rq=np.rint(response/rs[:,None]).clip(-response_qmax,response_qmax).astype(np.int32).T
        return cls(W,b,h,rq,rs)

    @property
    def input_width(self)->int:
        return int(self.weight.shape[1])

    @property
    def output_width(self)->int:
        return int(self.weight.shape[0])

    def initialize(self,x:np.ndarray)->np.ndarray:
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(self.input_width,):
            raise ValueError("input shape mismatch")
        self.anchor=x.copy()
        self.codes=np.zeros(self.input_width,dtype=np.int64)
        self.accumulator=np.zeros(self.output_width,dtype=np.int64)
        self.base_output=self.weight@x+self.bias
        return self.base_output.copy()

    @property
    def reconstructed_input(self)->np.ndarray:
        if self.anchor is None or self.codes is None:
            raise ValueError("state not initialized")
        return self.anchor+self.step*self.codes

    def update(self,x:np.ndarray)->tuple[np.ndarray,int]:
        x=np.asarray(x,dtype=np.float64)
        if self.anchor is None:
            return self.initialize(x),self.input_width
        assert self.codes is not None and self.accumulator is not None and self.base_output is not None
        residual=x-self.reconstructed_input
        inc=np.rint(residual/self.step).astype(np.int64)
        dirty=np.flatnonzero(inc)
        if len(dirty):
            self.codes[dirty]+=inc[dirty]
            self.accumulator += inc[dirty]@self.response_q[dirty].astype(np.int64)
        y=self.base_output+self.response_scale*self.accumulator
        return y,int(len(dirty))

    def compiled_full(self,x:np.ndarray)->np.ndarray:
        """Evaluate the same anchored sigma-delta semantics without mutating state."""
        if self.anchor is None:
            raise ValueError("state not initialized")
        assert self.base_output is not None
        code=np.rint((np.asarray(x,dtype=np.float64)-self.anchor)/self.step).astype(np.int64)
        acc=code@self.response_q.astype(np.int64)
        return self.base_output+self.response_scale*acc

    def rebase(self,x:np.ndarray)->np.ndarray:
        return self.initialize(x)

    def input_error(self,x:np.ndarray)->float:
        return float(np.linalg.norm(np.asarray(x)-self.reconstructed_input))
