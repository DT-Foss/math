from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def silu(x:np.ndarray)->np.ndarray:
    x=np.asarray(x)
    return x/(1.0+np.exp(-x))


def nearest_codes(values:np.ndarray,levels:np.ndarray)->np.ndarray:
    v=np.asarray(values)
    q=np.asarray(levels)
    if q.ndim!=1:
        raise ValueError("levels must be one-dimensional")
    return np.argmin(np.abs(v[:,None]-q[None,:]),axis=1)


@dataclass
class QuantizedSwiGLUDownState:
    """
    Temporal SwiGLU followed by a down projection, exact in the supplied
    activation-code arithmetic.

    Gate/up values are updated, the new SwiGLU vector is quantized, and only
    code positions that actually change cause down-projection weight reads.
    This turns dense floating drift into exact code sparsity.
    """
    down_weight:np.ndarray
    levels:np.ndarray
    gate:np.ndarray
    up:np.ndarray
    codes:np.ndarray
    down_output:np.ndarray

    @classmethod
    def from_gate_up(
        cls,down_weight:np.ndarray,levels:np.ndarray,gate:np.ndarray,up:np.ndarray
    )->"QuantizedSwiGLUDownState":
        W=np.asarray(down_weight)
        levels=np.asarray(levels)
        gate=np.asarray(gate).copy();up=np.asarray(up).copy()
        if gate.shape!=up.shape or W.ndim!=2 or W.shape[1]!=len(gate):
            raise ValueError("shape mismatch")
        hidden=silu(gate)*up
        codes=nearest_codes(hidden,levels)
        decoded=levels[codes]
        return cls(W,levels,gate,up,codes,W@decoded)

    def update(
        self,gate_delta:np.ndarray,up_delta:np.ndarray
    )->tuple[np.ndarray,np.ndarray]:
        dg=np.asarray(gate_delta);du=np.asarray(up_delta)
        if dg.shape!=self.gate.shape or du.shape!=self.up.shape:
            raise ValueError("delta shape mismatch")
        old_decoded=self.levels[self.codes]
        self.gate+=dg;self.up+=du
        new_hidden=silu(self.gate)*self.up
        new_codes=nearest_codes(new_hidden,self.levels)
        changed=np.flatnonzero(new_codes!=self.codes)
        if len(changed):
            new_decoded=self.levels[new_codes[changed]]
            delta=new_decoded-old_decoded[changed]
            self.down_output=self.down_output+self.down_weight[:,changed]@delta
            self.codes[changed]=new_codes[changed]
        return self.down_output,changed

    def full_reference(self)->np.ndarray:
        hidden=silu(self.gate)*self.up
        return self.down_weight@self.levels[nearest_codes(hidden,self.levels)]
