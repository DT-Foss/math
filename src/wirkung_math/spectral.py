from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SparseDFTState:
    """Cached DFT updated from point replacements; Fourier basis is generated, not stored."""
    signal: np.ndarray
    spectrum: np.ndarray

    @classmethod
    def from_signal(cls, x: np.ndarray) -> "SparseDFTState":
        x=np.asarray(x,dtype=np.complex128).copy()
        return cls(x,np.fft.fft(x))

    def replace(self, indices: np.ndarray, new_values: np.ndarray) -> None:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values,dtype=np.complex128)
        delta=new-self.signal[idx]
        k=np.arange(len(self.signal),dtype=np.float64)
        for u,dv in zip(idx,delta):
            self.spectrum += dv*np.exp((-2j*np.pi*int(u)/len(self.signal))*k)
        self.signal[idx]=new


def bit_reverse_indices(n: int) -> np.ndarray:
    if n <= 0 or n&(n-1):
        raise ValueError("n must be a positive power of two")
    bits=int(math.log2(n))
    x=np.arange(n,dtype=np.uint64)
    rev=np.zeros(n,dtype=np.uint64)
    for _ in range(bits):
        rev=(rev<<1)|(x&1)
        x >>= 1
    return rev.astype(np.int64)


class ButterflyFFTState:
    """
    Exact reference implementation of a dynamic radix-2 FFT DAG.

    A sparse input edit updates only butterflies reachable from dirty leaves.
    Multiple edits share downstream butterflies. Work in updated DAG nodes is
    O(N*(1+log e)) rather than O(e*N) once paths overlap, at the cost of storing
    stage checkpoints. This Python implementation is a correctness reference,
    not the intended microkernel.
    """
    def __init__(self, x: np.ndarray):
        x=np.asarray(x,dtype=np.complex128)
        self.n=len(x)
        self.bitrev=bit_reverse_indices(self.n)
        self.signal=x.copy()
        self.stages=[self.signal[self.bitrev].copy()]
        m=2
        while m <= self.n:
            prev=self.stages[-1]
            cur=prev.copy()
            h=m//2
            tw=np.exp(-2j*np.pi*np.arange(h)/m)
            for b in range(0,self.n,m):
                a=prev[b:b+h]
                v=prev[b+h:b+m]*tw
                cur[b:b+h]=a+v
                cur[b+h:b+m]=a-v
            self.stages.append(cur)
            m*=2

    @property
    def spectrum(self) -> np.ndarray:
        return self.stages[-1]

    def replace(self, indices: np.ndarray, new_values: np.ndarray) -> int:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values,dtype=np.complex128)
        self.signal[idx]=new
        dirty=np.unique(self.bitrev[idx])
        self.stages[0][self.bitrev[idx]]=new
        updated_nodes=len(dirty)
        m=2
        for s in range(1,len(self.stages)):
            prev=self.stages[s-1]
            cur=self.stages[s]
            h=m//2
            off=dirty % m
            lower=np.unique(np.where(off<h,dirty,dirty-h))
            partner=lower+h
            j=lower % m
            tw=np.exp(-2j*np.pi*j/m)
            a=prev[lower]
            v=prev[partner]*tw
            cur[lower]=a+v
            cur[partner]=a-v
            dirty=np.unique(np.concatenate([lower,partner]))
            updated_nodes += len(dirty)
            m*=2
        return int(updated_nodes)
