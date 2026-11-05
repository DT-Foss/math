from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TemporalResponseBasis:
    """
    Distribution-compiled low-rank image of a static linear operator.

    For temporal input delta d and weight W, learn an orthonormal output basis U
    from observed response deltas Wd. Precompute encoder A = W^T U. Then

        W d ≈ U (A^T d)

    with cost r*(input+output), while the exact residual remains a separately
    routable state. This is a temporal operator approximation, not a low-rank
    approximation of W over arbitrary inputs.
    """
    output_basis:np.ndarray  # [output,rank]
    encoder:np.ndarray  # [input,rank] = W.T @ output_basis
    singular_values:np.ndarray

    @classmethod
    def fit(
        cls,
        weight:np.ndarray,
        input_deltas:np.ndarray,
        *,
        rank:int,
    )->"TemporalResponseBasis":
        W=np.asarray(weight,dtype=np.float64)
        D=np.asarray(input_deltas,dtype=np.float64)
        if W.ndim!=2 or D.ndim!=2 or D.shape[1]!=W.shape[1] or not 0<rank<=min(D.shape[0],W.shape[0]):
            raise ValueError("shape/rank mismatch")
        response=D@W.T
        _,s,Vh=np.linalg.svd(response,full_matrices=False)
        U=Vh[:rank].T
        return cls(U,W.T@U,s[:rank])

    @property
    def input_width(self)->int:
        return int(self.encoder.shape[0])

    @property
    def output_width(self)->int:
        return int(self.output_basis.shape[0])

    @property
    def rank(self)->int:
        return int(self.output_basis.shape[1])

    @property
    def storage_ratio_to_dense_weight(self)->float:
        return (self.output_basis.size+self.encoder.size)/(self.input_width*self.output_width)

    @property
    def theoretical_multiply_speedup(self)->float:
        full=self.input_width*self.output_width
        lowrank=self.rank*(self.input_width+self.output_width)
        return full/lowrank

    def apply_delta(self,delta:np.ndarray)->np.ndarray:
        d=np.asarray(delta,dtype=np.float64)
        if d.shape!=(self.input_width,):
            raise ValueError("delta shape mismatch")
        return self.output_basis@(self.encoder.T@d)

    def apply_batch(self,deltas:np.ndarray)->np.ndarray:
        D=np.asarray(deltas,dtype=np.float64)
        if D.ndim!=2 or D.shape[1]!=self.input_width:
            raise ValueError("delta matrix shape mismatch")
        return (D@self.encoder)@self.output_basis.T

    def response_capture(self,weight:np.ndarray,deltas:np.ndarray)->float:
        W=np.asarray(weight,dtype=np.float64);D=np.asarray(deltas,dtype=np.float64)
        exact=D@W.T;approx=self.apply_batch(D)
        denom=np.sum(exact*exact)
        return 1.0 if denom==0 else 1.0-float(np.sum((exact-approx)**2))/float(denom)


@dataclass
class TemporalResponseState:
    basis:TemporalResponseBasis
    y:np.ndarray

    def update(self,delta:np.ndarray)->np.ndarray:
        self.y=self.y+self.basis.apply_delta(delta)
        return self.y


@dataclass(frozen=True)
class ResponseResidual:
    approximate:np.ndarray
    residual:np.ndarray
    residual_fraction:float


def split_response_delta(
    basis:TemporalResponseBasis,
    weight:np.ndarray,
    delta:np.ndarray,
)->ResponseResidual:
    W=np.asarray(weight,dtype=np.float64);d=np.asarray(delta,dtype=np.float64)
    exact=W@d;approx=basis.apply_delta(d);residual=exact-approx
    denom=np.linalg.norm(exact)
    return ResponseResidual(
        approx,residual,0.0 if denom==0 else float(np.linalg.norm(residual)/denom)
    )
