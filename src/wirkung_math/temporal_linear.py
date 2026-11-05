from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def sparse_linear_delta(
    weight: np.ndarray,
    indices: np.ndarray,
    delta_values: np.ndarray,
) -> np.ndarray:
    """Exact W[:,U] @ delta_x[U] without reading the other weight columns."""
    W=np.asarray(weight)
    idx=np.asarray(indices,dtype=np.int64)
    dv=np.asarray(delta_values)
    if idx.ndim!=1 or dv.shape!=(len(idx),):
        raise ValueError("indices/delta shape mismatch")
    return W[:,idx]@dv


@dataclass
class TemporalLinearState:
    """
    Exact stateful y_t = W x_t executor.

    The previous output is persistent. A new input is represented by the
    coordinates or blocks that changed, so unchanged weight columns are never
    read. Dense recomputation remains an explicit route, not the default.
    """
    weight: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_input(cls,weight:np.ndarray,x:np.ndarray)->"TemporalLinearState":
        W=np.asarray(weight)
        x=np.asarray(x).copy()
        if W.ndim!=2 or x.shape!=(W.shape[1],):
            raise ValueError("shape mismatch")
        return cls(W,x,W@x)

    def update_coordinates(self,indices:np.ndarray,new_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values)
        if new.shape!=(len(idx),):
            raise ValueError("new_values shape mismatch")
        delta=new-self.x[idx]
        self.y=self.y+sparse_linear_delta(self.weight,idx,delta)
        self.x[idx]=new
        return self.y

    def update_delta(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        delta=np.asarray(delta_values)
        self.y=self.y+sparse_linear_delta(self.weight,idx,delta)
        self.x[idx]+=delta
        return self.y

    def update_detect(
        self,new_x:np.ndarray,*,atol:float=0.0,max_sparse_fraction:float=0.25
    )->tuple[np.ndarray,str,int]:
        new=np.asarray(new_x)
        if new.shape!=self.x.shape:
            raise ValueError("shape mismatch")
        delta=new-self.x
        idx=np.flatnonzero(np.abs(delta)>atol)
        if len(idx)<=max_sparse_fraction*len(delta):
            self.y=self.y+self.weight[:,idx]@delta[idx]
            route="coordinate_delta"
        else:
            self.y=self.weight@new
            route="full"
        self.x=new.copy()
        return self.y,route,int(len(idx))


@dataclass
class BasisDeltaLinearState:
    """
    Exact linear state when temporal deltas arrive as

        delta_x = B delta_c + residual_delta.

    W B is precomputed once. The residual may itself be coordinate sparse. This
    is directly compatible with an Origami/HCGM shared activation basis.
    """
    weight: np.ndarray
    basis: np.ndarray
    weight_basis: np.ndarray
    y: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def create(
        cls,weight:np.ndarray,basis:np.ndarray,initial_x:np.ndarray,
        initial_coefficients:np.ndarray,
    )->"BasisDeltaLinearState":
        W=np.asarray(weight)
        B=np.asarray(basis)
        x=np.asarray(initial_x)
        c=np.asarray(initial_coefficients).copy()
        if W.ndim!=2 or B.shape[0]!=W.shape[1] or B.shape[1]!=len(c) or x.shape!=(W.shape[1],):
            raise ValueError("shape mismatch")
        return cls(W,B,W@B,W@x,c)

    def update(
        self,
        delta_coefficients:np.ndarray,
        residual_indices:np.ndarray|None=None,
        residual_delta:np.ndarray|None=None,
    )->np.ndarray:
        dc=np.asarray(delta_coefficients)
        if dc.shape!=self.coefficients.shape:
            raise ValueError("coefficient shape mismatch")
        dy=self.weight_basis@dc
        if residual_indices is not None:
            if residual_delta is None:
                raise ValueError("residual_delta is required")
            idx=np.asarray(residual_indices,dtype=np.int64)
            dr=np.asarray(residual_delta)
            if dr.shape!=(len(idx),):
                raise ValueError("residual shape mismatch")
            dy=dy+self.weight[:,idx]@dr
        self.coefficients+=dc
        self.y=self.y+dy
        return self.y


@dataclass
class BatchLowRankLinearState:
    """
    Exact shared-weight update for many lanes/tokens.

        delta_X = A B^T
        delta_Y = A (B^T W^T)

    If B lanes/speculative positions share rank r, one rank-r weight image
    replaces B independent matrix-vector passes.
    """
    weight:np.ndarray
    x:np.ndarray
    y:np.ndarray

    @classmethod
    def from_batch(cls,weight:np.ndarray,x:np.ndarray)->"BatchLowRankLinearState":
        W=np.asarray(weight)
        X=np.asarray(x).copy()
        if X.ndim!=2 or W.ndim!=2 or X.shape[1]!=W.shape[1]:
            raise ValueError("shape mismatch")
        return cls(W,X,X@W.T)

    def update_factors(self,left:np.ndarray,right:np.ndarray)->np.ndarray:
        A=np.asarray(left)
        B=np.asarray(right)
        if A.ndim!=2 or B.ndim!=2 or A.shape[0]!=self.x.shape[0] or B.shape[0]!=self.x.shape[1] or A.shape[1]!=B.shape[1]:
            raise ValueError("factor shape mismatch")
        self.y=self.y+A@(B.T@self.weight.T)
        self.x=self.x+A@B.T
        return self.y

    def operation_ratio(self,rank:int)->float:
        batch,din=self.x.shape
        dout=self.weight.shape[0]
        full=batch*din*dout
        factor=rank*din*dout+batch*rank*dout
        return full/factor


@dataclass
class TemporalRMSNormLinearState:
    """
    Exact fused RMSNorm -> Linear temporal update for a sparse input edit.

    RMSNorm does not have to destroy sparsity. If x' = x + delta_U:

      y' = (rho/rho') y + gamma_U*delta_U/rho'

    and therefore

      W y' = (rho/rho') W y + W[:,U](gamma_U*delta_U/rho').

    The full normalized vector and the unchanged weight columns never need to
    be materialized or read.
    """
    weight:np.ndarray
    gamma:np.ndarray
    x:np.ndarray
    projected:np.ndarray
    rho:float
    eps:float=1e-6

    @classmethod
    def from_input(
        cls,weight:np.ndarray,gamma:np.ndarray,x:np.ndarray,eps:float=1e-6
    )->"TemporalRMSNormLinearState":
        W=np.asarray(weight)
        g=np.asarray(gamma)
        x=np.asarray(x).copy()
        if W.ndim!=2 or x.shape!=(W.shape[1],) or g.shape!=x.shape:
            raise ValueError("shape mismatch")
        rho=float(np.sqrt(np.mean(x*x)+eps))
        projected=W@((x/rho)*g)
        return cls(W,g,x,projected,rho,float(eps))

    def update_sparse(self,indices:np.ndarray,new_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values)
        if new.shape!=(len(idx),):
            raise ValueError("new_values shape mismatch")
        old=self.x[idx].copy()
        delta=new-old
        d=len(self.x)
        rho2_new=self.rho*self.rho+(2.0*np.dot(old,delta)+np.dot(delta,delta))/d
        rho_new=float(np.sqrt(rho2_new))
        correction=(self.gamma[idx]*delta)/rho_new
        self.projected=(self.rho/rho_new)*self.projected+self.weight[:,idx]@correction
        self.x[idx]=new
        self.rho=rho_new
        return self.projected
