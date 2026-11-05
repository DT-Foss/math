from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ColumnPackedLinearOperator:
    """
    Delta-native layout: input-direction pages are contiguous.

    A conventional row-major W is ideal for full GEMV but makes W[:,U] a
    strided gather. Packing W.T makes every changed input direction a contiguous
    output vector, so sparse temporal updates become tiny AXPY/GEMV operations.
    Origami/HCGM codecs should expose the same direction-major interface without
    duplicating a dense matrix.
    """
    columns:np.ndarray

    @classmethod
    def from_weight(cls,weight:np.ndarray)->"ColumnPackedLinearOperator":
        W=np.asarray(weight)
        if W.ndim!=2:
            raise ValueError("weight must be a matrix")
        return cls(np.ascontiguousarray(W.T))

    @property
    def input_width(self)->int:
        return int(self.columns.shape[0])

    @property
    def output_width(self)->int:
        return int(self.columns.shape[1])

    def full(self,x:np.ndarray)->np.ndarray:
        return np.asarray(x)@self.columns

    def delta(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        dv=np.asarray(delta_values)
        if dv.shape!=(len(idx),):
            raise ValueError("shape mismatch")
        return dv@self.columns[idx]

    def basis_image(self,basis:np.ndarray)->np.ndarray:
        return np.asarray(basis).T@self.columns


@dataclass
class ColumnPackedTemporalLinearState:
    operator:ColumnPackedLinearOperator
    x:np.ndarray
    y:np.ndarray

    @classmethod
    def from_input(
        cls,operator:ColumnPackedLinearOperator,x:np.ndarray
    )->"ColumnPackedTemporalLinearState":
        x=np.asarray(x).copy()
        if x.shape!=(operator.input_width,):
            raise ValueError("shape mismatch")
        return cls(operator,x,operator.full(x))

    def update_delta(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        delta=np.asarray(delta_values)
        self.y=self.y+self.operator.delta(idx,delta)
        self.x[idx]+=delta
        return self.y


@dataclass
class ColumnPackedRMSNormLinearState:
    """Exact RMSNorm->Linear temporal state using direction-major weight pages."""
    operator:ColumnPackedLinearOperator
    gamma:np.ndarray
    x:np.ndarray
    projected:np.ndarray
    rho:float
    eps:float=1e-6

    @classmethod
    def from_input(
        cls,operator:ColumnPackedLinearOperator,gamma:np.ndarray,x:np.ndarray,eps:float=1e-6
    )->"ColumnPackedRMSNormLinearState":
        x=np.asarray(x).copy();g=np.asarray(gamma)
        if x.shape!=(operator.input_width,) or g.shape!=x.shape:
            raise ValueError("shape mismatch")
        rho=float(np.sqrt(np.mean(x*x)+eps))
        return cls(operator,g,x,operator.full((x/rho)*g),rho,float(eps))

    def update_sparse(self,indices:np.ndarray,new_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64);new=np.asarray(new_values)
        if new.shape!=(len(idx),):
            raise ValueError("shape mismatch")
        old=self.x[idx].copy();delta=new-old;d=len(self.x)
        rho_new=float(np.sqrt(
            self.rho*self.rho+(2.0*np.dot(old,delta)+np.dot(delta,delta))/d
        ))
        correction=(self.gamma[idx]*delta)/rho_new
        self.projected=(self.rho/rho_new)*self.projected+self.operator.delta(idx,correction)
        self.x[idx]=new;self.rho=rho_new
        return self.projected


@dataclass
class ColumnPackedRMSNormMultiLinearState:
    operators:dict[str,ColumnPackedLinearOperator]
    gamma:np.ndarray
    x:np.ndarray
    outputs:dict[str,np.ndarray]
    rho:float
    eps:float=1e-6

    @classmethod
    def from_input(
        cls,operators:dict[str,ColumnPackedLinearOperator],gamma:np.ndarray,
        x:np.ndarray,eps:float=1e-6,
    )->"ColumnPackedRMSNormMultiLinearState":
        if not operators:
            raise ValueError("operators required")
        x=np.asarray(x).copy();g=np.asarray(gamma)
        if g.shape!=x.shape or any(op.input_width!=len(x) for op in operators.values()):
            raise ValueError("shape mismatch")
        rho=float(np.sqrt(np.mean(x*x)+eps));normed=(x/rho)*g
        return cls(operators,g,x,{k:op.full(normed) for k,op in operators.items()},rho,float(eps))

    def update_sparse(self,indices:np.ndarray,new_values:np.ndarray)->dict[str,np.ndarray]:
        idx=np.asarray(indices,dtype=np.int64);new=np.asarray(new_values)
        old=self.x[idx].copy();delta=new-old;d=len(self.x)
        rho_new=float(np.sqrt(
            self.rho*self.rho+(2.0*np.dot(old,delta)+np.dot(delta,delta))/d
        ))
        correction=(self.gamma[idx]*delta)/rho_new;scale=self.rho/rho_new
        for name,op in self.operators.items():
            self.outputs[name]=scale*self.outputs[name]+op.delta(idx,correction)
        self.x[idx]=new;self.rho=rho_new
        return self.outputs
