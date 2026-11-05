from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Protocol
import numpy as np


class DeltaLinearOperator(Protocol):
    """Interface implemented by dense, Origami, HCGM or fx7 weight codecs."""
    input_width:int
    output_width:int

    def full(self,x:np.ndarray)->np.ndarray: ...
    def columns(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray: ...
    def basis_image(self,basis:np.ndarray)->np.ndarray: ...


@dataclass(frozen=True)
class DenseDeltaLinearOperator:
    weight:np.ndarray

    @property
    def input_width(self)->int:
        return int(self.weight.shape[1])

    @property
    def output_width(self)->int:
        return int(self.weight.shape[0])

    def full(self,x:np.ndarray)->np.ndarray:
        return self.weight@np.asarray(x)

    def columns(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        return self.weight[:,idx]@np.asarray(delta_values)

    def basis_image(self,basis:np.ndarray)->np.ndarray:
        return self.weight@np.asarray(basis)


@dataclass(frozen=True)
class CallbackDeltaLinearOperator:
    """
    Zero-copy bridge for a compressed runtime.

    Beast supplies native callbacks that operate directly on Origami/fx7/HCGM
    pages. Wirkung never asks for a decoded dense weight matrix.
    """
    input_width:int
    output_width:int
    full_callback:Callable[[np.ndarray],np.ndarray]
    column_callback:Callable[[np.ndarray,np.ndarray],np.ndarray]
    basis_callback:Callable[[np.ndarray],np.ndarray]

    def full(self,x:np.ndarray)->np.ndarray:
        return np.asarray(self.full_callback(np.asarray(x)))

    def columns(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        return np.asarray(self.column_callback(
            np.asarray(indices,dtype=np.int64),np.asarray(delta_values)
        ))

    def basis_image(self,basis:np.ndarray)->np.ndarray:
        return np.asarray(self.basis_callback(np.asarray(basis)))


@dataclass
class TemporalOperatorState:
    operator:DeltaLinearOperator
    x:np.ndarray
    y:np.ndarray

    @classmethod
    def from_input(cls,operator:DeltaLinearOperator,x:np.ndarray)->"TemporalOperatorState":
        x=np.asarray(x).copy()
        if x.shape!=(operator.input_width,):
            raise ValueError("shape mismatch")
        return cls(operator,x,operator.full(x))

    def update_delta(self,indices:np.ndarray,delta_values:np.ndarray)->np.ndarray:
        idx=np.asarray(indices,dtype=np.int64)
        delta=np.asarray(delta_values)
        self.y=self.y+self.operator.columns(idx,delta)
        self.x[idx]+=delta
        return self.y


@dataclass
class BasisOperatorState:
    operator:DeltaLinearOperator
    basis_image:np.ndarray
    coefficients:np.ndarray
    y:np.ndarray

    @classmethod
    def create(
        cls,operator:DeltaLinearOperator,basis:np.ndarray,
        coefficients:np.ndarray,initial_x:np.ndarray,
    )->"BasisOperatorState":
        B=np.asarray(basis);c=np.asarray(coefficients).copy();x=np.asarray(initial_x)
        if B.shape!=(operator.input_width,len(c)) or x.shape!=(operator.input_width,):
            raise ValueError("shape mismatch")
        return cls(operator,operator.basis_image(B),c,operator.full(x))

    def update_coefficients(self,delta:np.ndarray)->np.ndarray:
        dc=np.asarray(delta)
        if dc.shape!=self.coefficients.shape:
            raise ValueError("shape mismatch")
        self.y=self.y+self.basis_image@dc
        self.coefficients+=dc
        return self.y
