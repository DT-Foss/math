from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TemporalRMSNormMultiLinearState:
    """
    Exact fused RMSNorm feeding multiple static linear forms (Q/K/V/router or
    gate/up projections) from one sparse temporal input edit.

    One norm-statistic update is shared. Every projection reads only changed
    columns and rescales its previous output.
    """
    weights:dict[str,np.ndarray]
    gamma:np.ndarray
    x:np.ndarray
    outputs:dict[str,np.ndarray]
    rho:float
    eps:float=1e-6

    @classmethod
    def from_input(
        cls,weights:dict[str,np.ndarray],gamma:np.ndarray,x:np.ndarray,eps:float=1e-6
    )->"TemporalRMSNormMultiLinearState":
        if not weights:
            raise ValueError("at least one weight is required")
        g=np.asarray(gamma)
        x=np.asarray(x).copy()
        normalized_width=len(x)
        checked={}
        for name,W0 in weights.items():
            W=np.asarray(W0)
            if W.ndim!=2 or W.shape[1]!=normalized_width:
                raise ValueError(f"weight {name!r} shape mismatch")
            checked[name]=W
        if g.shape!=x.shape:
            raise ValueError("gamma shape mismatch")
        rho=float(np.sqrt(np.mean(x*x)+eps))
        normed=(x/rho)*g
        outputs={name:W@normed for name,W in checked.items()}
        return cls(checked,g,x,outputs,rho,float(eps))

    def update_sparse(
        self,indices:np.ndarray,new_values:np.ndarray
    )->dict[str,np.ndarray]:
        idx=np.asarray(indices,dtype=np.int64)
        new=np.asarray(new_values)
        if new.shape!=(len(idx),):
            raise ValueError("new_values shape mismatch")
        old=self.x[idx].copy();delta=new-old
        d=len(self.x)
        rho_new=float(np.sqrt(
            self.rho*self.rho+(2.0*np.dot(old,delta)+np.dot(delta,delta))/d
        ))
        correction=(self.gamma[idx]*delta)/rho_new
        scale=self.rho/rho_new
        for name,W in self.weights.items():
            self.outputs[name]=scale*self.outputs[name]+W[:,idx]@correction
        self.x[idx]=new
        self.rho=rho_new
        return self.outputs

    def full_reference(self)->dict[str,np.ndarray]:
        normed=(self.x/self.rho)*self.gamma
        return {name:W@normed for name,W in self.weights.items()}
