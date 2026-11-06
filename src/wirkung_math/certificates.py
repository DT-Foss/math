from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearOrderCertificate:
    weight:np.ndarray

    def __post_init__(self):
        W=np.asarray(self.weight,dtype=np.float64)
        if W.ndim!=2:
            raise ValueError("weight must be two-dimensional")
        object.__setattr__(self,"weight",W)

    def certify_argmax(
        self,logits:np.ndarray,winner:int,delta:np.ndarray
    )->bool:
        z=np.asarray(logits,dtype=np.float64)
        dx=np.asarray(delta,dtype=np.float64)
        if z.shape!=(self.weight.shape[0],) or dx.shape!=(self.weight.shape[1],):
            raise ValueError("shape mismatch")
        if not 0<=winner<len(z):
            raise ValueError("winner out of range")
        norm=float(np.linalg.norm(dx))
        diff=self.weight[winner][None,:]-self.weight
        bounds=np.linalg.norm(diff,axis=1)*norm
        margins=z[winner]-z
        mask=np.ones(len(z),dtype=bool);mask[winner]=False
        return bool(np.all(margins[mask]>bounds[mask]))

    def certify_topk(
        self,logits:np.ndarray,selected:np.ndarray,delta:np.ndarray
    )->bool:
        z=np.asarray(logits,dtype=np.float64)
        sel=np.unique(np.asarray(selected,dtype=np.int64))
        dx=np.asarray(delta,dtype=np.float64)
        if z.shape!=(self.weight.shape[0],) or dx.shape!=(self.weight.shape[1],):
            raise ValueError("shape mismatch")
        if len(sel)==0 or np.any(sel<0) or np.any(sel>=len(z)):
            raise ValueError("invalid selected set")
        outside=np.setdiff1d(np.arange(len(z)),sel,assume_unique=True)
        if len(outside)==0:
            return True
        norm=float(np.linalg.norm(dx))
        for i in sel:
            diffs=self.weight[i][None,:]-self.weight[outside]
            bounds=np.linalg.norm(diffs,axis=1)*norm
            margins=z[i]-z[outside]
            if np.any(margins<=bounds):
                return False
        return True


@dataclass(frozen=True)
class QuantizationCellCertificate:
    step:np.ndarray

    def __post_init__(self):
        step=np.asarray(self.step,dtype=np.float64)
        if step.ndim!=1 or np.any(step<=0):
            raise ValueError("step must be positive vector")
        object.__setattr__(self,"step",step)

    def stable(self,value:np.ndarray,delta_bound:np.ndarray)->bool:
        x=np.asarray(value,dtype=np.float64)
        b=np.asarray(delta_bound,dtype=np.float64)
        if x.shape!=self.step.shape or b.shape!=self.step.shape or np.any(b<0):
            raise ValueError("shape/bound mismatch")
        code=np.rint(x/self.step)
        center=code*self.step
        distance=.5*self.step-np.abs(x-center)
        return bool(np.all(b<distance))


@dataclass(frozen=True)
class ThresholdCertificate:
    threshold:float=0.0

    def stable(self,value:float,delta_bound:float)->bool:
        if delta_bound<0:
            raise ValueError("delta_bound must be nonnegative")
        return abs(float(value)-float(self.threshold))>float(delta_bound)
