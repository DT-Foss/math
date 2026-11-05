from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .delta_layout import ColumnPackedLinearOperator


@dataclass(frozen=True)
class TemporalEventFrame:
    """Changed coordinates and exact decoded-value deltas for one state step."""
    indices:np.ndarray
    delta_values:np.ndarray
    width:int

    @property
    def count(self)->int:
        return int(len(self.indices))

    @property
    def fraction(self)->float:
        return self.count/max(1,self.width)


@dataclass
class HystereticActivationCodec:
    """
    Stateful Schmitt-trigger activation code.

    Every coordinate keeps its previous decoded level. A new code is emitted
    only when the floating activation leaves a deadband around that persistent
    value. The emitted event contains the exact change in decoded arithmetic,
    so downstream delta operators never scan a dense vector for differences.

    This optimizes temporal event rate rather than framewise reconstruction
    error. `deadband_steps=0.5` recovers ordinary nearest-level quantization;
    larger values trade bounded state error for fewer state transitions.
    """
    scale:np.ndarray
    codes:np.ndarray
    bits:int
    deadband_steps:float

    @classmethod
    def fit(
        cls,
        trace:np.ndarray,
        *,
        bits:int=8,
        deadband_steps:float=1.0,
        quantile:float=1.0,
    )->"HystereticActivationCodec":
        x=np.asarray(trace,dtype=np.float64)
        if x.ndim!=2 or len(x)==0 or not 0.0<quantile<=1.0:
            raise ValueError("trace must be [time,width] and quantile in (0,1]")
        qmax=(1<<(bits-1))-1
        magnitude=np.quantile(np.abs(x),quantile,axis=0)
        tiny=np.finfo(np.float64).tiny
        scale=np.maximum(magnitude/max(1,qmax),tiny)
        initial=np.rint(x[0]/scale).clip(-qmax,qmax).astype(np.int16)
        return cls(scale,initial,bits,float(deadband_steps))

    @property
    def qmin(self)->int:
        return -(1<<(self.bits-1))+1

    @property
    def qmax(self)->int:
        return (1<<(self.bits-1))-1

    @property
    def decoded(self)->np.ndarray:
        return self.codes.astype(np.float64)*self.scale

    def encode(self,value:np.ndarray)->TemporalEventFrame:
        x=np.asarray(value,dtype=np.float64)
        if x.shape!=self.scale.shape:
            raise ValueError("value shape mismatch")
        previous_decoded=self.decoded
        outside=np.abs(x-previous_decoded)>=self.deadband_steps*self.scale
        candidates=np.flatnonzero(outside)
        if not len(candidates):
            return TemporalEventFrame(
                np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float64),len(x)
            )
        new_codes=np.rint(x[candidates]/self.scale[candidates]).clip(
            self.qmin,self.qmax
        ).astype(np.int16)
        changed_mask=new_codes!=self.codes[candidates]
        indices=candidates[changed_mask]
        if not len(indices):
            return TemporalEventFrame(
                np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float64),len(x)
            )
        old_codes=self.codes[indices].copy()
        self.codes[indices]=new_codes[changed_mask]
        delta=(self.codes[indices].astype(np.float64)-old_codes.astype(np.float64))*self.scale[indices]
        return TemporalEventFrame(indices.astype(np.int64),delta,len(x))

    def replay(self,trace:np.ndarray)->tuple[np.ndarray,list[TemporalEventFrame]]:
        x=np.asarray(trace,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.scale):
            raise ValueError("trace shape mismatch")
        decoded=[self.decoded.copy()]
        events=[]
        for row in x[1:]:
            event=self.encode(row)
            events.append(event)
            decoded.append(self.decoded.copy())
        return np.stack(decoded),events


@dataclass
class EventLinearState:
    """Linear state that consumes first-class temporal events without detection."""
    operator:ColumnPackedLinearOperator
    y:np.ndarray

    @classmethod
    def from_decoded(
        cls,operator:ColumnPackedLinearOperator,decoded:np.ndarray
    )->"EventLinearState":
        x=np.asarray(decoded)
        if x.shape!=(operator.input_width,):
            raise ValueError("shape mismatch")
        return cls(operator,operator.full(x))

    def apply(self,event:TemporalEventFrame)->np.ndarray:
        if event.width!=self.operator.input_width:
            raise ValueError("event/operator width mismatch")
        if event.count:
            self.y=self.y+self.operator.delta(event.indices,event.delta_values)
        return self.y


def temporal_rate_distortion(
    trace:np.ndarray,
    *,
    bits:int,
    deadband_steps:float,
    quantile:float=1.0,
)->dict:
    x=np.asarray(trace,dtype=np.float64)
    codec=HystereticActivationCodec.fit(
        x,bits=bits,deadband_steps=deadband_steps,quantile=quantile
    )
    decoded,events=codec.replay(x)
    counts=np.array([e.count for e in events],dtype=np.float64)
    error=np.linalg.norm(decoded-x)/max(np.linalg.norm(x),np.finfo(np.float64).tiny)
    max_error=np.max(np.abs(decoded-x))
    return {
        "bits":int(bits),
        "deadband_steps":float(deadband_steps),
        "quantile":float(quantile),
        "width":int(x.shape[1]),
        "transitions":int(max(0,len(x)-1)),
        "event_count_median":float(np.median(counts)) if len(counts) else 0.0,
        "event_count_p90":float(np.percentile(counts,90)) if len(counts) else 0.0,
        "event_fraction_median":float(np.median(counts)/x.shape[1]) if len(counts) else 0.0,
        "event_fraction_p90":float(np.percentile(counts,90)/x.shape[1]) if len(counts) else 0.0,
        "relative_l2_error":float(error),
        "max_abs_error":float(max_error),
        "decoded":decoded,
        "events":events,
        "scale":codec.scale.copy(),
    }
