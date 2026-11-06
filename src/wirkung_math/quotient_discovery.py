from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
from .consumer_ir import EvidenceLevel


def _freeze_signature(value):
    if isinstance(value,np.ndarray):
        return tuple(_freeze_signature(x) for x in value.tolist())
    if isinstance(value,(list,tuple)):
        return tuple(_freeze_signature(x) for x in value)
    if isinstance(value,dict):
        return tuple(sorted((k,_freeze_signature(v)) for k,v in value.items()))
    if isinstance(value,np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class EmpiricalQuotient:
    samples:int
    state_dimension:int
    distinct_classes:int
    compression_ratio:float
    signature_entropy_bits:float
    changed_transitions:int
    distinct_transitions:int
    transition_reuse_fraction:float
    evidence:EvidenceLevel=EvidenceLevel.VALIDATED


def discover_empirical_quotient(
    states:np.ndarray,signatures
)->EmpiricalQuotient:
    X=np.asarray(states)
    if X.ndim!=2:
        raise ValueError("states must be [samples,state_dimension]")
    frozen=[_freeze_signature(x) for x in signatures]
    if len(frozen)!=len(X):
        raise ValueError("signature count must match states")
    n=len(frozen)
    counts=Counter(frozen)
    entropy=0.0
    if n:
        for count in counts.values():
            p=count/n
            entropy-=p*math.log2(p)
    transitions=list(zip(frozen[:-1],frozen[1:]))
    changed=sum(a!=b for a,b in transitions)
    distinct=len(set(transitions))
    reuse=0.0 if not transitions else 1.0-distinct/len(transitions)
    return EmpiricalQuotient(
        samples=n,
        state_dimension=int(X.shape[1]),
        distinct_classes=len(counts),
        compression_ratio=(n/max(1,len(counts))),
        signature_entropy_bits=float(entropy),
        changed_transitions=int(changed),
        distinct_transitions=int(distinct),
        transition_reuse_fraction=float(reuse),
    )
