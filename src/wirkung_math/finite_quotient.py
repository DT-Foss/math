from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _freeze(value):
    if isinstance(value,np.ndarray):
        return tuple(_freeze(x) for x in value.tolist())
    if isinstance(value,(list,tuple)):
        return tuple(_freeze(x) for x in value)
    if isinstance(value,dict):
        return tuple(sorted((k,_freeze(v)) for k,v in value.items()))
    if isinstance(value,np.generic):
        return value.item()
    return value


def _partition(signatures)->np.ndarray:
    mapping={}
    classes=np.empty(len(signatures),dtype=np.int64)
    for i,sig in enumerate(signatures):
        cid=mapping.get(sig)
        if cid is None:
            cid=len(mapping)
            mapping[sig]=cid
        classes[i]=cid
    return classes


@dataclass(frozen=True)
class FiniteStateConsumerQuotient:
    class_of_state:np.ndarray
    reduced_transition:np.ndarray
    reduced_outputs:tuple
    representative_state:np.ndarray
    refinement_rounds:int
    proof_passed:bool

    @classmethod
    def fit(
        cls,transitions:np.ndarray,outputs
    )->"FiniteStateConsumerQuotient":
        T=np.asarray(transitions,dtype=np.int64)
        if T.ndim!=2:
            raise ValueError("transitions must be [states,actions]")
        n,actions=T.shape
        if n==0:
            raise ValueError("state machine must contain at least one state")
        if np.any(T<0) or np.any(T>=n):
            raise ValueError("transition points outside state space")
        if isinstance(outputs,np.ndarray):
            if len(outputs)!=n:
                raise ValueError("output count mismatch")
            raw_outputs=[outputs[i].copy() if outputs.ndim>1 else outputs[i].item() for i in range(n)]
        else:
            raw_outputs=list(outputs)
            if len(raw_outputs)!=n:
                raise ValueError("output count mismatch")
        frozen=[_freeze(x) for x in raw_outputs]
        classes=_partition(frozen)
        rounds=0
        while True:
            rounds+=1
            signatures=[
                (int(classes[s]),tuple(int(classes[T[s,a]]) for a in range(actions)))
                for s in range(n)
            ]
            refined=_partition(signatures)
            if np.array_equal(refined,classes):
                break
            classes=refined
            if rounds>n:
                raise RuntimeError("partition refinement did not converge")

        k=int(classes.max())+1
        reps=np.empty(k,dtype=np.int64)
        reduced_T=np.empty((k,actions),dtype=np.int64)
        reduced_out=[]
        proof=True
        for cid in range(k):
            members=np.flatnonzero(classes==cid)
            rep=int(members[0])
            reps[cid]=rep
            out_sig=frozen[rep]
            reduced_out.append(raw_outputs[rep])
            target_signature=classes[T[rep]]
            for s in members:
                if frozen[s]!=out_sig or not np.array_equal(classes[T[s]],target_signature):
                    proof=False
                    break
            reduced_T[cid]=target_signature

        return cls(
            class_of_state=classes,
            reduced_transition=reduced_T,
            reduced_outputs=tuple(reduced_out),
            representative_state=reps,
            refinement_rounds=int(rounds),
            proof_passed=bool(proof),
        )

    @property
    def class_count(self)->int:
        return int(len(self.representative_state))

    @property
    def state_count(self)->int:
        return int(len(self.class_of_state))

    @property
    def action_count(self)->int:
        return int(self.reduced_transition.shape[1])

    @property
    def state_compression(self)->float:
        return float(self.state_count/self.class_count)

    def step_class(self,state_class:int,action:int)->int:
        if not 0<=state_class<self.class_count:
            raise ValueError("state class out of range")
        if not 0<=action<self.action_count:
            raise ValueError("action out of range")
        return int(self.reduced_transition[state_class,action])

    def output_for_class(self,state_class:int):
        if not 0<=state_class<self.class_count:
            raise ValueError("state class out of range")
        return self.reduced_outputs[state_class]
