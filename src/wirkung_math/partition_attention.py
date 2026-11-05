from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CompiledPartitionAttention:
    """
    Exact normalized weighted sum for a low-dimensional discrete score delta.

    Let base scores be s_i and let every future query in one temporal regime
    change them by

        delta s_i = signature_i^T alpha.

    Group equal signatures c and store

        H_c = sum_{i:signature_i=c} exp(s_i-m)
        M_c = sum_{i:signature_i=c} exp(s_i-m) v_i.

    Then a query needs only occupied signature groups:

        Z(alpha) = sum_c exp(c^T alpha) H_c
        N(alpha) = sum_c exp(c^T alpha) M_c.

    This is exact relative to the supplied discrete/quantized score signatures.
    Context length disappears from query cost once many keys share signatures.
    """
    signatures:np.ndarray
    mass:np.ndarray
    weighted_values:np.ndarray
    base_shift:float
    token_count:int
    signature_to_group:dict[tuple,int]

    @classmethod
    def compile(
        cls,base_scores:np.ndarray,score_signatures:np.ndarray,values:np.ndarray
    )->"CompiledPartitionAttention":
        s=np.asarray(base_scores,dtype=np.float64)
        C=np.asarray(score_signatures)
        V=np.asarray(values,dtype=np.float64)
        if s.ndim!=1 or C.ndim!=2 or V.ndim!=2 or len(s)!=len(C) or len(s)!=len(V):
            raise ValueError("shape mismatch")
        unique,inverse=np.unique(C,axis=0,return_inverse=True)
        shift=float(np.max(s)) if len(s) else 0.0
        w=np.exp(s-shift)
        mass=np.bincount(inverse,weights=w,minlength=len(unique)).astype(np.float64)
        M=np.zeros((len(unique),V.shape[1]),dtype=np.float64)
        np.add.at(M,inverse,w[:,None]*V)
        mapping={tuple(row.tolist()):i for i,row in enumerate(unique)}
        return cls(unique,mass,M,shift,int(len(s)),mapping)

    @property
    def occupied_groups(self)->int:
        return int(len(self.signatures))

    @property
    def compression_ratio(self)->float:
        return self.token_count/max(1,self.occupied_groups)

    @property
    def signature_width(self)->int:
        return int(self.signatures.shape[1])

    def query(self,alpha:np.ndarray)->tuple[np.ndarray,float]:
        a=np.asarray(alpha,dtype=np.float64)
        if a.shape!=(self.signature_width,):
            raise ValueError("alpha shape mismatch")
        z=self.signatures@a
        # Include group mass in the stability shift. Empty groups never exist.
        logmass=np.log(self.mass)+z
        shift=float(np.max(logmass))
        factor=np.exp(z-shift)
        denominator=float(factor@self.mass)
        numerator=factor@self.weighted_values
        output=numerator/denominator
        log_partition=self.base_shift+shift+float(np.log(denominator*np.exp(-shift)))
        # Algebraically simpler and more stable expression:
        log_partition=self.base_shift+shift+float(np.log(np.sum(np.exp(logmass-shift))))
        return output,log_partition

    def append(
        self,base_score:float,signature:np.ndarray,value:np.ndarray
    )->None:
        sig=np.asarray(signature)
        val=np.asarray(value,dtype=np.float64)
        if sig.shape!=(self.signature_width,) or val.shape!=(self.weighted_values.shape[1],):
            raise ValueError("append shape mismatch")

        score=float(base_score)
        if score>self.base_shift:
            scale=float(np.exp(self.base_shift-score))
            self.mass*=scale
            self.weighted_values*=scale
            self.base_shift=score
        weight=float(np.exp(score-self.base_shift))
        key=tuple(sig.tolist())
        group=self.signature_to_group.get(key)
        if group is None:
            group=len(self.signatures)
            self.signatures=np.concatenate([self.signatures,sig[None,:]],axis=0)
            self.mass=np.concatenate([self.mass,np.array([weight])])
            self.weighted_values=np.concatenate(
                [self.weighted_values,(weight*val)[None,:]],axis=0
            )
            self.signature_to_group[key]=group
        else:
            self.mass[group]+=weight
            self.weighted_values[group]+=weight*val
        self.token_count+=1

    def dense_reference(
        self,base_scores:np.ndarray,score_signatures:np.ndarray,
        values:np.ndarray,alpha:np.ndarray,
    )->np.ndarray:
        s=np.asarray(base_scores,dtype=np.float64)+np.asarray(score_signatures)@np.asarray(alpha,dtype=np.float64)
        V=np.asarray(values,dtype=np.float64)
        m=float(np.max(s))
        w=np.exp(s-m)
        return (w@V)/w.sum()


@dataclass
class PartitionAttentionBank:
    """Compiled partitions for the temporal delta-support patterns seen most often."""
    partitions:dict[tuple[int,...],CompiledPartitionAttention]

    def query(self,support:tuple[int,...],alpha:np.ndarray)->tuple[np.ndarray,float]:
        try:
            part=self.partitions[tuple(support)]
        except KeyError as exc:
            raise KeyError(f"uncompiled temporal support {support}") from exc
        return part.query(alpha)

    def best_supports(self)->tuple[tuple[int,...],...]:
        return tuple(sorted(
            self.partitions,
            key=lambda s:self.partitions[s].compression_ratio,
            reverse=True,
        ))
