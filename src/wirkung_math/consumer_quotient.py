from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConsumerQuotientCodebook:
    """
    Block state codebook trained in downstream response space.

    For y=W x and blocks x_b, store code responses approximating W_b x_b.
    Directions of x that do not affect the consumer require no state at all.
    The persistent representation is therefore a quotient by the consumer map,
    not a reconstruction of the upstream activation.
    """
    responses:np.ndarray  # [blocks,codes,output]
    distortion_scale:np.ndarray
    input_width:int
    block_size:int

    @classmethod
    def fit(
        cls,
        trace:np.ndarray,
        weight:np.ndarray,
        *,
        block_size:int=16,
        codes:int=16,
        iterations:int=20,
        seed:int=0,
    )->"ConsumerQuotientCodebook":
        x=np.asarray(trace,dtype=np.float64)
        W=np.asarray(weight,dtype=np.float64)
        if x.ndim!=2 or W.ndim!=2 or x.shape[1]!=W.shape[1] or x.shape[1]%block_size or codes>len(x):
            raise ValueError("shape/config mismatch")
        rng=np.random.default_rng(seed)
        T,D=x.shape;B=D//block_size;O=W.shape[0]
        blocks=x.reshape(T,B,block_size)
        response=np.empty((T,B,O),dtype=np.float64)
        for b in range(B):
            lo=b*block_size;hi=lo+block_size
            response[:,b]=blocks[:,b]@W[:,lo:hi].T
        centroids=np.empty((B,codes,O),dtype=np.float64)
        for b in range(B):
            centroids[b]=response[rng.choice(T,codes,replace=False),b]
        assignment=np.full((T,B),-1,dtype=np.int64)
        for _ in range(iterations):
            distance=((response[:,:,None,:]-centroids[None,:,:,:])**2).sum(-1)
            new=distance.argmin(-1)
            if np.array_equal(new,assignment):break
            assignment=new
            for b in range(B):
                for k in range(codes):
                    mask=assignment[:,b]==k
                    centroids[b,k]=response[mask,b].mean(0) if np.any(mask) else response[rng.integers(T),b]
        distance=((response[:,:,None,:]-centroids[None,:,:,:])**2).sum(-1)
        nearest=distance.min(-1);variance=np.mean((response-response.mean(0,keepdims=True))**2,axis=(0,2))*O
        scale=np.maximum(np.median(nearest,axis=0),variance*1e-6+np.finfo(np.float64).tiny)
        return cls(centroids,scale,D,block_size)

    @property
    def blocks(self)->int:
        return int(self.responses.shape[0])

    @property
    def codes(self)->int:
        return int(self.responses.shape[1])

    @property
    def output_width(self)->int:
        return int(self.responses.shape[2])

    @property
    def storage_ratio_to_dense_weight(self)->float:
        return self.responses.size/(self.input_width*self.output_width)

    def encode_responses(
        self,
        block_responses:np.ndarray,
        *,
        previous_codes:np.ndarray|None=None,
        transition_penalty:float=0.0,
    )->np.ndarray:
        r=np.asarray(block_responses,dtype=np.float64)
        if r.shape!=(self.blocks,self.output_width):
            raise ValueError("response shape mismatch")
        cost=((r[:,None,:]-self.responses)**2).sum(-1)
        if previous_codes is not None and transition_penalty:
            previous=np.asarray(previous_codes,dtype=np.int64)
            cost+=transition_penalty*self.distortion_scale[:,None]
            cost[np.arange(self.blocks),previous]-=transition_penalty*self.distortion_scale
        return cost.argmin(1).astype(np.int64)

    def block_responses(self,value:np.ndarray,weight:np.ndarray)->np.ndarray:
        x=np.asarray(value,dtype=np.float64);W=np.asarray(weight,dtype=np.float64)
        if x.shape!=(self.input_width,) or W.shape!=(self.output_width,self.input_width):
            raise ValueError("shape mismatch")
        xb=x.reshape(self.blocks,self.block_size)
        out=np.empty((self.blocks,self.output_width),dtype=np.float64)
        for b in range(self.blocks):
            lo=b*self.block_size;hi=lo+self.block_size
            out[b]=W[:,lo:hi]@xb[b]
        return out

    def decode_output(self,codes:np.ndarray,bias:np.ndarray|None=None)->np.ndarray:
        c=np.asarray(codes,dtype=np.int64)
        y=self.responses[np.arange(self.blocks),c].sum(0)
        return y if bias is None else y+np.asarray(bias,dtype=np.float64)

    def encode_trace(
        self,
        trace:np.ndarray,
        weight:np.ndarray,
        *,
        transition_penalty:float=0.0,
        bias:np.ndarray|None=None,
    )->tuple[np.ndarray,np.ndarray]:
        x=np.asarray(trace,dtype=np.float64)
        codes=[];outputs=[];previous=None
        for row in x:
            response=self.block_responses(row,weight)
            current=self.encode_responses(
                response,previous_codes=previous,transition_penalty=transition_penalty
            )
            codes.append(current);outputs.append(self.decode_output(current,bias));previous=current
        return np.stack(codes),np.stack(outputs)


@dataclass
class ConsumerQuotientState:
    codebook:ConsumerQuotientCodebook
    codes:np.ndarray
    output:np.ndarray
    bias:np.ndarray|None=None

    @classmethod
    def create(
        cls,codebook:ConsumerQuotientCodebook,codes:np.ndarray,bias:np.ndarray|None=None
    )->"ConsumerQuotientState":
        c=np.asarray(codes,dtype=np.int64).copy()
        b=None if bias is None else np.asarray(bias,dtype=np.float64).copy()
        return cls(codebook,c,codebook.decode_output(c,b),b)

    def update(self,new_codes:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        new=np.asarray(new_codes,dtype=np.int64)
        changed=np.flatnonzero(new!=self.codes)
        if len(changed):
            self.output+=self.codebook.responses[changed,new[changed]].sum(0)
            self.output-=self.codebook.responses[changed,self.codes[changed]].sum(0)
            self.codes[changed]=new[changed]
        return self.output,changed
