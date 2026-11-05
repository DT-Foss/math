from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProductTemporalCodebook:
    """
    Product/vector codebook whose assignment objective includes persistence.

    Each activation block is one discrete state. A transition penalty biases the
    encoder toward the previous state until reconstruction gain justifies a
    code transition. This minimizes temporal state traffic directly.
    """
    centroids:np.ndarray  # [blocks,codes,block_width]
    distortion_scale:np.ndarray  # [blocks]
    width:int
    block_size:int

    @classmethod
    def fit(
        cls,
        trace:np.ndarray,
        *,
        block_size:int=16,
        codes:int=16,
        iterations:int=20,
        seed:int=0,
    )->"ProductTemporalCodebook":
        x=np.asarray(trace,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]%block_size or not 1<=codes<=len(x):
            raise ValueError("trace must be [time,width], width divisible by block_size, codes<=time")
        rng=np.random.default_rng(seed)
        blocks=x.reshape(len(x),x.shape[1]//block_size,block_size)
        B=blocks.shape[1]
        centroids=np.empty((B,codes,block_size),dtype=np.float64)
        for b in range(B):
            centroids[b]=blocks[rng.choice(len(x),codes,replace=False),b]
        assignment=np.zeros((len(x),B),dtype=np.int64)
        for _ in range(iterations):
            distance=((blocks[:,:,None,:]-centroids[None,:,:,:])**2).sum(axis=-1)
            new_assignment=distance.argmin(axis=-1)
            if np.array_equal(new_assignment,assignment):
                break
            assignment=new_assignment
            for b in range(B):
                for k in range(codes):
                    mask=assignment[:,b]==k
                    if np.any(mask):
                        centroids[b,k]=blocks[mask,b].mean(axis=0)
                    else:
                        centroids[b,k]=blocks[rng.integers(len(x)),b]
        distance=((blocks[:,:,None,:]-centroids[None,:,:,:])**2).sum(axis=-1)
        nearest=distance.min(axis=-1)
        scale=np.median(nearest,axis=0)
        variance=np.mean((blocks-blocks.mean(axis=0,keepdims=True))**2,axis=(0,2))*block_size
        scale=np.maximum(scale,variance*1e-6+np.finfo(np.float64).tiny)
        return cls(centroids,scale,int(x.shape[1]),int(block_size))

    @property
    def blocks(self)->int:
        return int(self.centroids.shape[0])

    @property
    def codes(self)->int:
        return int(self.centroids.shape[1])

    @property
    def response_storage_ratio(self)->float:
        # response pages / dense weight values for one output matrix
        return self.codes/self.block_size

    def encode(
        self,
        value:np.ndarray,
        *,
        previous_codes:np.ndarray|None=None,
        transition_penalty:float=0.0,
    )->np.ndarray:
        x=np.asarray(value,dtype=np.float64)
        if x.shape!=(self.width,):
            raise ValueError("value shape mismatch")
        blocks=x.reshape(self.blocks,self.block_size)
        cost=((blocks[:,None,:]-self.centroids)**2).sum(axis=-1)
        if previous_codes is not None and transition_penalty>0:
            previous=np.asarray(previous_codes,dtype=np.int64)
            if previous.shape!=(self.blocks,):
                raise ValueError("previous code shape mismatch")
            cost+=transition_penalty*self.distortion_scale[:,None]
            cost[np.arange(self.blocks),previous]-=transition_penalty*self.distortion_scale
        return cost.argmin(axis=1).astype(np.int64)

    def decode(self,codes:np.ndarray)->np.ndarray:
        c=np.asarray(codes,dtype=np.int64)
        if c.shape!=(self.blocks,):
            raise ValueError("code shape mismatch")
        return self.centroids[np.arange(self.blocks),c].reshape(self.width)

    def encode_trace(
        self,trace:np.ndarray,*,transition_penalty:float=0.0
    )->tuple[np.ndarray,np.ndarray]:
        x=np.asarray(trace,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.width:
            raise ValueError("trace shape mismatch")
        codes=[];decoded=[];previous=None
        for row in x:
            current=self.encode(
                row,previous_codes=previous,transition_penalty=transition_penalty
            )
            codes.append(current);decoded.append(self.decode(current));previous=current
        return np.stack(codes),np.stack(decoded)


@dataclass
class ResponseCodedLinearState:
    """
    Linear layer compiled into block-code response pages.

    response[b,k] = W[:, block_b] @ centroid[b,k].
    A code transition updates the output by response[new]-response[old].
    """
    responses:np.ndarray  # [blocks,codes,output]
    codes:np.ndarray
    y:np.ndarray
    bias:np.ndarray|None=None

    @classmethod
    def compile(
        cls,
        weight:np.ndarray,
        codebook:ProductTemporalCodebook,
        initial_codes:np.ndarray,
        bias:np.ndarray|None=None,
    )->"ResponseCodedLinearState":
        W=np.asarray(weight,dtype=np.float64)
        c=np.asarray(initial_codes,dtype=np.int64).copy()
        if W.ndim!=2 or W.shape[1]!=codebook.width or c.shape!=(codebook.blocks,):
            raise ValueError("shape mismatch")
        out=W.shape[0]
        responses=np.empty((codebook.blocks,codebook.codes,out),dtype=np.float64)
        for b in range(codebook.blocks):
            lo=b*codebook.block_size;hi=lo+codebook.block_size
            responses[b]=codebook.centroids[b]@W[:,lo:hi].T
        base=responses[np.arange(codebook.blocks),c].sum(axis=0)
        bvec=None if bias is None else np.asarray(bias,dtype=np.float64).copy()
        if bvec is not None:
            base=base+bvec
        return cls(responses,c,base,bvec)

    def update(self,new_codes:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        new=np.asarray(new_codes,dtype=np.int64)
        if new.shape!=self.codes.shape:
            raise ValueError("code shape mismatch")
        changed=np.flatnonzero(new!=self.codes)
        if len(changed):
            self.y+=self.responses[changed,new[changed]].sum(axis=0)
            self.y-=self.responses[changed,self.codes[changed]].sum(axis=0)
            self.codes[changed]=new[changed]
        return self.y,changed

    def full_from_codes(self,codes:np.ndarray)->np.ndarray:
        c=np.asarray(codes,dtype=np.int64)
        y=self.responses[np.arange(len(c)),c].sum(axis=0)
        return y if self.bias is None else y+self.bias


def product_code_rate_distortion(
    trace:np.ndarray,
    codebook:ProductTemporalCodebook,
    *,
    transition_penalty:float,
)->dict:
    x=np.asarray(trace,dtype=np.float64)
    codes,decoded=codebook.encode_trace(x,transition_penalty=transition_penalty)
    changed=(codes[1:]!=codes[:-1]).sum(axis=1)
    rel=np.linalg.norm(decoded-x)/max(np.linalg.norm(x),np.finfo(np.float64).tiny)
    return {
        "transition_penalty":float(transition_penalty),
        "block_size":codebook.block_size,
        "codes_per_block":codebook.codes,
        "blocks":codebook.blocks,
        "response_storage_ratio":codebook.response_storage_ratio,
        "changed_blocks_median":float(np.median(changed)) if len(changed) else 0.0,
        "changed_blocks_p90":float(np.percentile(changed,90)) if len(changed) else 0.0,
        "changed_block_fraction_median":float(np.median(changed)/codebook.blocks) if len(changed) else 0.0,
        "relative_l2_error":float(rel),
        "codes":codes,
        "decoded":decoded,
    }
