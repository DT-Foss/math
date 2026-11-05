from __future__ import annotations
import json,os,time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*args,**kwargs):
        return nullcontext()

from wirkung_math.temporal_linear import TemporalRMSNormLinearState
from wirkung_math.router_certificate import CertifiedTopKRouterState
from wirkung_math.partition_attention import CompiledPartitionAttention


def median_ms(fn,repeats=30):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();fn();samples.append((time.perf_counter_ns()-start)/1e6)
    return float(np.median(samples))


def rmsnorm_linear(W,x,gamma,eps=1e-6):
    rho=np.sqrt(np.mean(x*x)+eps)
    return W@((x/rho)*gamma)


def run():
    rng=np.random.default_rng(20261105)
    out={"experiment":"David Tom Foss temporal delta machine"}
    with threadpool_limits(limits=1):
        d=2560
        W=rng.normal(scale=1/np.sqrt(d),size=(d,d)).astype(np.float32)
        x=rng.normal(size=d).astype(np.float32)
        gamma=rng.normal(size=d).astype(np.float32)
        full_ms=median_ms(lambda:rmsnorm_linear(W,x,gamma),20)
        rows=[]
        for k in [1,4,8,16,32,64]:
            idx=np.sort(rng.choice(d,k,replace=False))
            state=TemporalRMSNormLinearState.from_input(W,gamma,x)
            new=x[idx]+rng.normal(scale=.1,size=k).astype(np.float32)
            delta_ms=median_ms(lambda:state.update_sparse(idx,new),30)
            rows.append({"changed_coordinates":k,"delta_ms":delta_ms,"speedup":full_ms/delta_ms})
        out["rmsnorm_linear"]={"full_ms":full_ms,"rows":rows}

        experts=384
        Wr=rng.normal(scale=.02,size=(experts,d)).astype(np.float32)
        xr=rng.normal(size=d).astype(np.float32);xr[0]=1
        Wr[:6,0]+=10
        router=CertifiedTopKRouterState.from_input(Wr,xr,6)
        idx=np.sort(rng.choice(d,8,replace=False));new=xr[idx]+1e-4
        full_router_ms=median_ms(lambda:Wr@xr,50)
        cert_ms=median_ms(lambda:router.update_sparse(idx,new),50)
        out["router"]={"full_ms":full_router_ms,"certificate_ms":cert_ms,"speedup":full_router_ms/cert_ms}

        n,dv,r=131072,64,4
        codebook=rng.integers(-3,4,size=(2401,r))
        sig=codebook[rng.integers(0,len(codebook),size=n)]
        base=rng.normal(size=n);values=rng.normal(size=(n,dv));alpha=rng.normal(scale=.1,size=r)
        compiled=CompiledPartitionAttention.compile(base,sig,values)
        def dense():
            scores=base+sig@alpha;m=np.max(scores);w=np.exp(scores-m);return (w@values)/w.sum()
        dense_ms=median_ms(dense,15)
        compiled_ms=median_ms(lambda:compiled.query(alpha),50)
        out["partition_attention"]={
            "tokens":n,"groups":compiled.occupied_groups,
            "dense_ms":dense_ms,"compiled_ms":compiled_ms,
            "speedup":dense_ms/compiled_ms,
        }
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    run()
