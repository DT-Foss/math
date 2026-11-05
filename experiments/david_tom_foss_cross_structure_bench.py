import os,time,json
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
from threadpoolctl import threadpool_limits
from wirkung_math.kernel import rbf_single_point_sparse_feature_update
from wirkung_math.reduction import SegmentReductionTree

def med(fn,repeats=7,scale=1e-6):
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns(); fn(); values.append((time.perf_counter_ns()-start)*scale)
    return float(np.median(values))

def full_rbf_row(x,u,gamma):
    q=x[u]
    dist=np.sum((x-q)**2,axis=1,dtype=np.float64)
    return np.exp(-gamma*dist)

rng=np.random.default_rng(20261105)
rows=[]; d=256; gamma=.05
with threadpool_limits(limits=1):
    for n,s in [(32768,1),(32768,4),(65536,1),(65536,4),(65536,16)]:
        x=rng.normal(size=(n,d)).astype(np.float64); u=17
        feat=np.sort(rng.choice(d,s,replace=False)); delta=rng.normal(scale=.03,size=s)
        old=x[u].copy(); old_sq=np.sum((x-old)**2,axis=1,dtype=np.float64)
        x2=x.copy(); x2[u,feat]+=delta
        full=lambda: full_rbf_row(x2,u,gamma)
        inc=lambda: rbf_single_point_sparse_feature_update(old_sq,x[:,feat],old[feat],delta,u,gamma)
        full(); inc()
        full_ms=med(full); delta_ms=med(inc)
        a=full(); b=inc()
        rows.append({"n":n,"d":d,"changed_features":s,"full_ms":full_ms,"delta_ms":delta_ms,
                     "speedup":full_ms/delta_ms,"max_abs_error":float(np.max(np.abs(a-b)))})

n=1_048_576
a=rng.normal(size=n)
tree=SegmentReductionTree(a.tolist(),max,float("-inf"))
index=123456
full_us=[]; tree_us=[]
for _ in range(31):
    value=float(rng.normal()); a[index]=value
    start=time.perf_counter_ns(); np.max(a); full_us.append((time.perf_counter_ns()-start)/1e3)
    value=float(rng.normal())
    start=time.perf_counter_ns(); tree.replace(index,value); tree_us.append((time.perf_counter_ns()-start)/1e3)
    a[index]=value
out={"sparse_feature_rbf":rows,"associative_max_tree":{
    "n":n,"full_reduce_us_median":float(np.median(full_us)),
    "tree_update_us_median":float(np.median(tree_us)),
    "speedup":float(np.median(full_us)/np.median(tree_us)),
    "exact":bool(tree.aggregate==np.max(a))}}
print(json.dumps(out,indent=2))
