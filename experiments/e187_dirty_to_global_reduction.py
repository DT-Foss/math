import numpy as np
from _common import median_ms, emit
from wirkung_math.dirty import update_global_sum, update_projected_global_sum

rng=np.random.default_rng(187)
n,d,e,p=131072,64,256,8
x=rng.normal(size=(n,d)).astype(np.float64)
idx=rng.choice(n,e,replace=False)
old=x[idx].copy()
new=old+rng.normal(scale=.05,size=old.shape)
x2=x.copy(); x2[idx]=new
old_sum=x.sum(axis=0)
W=rng.normal(size=(d,p))
old_projected=old_sum@W

full_sum_ms=median_ms(lambda: x2.sum(axis=0),7)
delta_sum_ms=median_ms(lambda: update_global_sum(old_sum,old,new),7)
full_proj_ms=median_ms(lambda: x2.sum(axis=0)@W,7)
delta_proj_ms=median_ms(lambda: update_projected_global_sum(old_projected,old,new,W),7)

sum_ref=x2.sum(axis=0)
sum_delta=update_global_sum(old_sum,old,new)
proj_ref=sum_ref@W
proj_delta=update_projected_global_sum(old_projected,old,new,W)

emit({
    "experiment":"E187",
    "n":n,"d":d,"edits":e,"projection_width":p,
    "sum_full_ms":full_sum_ms,
    "sum_delta_ms":delta_sum_ms,
    "sum_speedup":full_sum_ms/delta_sum_ms,
    "sum_max_abs_error":float(np.max(np.abs(sum_ref-sum_delta))),
    "projected_full_ms":full_proj_ms,
    "projected_delta_ms":delta_proj_ms,
    "projected_speedup":full_proj_ms/delta_proj_ms,
    "projected_max_abs_error":float(np.max(np.abs(proj_ref-proj_delta))),
})
