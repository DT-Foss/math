import numpy as np
from _common import emit
from wirkung_math.nonlinear import rank_barrier

rng=np.random.default_rng(176); n,d,r=512,64,4
X=rng.normal(size=(n,d)); B=rng.normal(size=(d,r))
rows=[]
for scale in [1e-3,1e-2,1e-1,0.5,1.0]:
    A=rng.normal(scale=scale,size=(n,r))
    x=rank_barrier(X,A,B)
    x["delta_scale"]=scale; rows.append(x)
emit({"experiment":"E176","rows":rows})
