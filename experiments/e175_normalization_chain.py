import numpy as np
from _common import emit
from wirkung_math.norms import rmsnorm,rmsnorm_lowrank_update,layernorm,layernorm_lowrank_update

rng=np.random.default_rng(175); n,d,r=512,64,4
X=rng.normal(size=(n,d)); A=rng.normal(scale=.05,size=(n,r)); B=rng.normal(scale=.05,size=(d,r)); g=rng.normal(size=d); beta=rng.normal(size=d)
rms=rmsnorm_lowrank_update(X,A,B,g).materialize(); rms_ref=rmsnorm(X+A@B.T,g)
ln=layernorm_lowrank_update(X,A,B,g,beta).materialize(); ln_ref=layernorm(X+A@B.T,g,beta)
emit({"experiment":"E175","n":n,"d":d,"rank":r,"rms_max_abs_error":float(np.max(np.abs(rms-rms_ref))),
      "layernorm_max_abs_error":float(np.max(np.abs(ln-ln_ref)))})
