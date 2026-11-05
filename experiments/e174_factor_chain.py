import numpy as np
from _common import median_ms, emit
from wirkung_math.lowrank import FactorDelta

rng=np.random.default_rng(174); n,d,r,layers=4096,64,4,6
C=rng.normal(size=(n,r)).astype(np.float32); B=rng.normal(size=(d,r)).astype(np.float32)
Ws=[rng.normal(size=(d,d)).astype(np.float32)/np.sqrt(d) for _ in range(layers)]

def dense_run():
    x=C@B.T
    for W in Ws: x=x@W
    return x

def factor_run():
    f=FactorDelta(C,B)
    for W in Ws: f=f.compose_right(W)
    return f.materialize()

dense_ms=median_ms(dense_run,5); factor_ms=median_ms(factor_run,5)
a=dense_run(); b=factor_run()
emit({"experiment":"E174","n":n,"d":d,"rank":r,"layers":layers,"dense_ms":dense_ms,"factor_ms":factor_ms,
      "speedup":dense_ms/factor_ms,"max_abs_error":float(np.max(np.abs(a-b)))})
