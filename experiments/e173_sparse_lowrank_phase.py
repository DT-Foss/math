import numpy as np
from _common import median_ms, emit
from wirkung_math.lowrank import crossover_rank

rng=np.random.default_rng(173); n,e,d=4096,512,64
E=rng.normal(size=(n,e)).astype(np.float32)
rows=[]
for r in [1,2,4,8,16,32,64]:
    A=rng.normal(size=(e,r)).astype(np.float32); B=rng.normal(size=(d,r)).astype(np.float32); DV=A@B.T
    direct_ms=median_ms(lambda: E@DV,5)
    fact_ms=median_ms(lambda: (E@A)@B.T,5)
    err=float(np.max(np.abs((E@DV)-((E@A)@B.T))))
    rows.append({"rank":r,"direct_ms":direct_ms,"factorized_ms":fact_ms,"speedup":direct_ms/fact_ms,"max_abs_error":err})
emit({"experiment":"E173","n":n,"e":e,"d":d,"crossover_rank":crossover_rank(e,d),"rows":rows})
