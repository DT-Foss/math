import time
import numpy as np
from _common import emit
from wirkung_math.lowrank import FactorDelta

rng=np.random.default_rng(177); n,d,latent,term_rank,terms=4096,64,8,4,32
L0=rng.normal(size=(n,latent)).astype(np.float64); R0=rng.normal(size=(d,latent)).astype(np.float64)
f=None
for _ in range(terms):
    P=rng.normal(size=(latent,term_rank)); Q=rng.normal(size=(latent,term_rank))
    term=FactorDelta(L0@P,R0@Q)
    f=term if f is None else f.add(term)
nominal=f.rank
t=time.perf_counter(); c=f.compact(rtol=1e-11); compact_ms=(time.perf_counter()-t)*1000
t=time.perf_counter(); a=f.materialize(); materialize_uncompacted_ms=(time.perf_counter()-t)*1000
t=time.perf_counter(); b=c.materialize(); materialize_compacted_ms=(time.perf_counter()-t)*1000
emit({"experiment":"E177","n":n,"d":d,"terms":terms,"term_rank":term_rank,"nominal_rank":nominal,
      "compacted_rank":c.rank,"compression":nominal/max(c.rank,1),"compact_ms":compact_ms,
      "materialize_uncompacted_ms":materialize_uncompacted_ms,"materialize_compacted_ms":materialize_compacted_ms,
      "materialize_speedup_after_compaction":materialize_uncompacted_ms/materialize_compacted_ms,
      "max_abs_error":float(np.max(np.abs(a-b)))})
