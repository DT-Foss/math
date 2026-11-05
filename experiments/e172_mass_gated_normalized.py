import time
import numpy as np
from _common import emit
from wirkung_math.normalized import NormalizedState, full_normalized, mass_gated_update, minimum_topk

rng=np.random.default_rng(172); n,d=1024,64
base_scores=rng.normal(size=(n,n)).astype(np.float32); base_values=rng.normal(size=(n,d)).astype(np.float32)
base=NormalizedState.from_full(base_scores,base_values)
rows=[]
for e in [64,128,256,512]:
    cols=np.arange(e)
    new_scores=base_scores[:,cols]+rng.normal(scale=.05,size=(n,e)).astype(np.float32)
    new_values=base_values[cols]+rng.normal(scale=.05,size=(e,d)).astype(np.float32)
    s_full=base_scores.copy(); v_full=base_values.copy(); s_full[:,cols]=new_scores; v_full[cols]=new_values
    full_times=[]; delta_times=[]
    for _ in range(5):
        t=time.perf_counter(); full_normalized(s_full,v_full); full_times.append((time.perf_counter()-t)*1000)
        st=NormalizedState(base.scores.copy(),base.values.copy(),base.output.copy(),base.logz.copy())
        t=time.perf_counter(); mass_gated_update(st,cols,new_scores,new_values); delta_times.append((time.perf_counter()-t)*1000)
    full_ms=float(np.median(full_times)); delta_ms=float(np.median(delta_times))
    st=NormalizedState(base.scores.copy(),base.values.copy(),base.output.copy(),base.logz.copy())
    stats=mass_gated_update(st,cols,new_scores,new_values)
    ref,_=full_normalized(st.scores,st.values)
    rows.append({"e":e,"fraction":e/n,"full_ms":full_ms,"delta_ms":delta_ms,"speedup":full_ms/delta_ms,
                 "fallback_rows":stats["fallback_rows"],"max_abs_error":float(np.max(np.abs(st.output-ref))),
                 "topk_for_1e-3_expected_fallback":minimum_topk(n,e,1e-3)})
emit({"experiment":"E172","rows":rows})
