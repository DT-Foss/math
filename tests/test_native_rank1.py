import numpy as np
from wirkung_math.native_rank1 import NativeRank1StatsState

def test_native_rank1_update_exact():
    rng=np.random.default_rng(160)
    x=rng.normal(size=(100,64))
    s=x.sum(0); q=(x*x).sum(0); g=x.T@x
    state=NativeRank1StatsState(s.copy(),q.copy(),g.copy())
    i=7; old=x[i].copy(); new=old+rng.normal(scale=.1,size=64)
    state.update(old,new); x[i]=new
    np.testing.assert_allclose(state.sum,x.sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.sumsq,(x*x).sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.gram,x.T@x,rtol=1e-12,atol=1e-12)
