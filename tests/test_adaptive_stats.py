import numpy as np
from wirkung_math.adaptive_stats import AdaptiveStatsState

def test_adaptive_stats_single_and_batch_exact():
    rng=np.random.default_rng(170)
    x=rng.normal(size=(200,32))
    s=AdaptiveStatsState(x.sum(0),(x*x).sum(0),x.T@x)
    i=3; old=x[i].copy(); new=old+rng.normal(scale=.05,size=32)
    s.replace_rows(old,new); x[i]=new
    ids=np.array([4,10,33,90])
    old=x[ids].copy(); new=old+rng.normal(scale=.05,size=old.shape)
    s.replace_rows(old,new); x[ids]=new
    np.testing.assert_allclose(s.sum,x.sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(s.sumsq,(x*x).sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(s.gram,x.T@x,rtol=1e-11,atol=1e-11)
