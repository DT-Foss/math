import numpy as np
from wirkung_math.replacement import lowrank_delta_statistics_from_base

def test_factored_delta_updates_stats_without_materializing_new():
    rng=np.random.default_rng(181)
    e,d,r=80,48,3
    old=rng.normal(size=(e,d))
    A=rng.normal(scale=.02,size=(e,r)); B=rng.normal(size=(d,r))
    new=old+A@B.T
    ds,dq,dg=lowrank_delta_statistics_from_base(A,B,old)
    np.testing.assert_allclose(ds,new.sum(0)-old.sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(dq,(new*new).sum(0)-(old*old).sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(dg.materialize(),new.T@new-old.T@old,rtol=1e-11,atol=1e-11)
