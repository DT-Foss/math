import numpy as np
from wirkung_math.replacement import lowrank_replacement_statistics

def test_lowrank_replacement_statistics_exact():
    rng=np.random.default_rng(180)
    e,d,r=64,32,4
    old=rng.normal(size=(e,d))
    A=rng.normal(scale=.03,size=(e,r)); B=rng.normal(size=(d,r))
    new=old+A@B.T
    ds,dq,dg=lowrank_replacement_statistics(A,B,old,new)
    np.testing.assert_allclose(ds,new.sum(0)-old.sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(dq,(new*new).sum(0)-(old*old).sum(0),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(dg.materialize(),new.T@new-old.T@old,rtol=1e-11,atol=1e-11)
    assert dg.rank_bound==2*r
