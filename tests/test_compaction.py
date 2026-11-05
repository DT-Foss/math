import numpy as np
from wirkung_math.lowrank import FactorDelta

def test_compaction_preserves_delta_and_reduces_dependent_rank():
    rng=np.random.default_rng(7); n,d,k=80,40,6
    L=rng.normal(size=(n,k)); R=rng.normal(size=(d,k))
    f=None
    for _ in range(5):
        P=rng.normal(size=(k,3)); Q=rng.normal(size=(k,3))
        term=FactorDelta(L@P,R@Q)
        f=term if f is None else f.add(term)
    dense=f.materialize(); c=f.compact(rtol=1e-11)
    assert c.rank <= k
    np.testing.assert_allclose(c.materialize(),dense,rtol=1e-10,atol=1e-10)
