import numpy as np
from wirkung_math.lowrank import FactorDelta


def test_factor_chain_exact():
    rng=np.random.default_rng(5); n,d,r=80,24,3
    C=rng.normal(size=(n,r)); B=rng.normal(size=(d,r)); f=FactorDelta(C,B); dense=f.materialize()
    for _ in range(4):
        W=rng.normal(size=(d,d))/d**.5
        dense=dense@W
        f=f.compose_right(W)
    np.testing.assert_allclose(f.materialize(),dense,rtol=2e-12,atol=2e-12)
