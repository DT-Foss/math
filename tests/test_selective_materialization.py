import numpy as np
from wirkung_math.lowrank import FactorDelta


def test_selective_block_materialization():
    rng=np.random.default_rng(93)
    L=rng.normal(size=(100,4)); R=rng.normal(size=(80,4))
    f=FactorDelta(L,R)
    rows=np.array([1,7,50]); cols=np.array([2,9,20,70])
    np.testing.assert_allclose(
        f.materialize_block(rows,cols),
        f.materialize()[np.ix_(rows,cols)],
        rtol=1e-12,atol=1e-12,
    )
