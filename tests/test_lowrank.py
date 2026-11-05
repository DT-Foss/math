import numpy as np
from wirkung_math.lowrank import sparse_lowrank_update, crossover_rank, effective_edits


def test_sparse_lowrank_exact():
    rng=np.random.default_rng(1); n,e,d,r=64,12,16,4
    E=rng.normal(size=(n,e)); A=rng.normal(size=(e,r)); B=rng.normal(size=(d,r))
    got=sparse_lowrank_update(E,A,B).materialize()
    ref=E@(A@B.T)
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)


def test_router_math():
    assert abs(crossover_rank(512,64)-56.888888888888886)<1e-12
    assert effective_edits(512,64,4)==36.0
