import numpy as np
from wirkung_math.norms import rmsnorm, rmsnorm_lowrank_update, layernorm, layernorm_lowrank_update


def test_rmsnorm_lowrank_exact():
    rng=np.random.default_rng(3); n,d,r=64,32,4
    X=rng.normal(size=(n,d)); A=rng.normal(scale=.1,size=(n,r)); B=rng.normal(scale=.1,size=(d,r)); g=rng.normal(size=d)
    got=rmsnorm_lowrank_update(X,A,B,g).materialize()
    ref=rmsnorm(X+A@B.T,g)
    np.testing.assert_allclose(got,ref,rtol=2e-12,atol=2e-12)


def test_layernorm_lowrank_exact():
    rng=np.random.default_rng(4); n,d,r=64,32,4
    X=rng.normal(size=(n,d)); A=rng.normal(scale=.1,size=(n,r)); B=rng.normal(scale=.1,size=(d,r)); g=rng.normal(size=d); b=rng.normal(size=d)
    got=layernorm_lowrank_update(X,A,B,g,b).materialize()
    ref=layernorm(X+A@B.T,g,b)
    np.testing.assert_allclose(got,ref,rtol=3e-12,atol=3e-12)
