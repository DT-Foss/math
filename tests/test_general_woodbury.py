import numpy as np
from wirkung_math.woodbury import GeneralLowRankUpdate,GeneralWoodburyCache


def test_general_nonsymmetric_woodbury():
    rng=np.random.default_rng(91)
    n,r=30,4
    A=rng.normal(size=(n,n)); A=A+5*np.eye(n)
    U=rng.normal(scale=.1,size=(n,r)); V=rng.normal(scale=.1,size=(n,r))
    update=GeneralLowRankUpdate(U,V)
    A2=A+update.materialize()
    cache=GeneralWoodburyCache.prepare(np.linalg.inv(A),update)
    rhs=rng.normal(size=(n,3))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(A2,rhs),rtol=1e-10,atol=1e-10)
    s0,l0=np.linalg.slogdet(A); s,l=cache.logdet_update(s0,l0); sr,lr=np.linalg.slogdet(A2)
    assert s==sr and abs(l-lr)<1e-10
