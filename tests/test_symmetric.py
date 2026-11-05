import numpy as np
from wirkung_math.symmetric import SymmetricLowRankDelta, WoodburyCache


def test_gram_replacement_and_woodbury_fanout():
    rng=np.random.default_rng(30)
    d,e=40,4
    z=rng.normal(size=(80,d)); A=z.T@z+5*np.eye(d)
    old=rng.normal(scale=.1,size=(e,d)); new=rng.normal(scale=.1,size=(e,d))
    delta=SymmetricLowRankDelta.from_gram_replacement(old,new)
    ref=A-old.T@old+new.T@new
    np.testing.assert_allclose(A+delta.materialize(),ref,rtol=1e-12,atol=1e-12)
    Ainv=np.linalg.inv(A)
    cache=WoodburyCache.prepare(Ainv,delta)
    rhs=rng.normal(size=(d,5))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(ref,rhs),rtol=1e-11,atol=1e-11)
    sign0,log0=np.linalg.slogdet(A)
    sign1,log1=cache.logdet_update(sign0,log0)
    sr,lr=np.linalg.slogdet(ref)
    assert sign1==sr
    assert abs(log1-lr)<1e-10


def test_symmetric_compaction():
    rng=np.random.default_rng(31)
    d=50
    base=rng.normal(size=(d,5))
    U=np.concatenate([base@rng.normal(size=(5,4)) for _ in range(5)],axis=1)
    C=np.diag(rng.choice([-1.,1.],size=U.shape[1]))
    delta=SymmetricLowRankDelta(U,C)
    compact=delta.compact(rtol=1e-10)
    assert compact.rank_bound <= 5
    np.testing.assert_allclose(compact.materialize(),delta.materialize(),rtol=1e-9,atol=1e-9)
