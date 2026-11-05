import numpy as np
from wirkung_math.replacement import replacement_statistics_delta,replacement_gram_delta,replacement_cross_delta

def test_midpoint_replacement_statistics_exact():
    rng=np.random.default_rng(150)
    old=rng.normal(size=(17,32)); new=old+rng.normal(scale=.02,size=old.shape)
    ds,dq,dg=replacement_statistics_delta(old,new)
    np.testing.assert_allclose(ds,new.sum(0)-old.sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(dq,(new*new).sum(0)-(old*old).sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(dg,new.T@new-old.T@old,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(replacement_gram_delta(old,new).materialize(),dg,rtol=1e-12,atol=1e-12)

def test_cross_replacement_exact():
    rng=np.random.default_rng(151)
    ox=rng.normal(size=(9,12)); oy=rng.normal(size=(9,7))
    nx=ox+rng.normal(scale=.03,size=ox.shape); ny=oy+rng.normal(scale=.03,size=oy.shape)
    got=replacement_cross_delta(ox,nx,oy,ny)
    ref=nx.T@ny-ox.T@oy
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
