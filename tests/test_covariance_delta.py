import numpy as np
from wirkung_math.replacement import replacement_statistics_delta,replacement_gram_delta
from wirkung_math.covariance import covariance_replacement_delta

def test_covariance_delta_exact_and_lowrank():
    rng=np.random.default_rng(190)
    n,d,e=200,24,5
    x=rng.normal(size=(n,d))
    ids=np.array([1,7,31,90,150])
    old=x[ids].copy(); new=old+rng.normal(scale=.05,size=old.shape)
    old_mean=x.mean(0)
    old_cov=(x.T@x)/n-np.outer(old_mean,old_mean)
    ds,_,_=replacement_statistics_delta(old,new)
    gd=replacement_gram_delta(old,new)
    cd=covariance_replacement_delta(n,old_mean,ds,gd)
    x[ids]=new
    new_mean=x.mean(0)
    new_cov=(x.T@x)/n-np.outer(new_mean,new_mean)
    np.testing.assert_allclose(old_cov+cd.materialize(),new_cov,rtol=1e-11,atol=1e-11)
    assert cd.rank_bound<=2*e+2
