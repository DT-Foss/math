import numpy as np
from wirkung_math.kernel import rbf_kernel, rbf_replacement_delta, rbf_single_point_sparse_feature_update


def test_rbf_rowcolumn_delta_and_implicit_woodbury():
    rng=np.random.default_rng(40)
    n,d,e=80,6,3
    x=rng.normal(size=(n,d)); idx=np.array([3,20,61]); new=x[idx]+rng.normal(scale=.1,size=(e,d))
    gamma=.3
    K=rbf_kernel(x,gamma)+.5*np.eye(n)
    x2=x.copy(); x2[idx]=new
    K2=rbf_kernel(x2,gamma)+.5*np.eye(n)
    delta=rbf_replacement_delta(x,idx,new,gamma)
    np.testing.assert_allclose(K+delta.materialize(),K2,rtol=1e-11,atol=1e-11)
    cache=delta.prepare_woodbury(np.linalg.inv(K))
    rhs=rng.normal(size=(n,4))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(K2,rhs),rtol=1e-10,atol=1e-10)


def test_sparse_feature_rbf_row_update():
    rng=np.random.default_rng(41)
    n,d=100,20
    x=rng.normal(size=(n,d)); u=7; feat=np.array([1,4,11])
    delta=np.array([.1,-.2,.05]); gamma=.2
    old_sq=((x[u]-x)**2).sum(1)
    row=rbf_single_point_sparse_feature_update(old_sq,x[:,feat],x[u,feat],delta,u,gamma)
    x2=x.copy(); x2[u,feat]+=delta
    ref=rbf_kernel(x2,gamma)[u]
    np.testing.assert_allclose(row,ref,rtol=1e-12,atol=1e-12)
