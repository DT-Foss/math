import numpy as np
from wirkung_math.cholesky import CholeskyState
from wirkung_math.kernel import rbf_kernel,rbf_replacement_delta,RowColumnSolverWoodbury


def test_rbf_rowcolumn_update_over_cholesky_without_inverse():
    rng=np.random.default_rng(42)
    n,d,e=70,5,2
    x=rng.normal(size=(n,d)); idx=np.array([5,41]); new=x[idx]+rng.normal(scale=.1,size=(e,d))
    K=rbf_kernel(x,.25)+.5*np.eye(n)
    x2=x.copy(); x2[idx]=new
    K2=rbf_kernel(x2,.25)+.5*np.eye(n)
    delta=rbf_replacement_delta(x,idx,new,.25)
    chol=CholeskyState.from_matrix(K)
    cache=RowColumnSolverWoodbury.prepare(chol.solve,delta)
    rhs=rng.normal(size=(n,3))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(K2,rhs),rtol=1e-10,atol=1e-10)
    _,ref=np.linalg.slogdet(K2)
    _,got=cache.logdet_update(1.0,chol.logdet)
    assert abs(got-ref)<1e-10
