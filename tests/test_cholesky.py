import numpy as np
from wirkung_math.cholesky import CholeskyState


def test_cholesky_gram_replace_solve_logdet():
    rng=np.random.default_rng(60)
    d,e=48,3
    z=rng.normal(size=(100,d)); A=z.T@z+20*np.eye(d)
    old=rng.normal(scale=.1,size=(e,d)); new=rng.normal(scale=.1,size=(e,d))
    ref=A-old.T@old+new.T@new
    state=CholeskyState.from_matrix(A)
    state.replace_gram_rows(old,new)
    np.testing.assert_allclose(state.matrix(),ref,rtol=1e-11,atol=1e-11)
    rhs=rng.normal(size=(d,4))
    np.testing.assert_allclose(state.solve(rhs),np.linalg.solve(ref,rhs),rtol=1e-10,atol=1e-10)
    _,logref=np.linalg.slogdet(ref)
    assert abs(state.logdet-logref)<1e-10
