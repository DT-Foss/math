import numpy as np
from wirkung_math.cholesky import CholeskyState
from wirkung_math.woodbury import GeneralLowRankUpdate,SolverWoodburyCache


def test_woodbury_over_cholesky_solver_without_inverse():
    rng=np.random.default_rng(92)
    n,r=50,4
    z=rng.normal(size=(100,n)); A=z.T@z+10*np.eye(n)
    # symmetric update represented in general U,V form
    U=rng.normal(scale=.05,size=(n,r))
    V=U.copy()
    A2=A+U@V.T
    chol=CholeskyState.from_matrix(A)
    cache=SolverWoodburyCache.prepare(chol.solve,GeneralLowRankUpdate(U,V))
    rhs=rng.normal(size=(n,5))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(A2,rhs),rtol=1e-10,atol=1e-10)
    cache.logdet_update(1.0,chol.logdet)
    _,ref=np.linalg.slogdet(A2)
    _,got=cache.logdet_update(1.0,chol.logdet)
    assert abs(got-ref)<1e-10
