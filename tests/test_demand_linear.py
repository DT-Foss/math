import numpy as np
from wirkung_math.demand_linear import sparse_requested_linear_delta

def test_two_sided_sparse_linear_response():
    rng=np.random.default_rng(130)
    A=rng.normal(size=(80,100)); x=rng.normal(size=100)
    U=np.array([2,9,40,77]); K=np.array([1,7,20,60])
    dx=rng.normal(size=len(U))
    got=sparse_requested_linear_delta(lambda k,u:A[np.ix_(k,u)],K,U,dx)
    x2=x.copy(); x2[U]+=dx
    ref=(A@x2-A@x)[K]
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
