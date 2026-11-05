import numpy as np
from wirkung_math.block import BlockAppendCache,delete_trailing_block_from_inverse


def test_block_append_solve_logdet_and_delete():
    rng=np.random.default_rng(70)
    n,k=40,3
    z=rng.normal(size=(n+k,n+k)); full=z@z.T+10*np.eye(n+k)
    A=full[:n,:n]; B=full[:n,n:]; D=full[n:,n:]
    Ainv=np.linalg.inv(A)
    cache=BlockAppendCache.prepare(Ainv,B,D)
    rhs=rng.normal(size=(n+k,4))
    np.testing.assert_allclose(cache.apply_inverse(rhs),np.linalg.solve(full,rhs),rtol=1e-11,atol=1e-11)
    sign0,log0=np.linalg.slogdet(A); sign,log=cache.logdet_update(sign0,log0)
    sr,lr=np.linalg.slogdet(full)
    assert sign==sr and abs(log-lr)<1e-10
    full_inv=cache.materialize_inverse()
    np.testing.assert_allclose(delete_trailing_block_from_inverse(full_inv,k),Ainv,rtol=1e-11,atol=1e-11)
