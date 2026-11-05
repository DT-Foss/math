import numpy as np
from wirkung_math.router_certificate import CertifiedTopKRouterState


def topk(x,k):
    return set(np.argpartition(x,-k)[-k:].tolist())


def test_certified_router_matches_full_and_refreshes_when_needed():
    rng=np.random.default_rng(210)
    experts,d,k=32,24,4
    W=rng.normal(scale=.1,size=(experts,d))
    x=rng.normal(size=d)
    # Give the initial winners a large exact margin.
    W[:k,0]+=20.0
    x[0]=1.0
    state=CertifiedTopKRouterState.from_input(W,x,k)

    idx=np.array([5,7]);new=x[idx]+np.array([1e-4,-1e-4])
    selected,weights,certified=state.update_sparse(idx,new)
    x[idx]=new
    full=W@x
    assert certified
    assert set(selected.tolist())==topk(full,k)
    exact=np.exp(full[selected]-np.max(full[selected]));exact/=exact.sum()
    np.testing.assert_allclose(weights,exact,rtol=1e-12,atol=1e-12)

    # A large edit invalidates the interval certificate and forces a refresh.
    idx=np.array([0]);new=np.array([-2.0])
    selected,_,certified=state.update_sparse(idx,new)
    x[idx]=new
    assert not certified
    assert set(selected.tolist())==topk(W@x,k)
    assert state.full_refreshes==1
