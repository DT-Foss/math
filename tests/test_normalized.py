import numpy as np
from wirkung_math.normalized import NormalizedState, full_normalized, mass_gated_update, topk_miss_probability, minimum_topk


def test_mass_gated_update_exact_typical():
    rng=np.random.default_rng(2); n,d=128,16
    scores=rng.normal(size=(n,n)); values=rng.normal(size=(n,d))
    st=NormalizedState.from_full(scores,values)
    cols=np.array([1,7,9,31,63,95,100,111])
    ns=st.scores[:,cols]+rng.normal(scale=.1,size=(n,len(cols)))
    nv=st.values[cols]+rng.normal(scale=.1,size=(len(cols),d))
    mass_gated_update(st,cols,ns,nv)
    ref,_=full_normalized(st.scores,st.values)
    np.testing.assert_allclose(st.output,ref,rtol=2e-12,atol=2e-12)


def test_mass_gate_rebases_dominant_deleted_term():
    n,d=32,4
    scores=np.zeros((n,n)); scores[:,0]=80.0
    values=np.arange(n*d,dtype=float).reshape(n,d)/10
    st=NormalizedState.from_full(scores,values)
    ns=np.full((n,1),-80.0)
    stats=mass_gated_update(st,np.array([0]),ns,cancellation_floor=1e-8)
    ref,_=full_normalized(st.scores,st.values)
    assert stats["fallback_rows"]==n
    np.testing.assert_allclose(st.output,ref,rtol=1e-12,atol=1e-12)


def test_small_topk_probability():
    p=topk_miss_probability(8192,1024,8)
    assert p < 6e-8
    assert minimum_topk(8192,1024,expected_fallback_rows=1e-3) <= 8
