import numpy as np
from wirkung_math.bilinear import DynamicNormalizedBilinearState,full_normalized_bilinear


def test_dynamic_normalized_bilinear_replacement_exact():
    rng=np.random.default_rng(80)
    n,d,e=128,16,5
    q=rng.normal(size=(n,d)); k=rng.normal(size=(n,d)); v=rng.normal(size=(n,d))
    state=DynamicNormalizedBilinearState.from_qkv(q,k,v)
    idx=np.array([1,7,31,80,111])
    nq=q[idx]+rng.normal(scale=.05,size=(e,d))
    nk=k[idx]+rng.normal(scale=.05,size=(e,d))
    nv=v[idx]+rng.normal(scale=.05,size=(e,d))
    stats=state.replace_rows(idx,nq,nk,nv)
    ref,logref=full_normalized_bilinear(state.q,state.k,state.v,state.scale)
    np.testing.assert_allclose(state.output,ref,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(state.logz,logref,rtol=1e-11,atol=1e-11)
    assert stats["fallback_clean_rows"]==0
