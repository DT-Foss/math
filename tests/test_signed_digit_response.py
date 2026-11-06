import numpy as np
from wirkung_math.signed_digit_response import (
    balanced_ternary_scales,balanced_ternary_encode,
    SignedDigitResponseLinearState,
)


def test_signed_digit_range_and_precision():
    maxabs=np.array([1.0,2.0])
    s=balanced_ternary_scales(maxabs,5)
    np.testing.assert_allclose(s.sum(0),maxabs,rtol=1e-12,atol=1e-12)
    x=np.array([.733,-1.111])
    c=balanced_ternary_encode(x,s)
    recon=(c*s).sum(0)
    assert np.max(np.abs(recon-x)) < np.max(s[-1])


def test_temporal_signed_digit_equals_full_compiled_state():
    rng=np.random.default_rng(310)
    W=rng.normal(size=(16,24))
    xs=rng.normal(size=(50,24))
    state=SignedDigitResponseLinearState.compile(
        W,np.max(np.abs(xs),axis=0),stages=5,block_size=2
    )
    refs=[]
    for x in xs:
        state.reset();refs.append(state.full(x))
    state.reset();got=[]
    for x in xs:
        y,_=state.update(x);got.append(y.copy())
    np.testing.assert_array_equal(np.stack(got),np.stack(refs))
