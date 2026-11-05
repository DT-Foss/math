import numpy as np
from wirkung_math.temporal_activation import QuantizedSwiGLUDownState


def test_quantized_swiglu_down_exact():
    rng=np.random.default_rng(250)
    hidden,out=96,48
    W=rng.normal(size=(out,hidden));levels=np.linspace(-2,2,17)
    gate=rng.normal(size=hidden);up=rng.normal(size=hidden)
    state=QuantizedSwiGLUDownState.from_gate_up(W,levels,gate,up)
    dg=rng.normal(scale=.02,size=hidden);du=rng.normal(scale=.02,size=hidden)
    got,changed=state.update(dg,du)
    np.testing.assert_allclose(got,state.full_reference(),rtol=1e-12,atol=1e-12)
    assert len(changed)<=hidden
