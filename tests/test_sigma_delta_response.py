import numpy as np
from wirkung_math.sigma_delta_response import SigmaDeltaResponseLinearState


def test_sigma_delta_temporal_matches_compiled_full_each_step():
    rng=np.random.default_rng(320)
    W=rng.normal(size=(20,32)); b=rng.normal(size=20)
    step=np.full(32,.02)
    xs=np.cumsum(rng.normal(scale=.008,size=(100,32)),axis=0)
    s=SigmaDeltaResponseLinearState.compile(W,step,b)
    s.initialize(xs[0])
    for x in xs[1:]:
        ref=s.compiled_full(x)
        got,_=s.update(x)
        np.testing.assert_array_equal(got,ref)


def test_sigma_delta_error_feedback_does_not_drift():
    W=np.eye(4); step=np.full(4,.1)
    s=SigmaDeltaResponseLinearState.compile(W,step)
    s.initialize(np.zeros(4))
    # Sub-threshold innovations accumulate in the residual until a level crosses.
    events=[]
    for t in range(1,10):
        x=np.full(4,.03*t)
        _,dirty=s.update(x);events.append(dirty)
        assert np.max(np.abs(x-s.reconstructed_input)) <= .05+1e-12
    assert 0 in events and 4 in events
