import numpy as np
from wirkung_math.delta_layout import ColumnPackedLinearOperator
from wirkung_math.temporal_events import (
    HystereticActivationCodec,EventLinearState,temporal_rate_distortion,
)


def test_hysteretic_events_reconstruct_persistent_code_state():
    trace=np.array([
        [0.0,0.0,0.0],
        [0.1,0.0,0.0],
        [0.1,0.7,0.0],
        [0.1,0.7,-1.0],
    ])
    codec=HystereticActivationCodec.fit(trace,bits=4,deadband_steps=.5)
    decoded,events=codec.replay(trace)
    current=decoded[0].copy()
    for i,event in enumerate(events,1):
        current[event.indices]+=event.delta_values
        np.testing.assert_allclose(current,decoded[i],rtol=0,atol=0)


def test_event_linear_matches_full_decoded_replay():
    rng=np.random.default_rng(300)
    trace=rng.normal(size=(20,32))
    codec=HystereticActivationCodec.fit(trace,bits=6,deadband_steps=2.0)
    decoded,events=codec.replay(trace)
    W=rng.normal(size=(24,32))
    state=EventLinearState.from_decoded(ColumnPackedLinearOperator.from_weight(W),decoded[0])
    for t,event in enumerate(events,1):
        got=state.apply(event)
        np.testing.assert_allclose(got,W@decoded[t],rtol=1e-11,atol=1e-11)


def test_deadband_reduces_or_preserves_event_count():
    rng=np.random.default_rng(301)
    base=np.cumsum(rng.normal(scale=.02,size=(100,64)),axis=0)
    fine=temporal_rate_distortion(base,bits=8,deadband_steps=.5)
    wide=temporal_rate_distortion(base,bits=8,deadband_steps=4.0)
    assert wide["event_fraction_median"]<=fine["event_fraction_median"]
    assert wide["relative_l2_error"]>=fine["relative_l2_error"]
