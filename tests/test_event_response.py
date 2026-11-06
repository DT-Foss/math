import numpy as np
from wirkung_math.event_response import IntegerResponseLinearState


def test_integer_response_temporal_equals_full_compiled_arithmetic():
    rng=np.random.default_rng(300)
    out,width=20,32
    W=rng.normal(size=(out,width))
    scale=np.maximum(rng.random(width)*.2,.02)
    state=IntegerResponseLinearState.compile(W,scale,qmax=3,block_size=2)
    xs=rng.normal(scale=.3,size=(40,width))
    refs=[]
    # Independent full compiled semantics.
    for x in xs:
        codes=state.encode(x)
        states=state.block_states(codes)
        acc=state.table[np.arange(len(states)),states].sum(axis=0,dtype=np.int64)
        refs.append(state.response_scale*acc)
    state.reset()
    got=[]
    for x in xs:
        y,_=state.update(x);got.append(y.copy())
    np.testing.assert_array_equal(np.stack(got),np.stack(refs))


def test_event_state_reports_changed_blocks():
    W=np.eye(8)
    scale=np.ones(8)
    s=IntegerResponseLinearState.compile(W,scale,qmax=1,block_size=2)
    x=np.zeros(8);s.update(x)
    x[2]=1
    _,changed=s.update(x)
    assert changed==1
