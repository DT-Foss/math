import numpy as np
from wirkung_math.consumer_quotient import ConsumerQuotientCodebook,ConsumerQuotientState


def test_consumer_quotient_state_matches_decoded_outputs():
    rng=np.random.default_rng(320)
    trace=rng.normal(size=(40,32));W=rng.normal(size=(12,32));bias=rng.normal(size=12)
    codebook=ConsumerQuotientCodebook.fit(trace[:24],W,block_size=8,codes=8,seed=1)
    codes,outputs=codebook.encode_trace(trace[24:],W,transition_penalty=1.0,bias=bias)
    state=ConsumerQuotientState.create(codebook,codes[0],bias)
    np.testing.assert_allclose(state.output,outputs[0],rtol=1e-12,atol=1e-12)
    for t in range(1,len(codes)):
        got,_=state.update(codes[t])
        np.testing.assert_allclose(got,outputs[t],rtol=1e-12,atol=1e-12)


def test_consumer_quotient_ignores_exact_nullspace_motion():
    # W observes only the first two coordinates. Motion in the remaining two
    # coordinates is exactly quotient-equivalent to no motion.
    W=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    trace=np.array([
        [1.,2.,0.,0.],
        [1.,2.,1.,3.],
        [1.,2.,-4.,5.],
        [2.,2.,7.,8.],
    ])
    codebook=ConsumerQuotientCodebook.fit(trace,W,block_size=2,codes=4,seed=2)
    codes,outputs=codebook.encode_trace(trace,W)
    np.testing.assert_allclose(outputs,trace@W.T,rtol=1e-12,atol=1e-12)
    # Second block's code may vary internally, but its response is identically zero.
    assert np.allclose(codebook.responses[1],0)
