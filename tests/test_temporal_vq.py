import numpy as np
from wirkung_math.temporal_vq import (
    ProductTemporalCodebook,ResponseCodedLinearState,product_code_rate_distortion,
)


def test_response_coded_linear_matches_decoded_state():
    rng=np.random.default_rng(310)
    trace=rng.normal(size=(40,32))
    codebook=ProductTemporalCodebook.fit(trace[:20],block_size=8,codes=8,seed=1)
    codes,decoded=codebook.encode_trace(trace[20:],transition_penalty=1.0)
    W=rng.normal(size=(24,32));bias=rng.normal(size=24)
    state=ResponseCodedLinearState.compile(W,codebook,codes[0],bias)
    np.testing.assert_allclose(state.y,W@decoded[0]+bias,rtol=1e-11,atol=1e-11)
    for t in range(1,len(codes)):
        got,_=state.update(codes[t])
        np.testing.assert_allclose(got,W@decoded[t]+bias,rtol=1e-11,atol=1e-11)


def test_transition_penalty_reduces_code_changes():
    rng=np.random.default_rng(311)
    trace=np.cumsum(rng.normal(scale=.05,size=(60,32)),axis=0)
    codebook=ProductTemporalCodebook.fit(trace[:30],block_size=8,codes=8,seed=2)
    plain=product_code_rate_distortion(trace[30:],codebook,transition_penalty=0.0)
    sticky=product_code_rate_distortion(trace[30:],codebook,transition_penalty=8.0)
    assert sticky["changed_block_fraction_median"]<=plain["changed_block_fraction_median"]
    assert sticky["relative_l2_error"]>=plain["relative_l2_error"]


def test_response_storage_equal_when_codes_equal_block_width():
    rng=np.random.default_rng(312)
    trace=rng.normal(size=(24,32))
    codebook=ProductTemporalCodebook.fit(trace,block_size=8,codes=8)
    assert codebook.response_storage_ratio==1.0
