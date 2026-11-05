import numpy as np
from wirkung_math.temporal_multilinear import TemporalRMSNormMultiLinearState
from wirkung_math.temporal_quantized import QuantizedVectorState,QuantizedTemporalLinearState


def test_rmsnorm_multi_projection_exact():
    rng=np.random.default_rng(240)
    d=64;x=rng.normal(size=d);g=rng.normal(size=d)
    weights={"q":rng.normal(size=(32,d)),"k":rng.normal(size=(24,d)),"router":rng.normal(size=(10,d))}
    s=TemporalRMSNormMultiLinearState.from_input(weights,g,x)
    idx=np.array([1,8,20,55]);new=x[idx]+rng.normal(scale=.04,size=4)
    got=s.update_sparse(idx,new);x[idx]=new
    rho=np.sqrt(np.mean(x*x)+s.eps);normed=(x/rho)*g
    for name,W in weights.items():
        np.testing.assert_allclose(got[name],W@normed,rtol=1e-11,atol=1e-11)


def test_quantized_temporal_linear_exact():
    rng=np.random.default_rng(241)
    d,o=80,48
    levels=np.array([-1.0,-.5,0.0,.5,1.0])
    codes=rng.integers(0,len(levels),size=d,dtype=np.int16)
    W=rng.normal(size=(o,d))
    s=QuantizedTemporalLinearState.from_codes(W,codes,levels)
    idx=np.array([2,9,40,70]);new_codes=np.array([4,0,3,1],dtype=np.int16)
    got,changed=s.update_codes(new_codes,idx)
    codes[idx]=new_codes
    np.testing.assert_allclose(got,W@levels[codes],rtol=1e-12,atol=1e-12)
    assert set(changed.tolist()).issubset(set(idx.tolist()))


def test_bitplane_flip_count():
    old=np.array([0,1,2,3],dtype=np.uint8)
    new=np.array([1,1,3,0],dtype=np.uint8)
    # 0->1:1; 2->3:1; 3->0:2
    assert QuantizedVectorState.bitplane_flips(old,new,2)==4
