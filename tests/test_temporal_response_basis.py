import numpy as np
from wirkung_math.temporal_response_basis import TemporalResponseBasis,split_response_delta


def test_temporal_response_basis_exact_on_training_subspace():
    rng=np.random.default_rng(330)
    din,dout,r=40,30,5
    W=rng.normal(size=(dout,din))
    latent=rng.normal(size=(r,din));coeff=rng.normal(size=(80,r));D=coeff@latent
    basis=TemporalResponseBasis.fit(W,D,rank=r)
    got=basis.apply_batch(D);ref=D@W.T
    np.testing.assert_allclose(got,ref,rtol=1e-10,atol=1e-10)
    assert basis.response_capture(W,D)>1-1e-12


def test_response_residual_closes_exactly():
    rng=np.random.default_rng(331)
    W=rng.normal(size=(24,32));D=rng.normal(size=(50,32))
    basis=TemporalResponseBasis.fit(W,D[:30],rank=8)
    delta=D[31]
    split=split_response_delta(basis,W,delta)
    np.testing.assert_allclose(split.approximate+split.residual,W@delta,rtol=1e-12,atol=1e-12)
    assert 0<=split.residual_fraction<=1.0000000001
