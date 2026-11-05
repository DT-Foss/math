import numpy as np
from wirkung_math.temporal_basis import TemporalDeltaBasis,exact_batch_factorization


def test_temporal_basis_exact_residual():
    rng=np.random.default_rng(280)
    latent=rng.normal(size=(6,4));coeff=rng.normal(size=(200,4));D=coeff@latent.T
    basis=TemporalDeltaBasis.fit(D,4)
    x=D[7]
    encoded=basis.encode(x,residual_atol=1e-10)
    recon=basis.basis@encoded.coefficients
    recon[encoded.residual_indices]+=encoded.residual_values
    np.testing.assert_allclose(recon,x,rtol=1e-11,atol=1e-11)
    assert encoded.captured_energy>0.999999


def test_exact_cross_lane_factorization():
    rng=np.random.default_rng(281)
    A0=rng.normal(size=(80,5));B0=rng.normal(size=(64,5));D=A0@B0.T
    A,B=exact_batch_factorization(D)
    assert A.shape[1]<=5
    np.testing.assert_allclose(A@B.T,D,rtol=1e-10,atol=1e-10)
