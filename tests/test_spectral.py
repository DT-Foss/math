import numpy as np
from wirkung_math.spectral import SparseDFTState, ButterflyFFTState


def test_sparse_dft_update_without_stored_basis():
    rng=np.random.default_rng(50)
    x=rng.normal(size=256)
    s=SparseDFTState.from_signal(x)
    idx=np.array([1,17,93]); new=rng.normal(size=3)
    s.replace(idx,new); x[idx]=new
    np.testing.assert_allclose(s.spectrum,np.fft.fft(x),rtol=1e-11,atol=1e-11)


def test_incremental_butterfly_dag_exact_and_shares_paths():
    rng=np.random.default_rng(51)
    n=1024; x=rng.normal(size=n)
    s=ButterflyFFTState(x)
    idx=np.array([2,3,4,5]); new=rng.normal(size=4)
    nodes=s.replace(idx,new); x[idx]=new
    np.testing.assert_allclose(s.spectrum,np.fft.fft(x),rtol=1e-11,atol=1e-11)
    assert nodes < len(idx)*n
