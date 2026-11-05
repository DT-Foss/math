import numpy as np
from wirkung_math.temporal_linear import (
    TemporalLinearState,BasisDeltaLinearState,BatchLowRankLinearState,
    TemporalRMSNormLinearState,
)


def test_sparse_temporal_linear_exact():
    rng=np.random.default_rng(200)
    W=rng.normal(size=(48,64));x=rng.normal(size=64)
    s=TemporalLinearState.from_input(W,x)
    idx=np.array([1,7,20,55]);new=x[idx]+rng.normal(scale=.1,size=len(idx))
    got=s.update_coordinates(idx,new)
    x[idx]=new
    np.testing.assert_allclose(got,W@x,rtol=1e-12,atol=1e-12)


def test_basis_plus_sparse_residual_exact():
    rng=np.random.default_rng(201)
    din,dout,r=80,50,6
    W=rng.normal(size=(dout,din));B=rng.normal(size=(din,r))
    c=rng.normal(size=r);res=rng.normal(size=din);x=B@c+res
    s=BasisDeltaLinearState.create(W,B,x,c)
    dc=rng.normal(scale=.05,size=r);idx=np.array([2,11,63]);dr=rng.normal(scale=.03,size=3)
    got=s.update(dc,idx,dr)
    x=x+B@dc;x[idx]+=dr
    np.testing.assert_allclose(got,W@x,rtol=1e-12,atol=1e-12)


def test_batch_lowrank_temporal_update_exact():
    rng=np.random.default_rng(202)
    batch,din,dout,r=24,40,36,3
    W=rng.normal(size=(dout,din));X=rng.normal(size=(batch,din))
    s=BatchLowRankLinearState.from_batch(W,X)
    A=rng.normal(scale=.05,size=(batch,r));B=rng.normal(scale=.05,size=(din,r))
    got=s.update_factors(A,B)
    X=X+A@B.T
    np.testing.assert_allclose(got,X@W.T,rtol=1e-12,atol=1e-12)


def test_rmsnorm_linear_sparse_update_exact():
    rng=np.random.default_rng(203)
    din,dout=96,72
    W=rng.normal(size=(dout,din));g=rng.normal(size=din);x=rng.normal(size=din)
    s=TemporalRMSNormLinearState.from_input(W,g,x)
    idx=np.array([3,10,44,80]);new=x[idx]+rng.normal(scale=.05,size=4)
    got=s.update_sparse(idx,new);x[idx]=new
    rho=np.sqrt(np.mean(x*x)+s.eps)
    ref=W@((x/rho)*g)
    np.testing.assert_allclose(got,ref,rtol=1e-11,atol=1e-11)
