import numpy as np
from wirkung_math.delta_layout import (
    ColumnPackedLinearOperator,ColumnPackedTemporalLinearState,
    ColumnPackedRMSNormLinearState,ColumnPackedRMSNormMultiLinearState,
)


def test_column_packed_temporal_linear_exact():
    rng=np.random.default_rng(290)
    W=rng.normal(size=(50,70));x=rng.normal(size=70)
    op=ColumnPackedLinearOperator.from_weight(W)
    state=ColumnPackedTemporalLinearState.from_input(op,x)
    idx=np.array([1,9,30]);delta=rng.normal(scale=.1,size=3)
    got=state.update_delta(idx,delta);x[idx]+=delta
    np.testing.assert_allclose(got,W@x,rtol=1e-12,atol=1e-12)


def test_column_packed_rmsnorm_exact():
    rng=np.random.default_rng(291)
    W=rng.normal(size=(48,64));x=rng.normal(size=64);g=rng.normal(size=64)
    state=ColumnPackedRMSNormLinearState.from_input(ColumnPackedLinearOperator.from_weight(W),g,x)
    idx=np.array([2,11,40]);new=x[idx]+rng.normal(scale=.05,size=3)
    got=state.update_sparse(idx,new);x[idx]=new
    rho=np.sqrt(np.mean(x*x)+state.eps)
    np.testing.assert_allclose(got,W@((x/rho)*g),rtol=1e-11,atol=1e-11)


def test_column_packed_multi_exact():
    rng=np.random.default_rng(292)
    x=rng.normal(size=50);g=rng.normal(size=50)
    weights={"q":rng.normal(size=(20,50)),"k":rng.normal(size=(16,50))}
    ops={name:ColumnPackedLinearOperator.from_weight(W) for name,W in weights.items()}
    state=ColumnPackedRMSNormMultiLinearState.from_input(ops,g,x)
    idx=np.array([3,7]);new=x[idx]+np.array([.1,-.05])
    got=state.update_sparse(idx,new);x[idx]=new
    rho=np.sqrt(np.mean(x*x)+state.eps);normed=(x/rho)*g
    for name,W in weights.items():
        np.testing.assert_allclose(got[name],W@normed,rtol=1e-11,atol=1e-11)
