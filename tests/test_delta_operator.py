import numpy as np
from wirkung_math.delta_operator import (
    DenseDeltaLinearOperator,CallbackDeltaLinearOperator,
    TemporalOperatorState,BasisOperatorState,
)


def test_dense_and_callback_delta_operator_exact():
    rng=np.random.default_rng(270)
    W=rng.normal(size=(40,60));x=rng.normal(size=60)
    dense=DenseDeltaLinearOperator(W)
    callback=CallbackDeltaLinearOperator(
        60,40,
        lambda v:W@v,
        lambda idx,d:W[:,idx]@d,
        lambda B:W@B,
    )
    for op in (dense,callback):
        state=TemporalOperatorState.from_input(op,x)
        idx=np.array([1,8,30]);delta=rng.normal(scale=.1,size=3)
        got=state.update_delta(idx,delta)
        refx=x.copy();refx[idx]+=delta
        np.testing.assert_allclose(got,W@refx,rtol=1e-12,atol=1e-12)


def test_basis_operator_state_exact():
    rng=np.random.default_rng(271)
    W=rng.normal(size=(30,50));B=rng.normal(size=(50,5));c=rng.normal(size=5)
    residual=rng.normal(size=50);x=B@c+residual
    state=BasisOperatorState.create(DenseDeltaLinearOperator(W),B,c,x)
    dc=rng.normal(scale=.05,size=5)
    got=state.update_coefficients(dc)
    np.testing.assert_allclose(got,W@(x+B@dc),rtol=1e-12,atol=1e-12)
