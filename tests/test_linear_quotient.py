import numpy as np

from wirkung_math.linear_quotient import (
    LinearConsumerQuotient, ObservableLinearQuotient
)


def test_linear_consumer_quotient_eliminates_nullspace_and_composes_upstream():
    C=np.array([[1.,0.,1.,0.],[2.,0.,2.,0.]])
    q=LinearConsumerQuotient.fit(C)
    assert q.dimension==1
    rng=np.random.default_rng(400)
    A=rng.normal(size=(4,7))
    u=rng.normal(size=7)
    z=q.project_state(A@u)
    np.testing.assert_allclose(q.consumer_from_state(z),C@A@u,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(
        q.reduced_consumer @ q.compose_upstream(A) @ u,
        C@A@u,rtol=1e-12,atol=1e-12,
    )


def test_zero_consumer_has_zero_dimensional_exact_state():
    C=np.zeros((3,5))
    q=LinearConsumerQuotient.fit(C)
    assert q.dimension==0
    z=q.project_state(np.arange(5.0))
    assert z.shape==(0,)
    np.testing.assert_array_equal(q.consumer_from_state(z),np.zeros(3))


def test_observable_quotient_discovers_two_state_machine_and_replays_exactly():
    A=np.array([
        [1.0,0.2,0,0,0,0],
        [0.0,0.9,0,0,0,0],
        [0,0,.8,.1,0,0],
        [0,0,0,.7,.2,0],
        [0,0,0,0,.6,.1],
        [0,0,0,0,0,.5],
    ])
    B=np.array([[1.0],[0.5],[3.0],[2.0],[1.0],[4.0]])
    C=np.array([[1.0,0.0,0,0,0,0]])
    q=ObservableLinearQuotient.fit(A,B,C)
    assert q.dimension==2
    assert q.invariance_residual < 1e-11

    rng=np.random.default_rng(401)
    x=rng.normal(size=6)
    z=q.project_state(x)
    for _ in range(30):
        np.testing.assert_allclose(q.output(z),C@x,rtol=1e-11,atol=1e-11)
        u=rng.normal(size=1)
        x=A@x+B@u
        z=q.step(z,u)


def test_observable_quotient_full_rank_when_every_state_is_visible():
    A=np.diag([.9,.8,.7])
    B=np.eye(3)
    C=np.eye(3)
    q=ObservableLinearQuotient.fit(A,B,C)
    assert q.dimension==3
    assert q.invariance_residual < 1e-12


def test_observable_discovery_uses_low_rank_krylov_closure():
    rng=np.random.default_rng(402)
    n,r=128,4
    Q,_=np.linalg.qr(rng.normal(size=(n,r)))
    P=np.roll(np.eye(r),-1,axis=1)*.9
    A=.3*np.eye(n)+Q@(P-.3*np.eye(r))@Q.T
    B=rng.normal(size=(n,2))
    C=np.zeros((1,r));C[0,0]=1.;C=C@Q.T
    q=ObservableLinearQuotient.fit(A,B,C)
    assert q.dimension==r
    assert q.discovery_method=="krylov_closure"
    assert q.closure_iterations <= r+1
    assert q.generated_basis_columns < n
