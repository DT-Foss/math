import numpy as np
from wirkung_math.update_ledger import LowRankUpdateLedger


def test_streaming_lowrank_ledger_matches_full_matrix():
    rng=np.random.default_rng(111)
    n=40
    z=rng.normal(size=(80,n)); A=z.T@z+20*np.eye(n)
    ledger=LowRankUpdateLedger(np.linalg.solve.__get__(A) if False else lambda b:np.linalg.solve(A,b))
    current=A.copy()
    for _ in range(5):
        U=rng.normal(scale=.03,size=(n,2));V=rng.normal(scale=.03,size=(n,2))
        ledger.append(U,V); current+=U@V.T
    rhs=rng.normal(size=(n,3))
    np.testing.assert_allclose(ledger.solve(rhs),np.linalg.solve(current,rhs),rtol=1e-10,atol=1e-10)
    before=ledger.nominal_rank
    ledger.compact()
    assert ledger.nominal_rank<=before
    np.testing.assert_allclose(ledger.solve(rhs),np.linalg.solve(current,rhs),rtol=1e-10,atol=1e-10)
