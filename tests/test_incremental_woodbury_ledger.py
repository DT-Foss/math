import numpy as np
from wirkung_math.update_ledger import IncrementalWoodburyLedger


def test_incremental_woodbury_core_stream():
    rng=np.random.default_rng(112)
    n=50
    z=rng.normal(size=(100,n)); A=z.T@z+20*np.eye(n)
    ledger=IncrementalWoodburyLedger(lambda b:np.linalg.solve(A,b))
    current=A.copy()
    rhs=rng.normal(size=(n,4))
    for step in range(8):
        U=rng.normal(scale=.02,size=(n,2));V=rng.normal(scale=.02,size=(n,2))
        ledger.append(U,V);current+=U@V.T
        np.testing.assert_allclose(ledger.solve(rhs),np.linalg.solve(current,rhs),rtol=1e-10,atol=1e-10)
        assert ledger.rank==2*(step+1)
