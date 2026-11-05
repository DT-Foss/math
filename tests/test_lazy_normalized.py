import numpy as np
from wirkung_math.bilinear import full_normalized_bilinear
from wirkung_math.lazy_normalized import LazyNormalizedKVState

def test_lazy_reconcile():
    rng=np.random.default_rng(120); n,d=64,8
    q=rng.normal(size=(n,d)); k=rng.normal(size=(n,d)); v=rng.normal(size=(n,d))
    s=LazyNormalizedKVState.from_qkv(q,k,v)
    idx=np.array([2,7,19])
    s.append_kv_update(idx,s.k[idx]+.02,s.v[idx]-.01)
    rows=np.array([1,3,11,30])
    got=s.query(rows)
    ref,_=full_normalized_bilinear(s.q[rows],s.k,s.v,s.scale)
    np.testing.assert_allclose(got,ref,rtol=1e-11,atol=1e-11)
