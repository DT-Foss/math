import numpy as np
from wirkung_math.graph import laplacian_edge_delta,laplacian_edges_delta


def test_laplacian_edge_update_exact():
    n=8
    L=np.zeros((n,n))
    i,j,w=2,6,3.5
    delta=laplacian_edge_delta(n,i,j,w)
    ref=L.copy()
    ref[i,i]+=w; ref[j,j]+=w; ref[i,j]-=w; ref[j,i]-=w
    np.testing.assert_allclose(delta.materialize(),ref)


def test_batch_edge_rank_bound():
    d=laplacian_edges_delta(10,[(0,1,1.0),(2,4,-.2),(7,9,.5)])
    assert d.rank_bound==3
