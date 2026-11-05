import numpy as np
from wirkung_math.statistics import SufficientStatisticsState, NormalEquationsState


def test_sufficient_statistics_replace_insert_delete():
    rng=np.random.default_rng(20)
    x=rng.normal(size=(100,8))
    state=SufficientStatisticsState.from_rows(x)
    ids=np.array([2,7,31,80])
    old=x[ids].copy(); new=old+rng.normal(scale=.1,size=old.shape)
    state.replace_rows(old,new); x[ids]=new
    np.testing.assert_allclose(state.sum,x.sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.sumsq,(x*x).sum(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.gram,x.T@x,rtol=1e-12,atol=1e-12)
    extra=rng.normal(size=(5,8)); state.insert_rows(extra); x=np.vstack([x,extra])
    state.delete_rows(extra); x=x[:-5]
    np.testing.assert_allclose(state.covariance,(x.T@x)/len(x)-np.outer(x.mean(0),x.mean(0)),rtol=1e-12,atol=1e-12)


def test_normal_equations_replacement():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(200,12)); y=rng.normal(size=(200,3))
    state=NormalEquationsState.from_xy(x,y)
    ids=np.array([1,8,50])
    ox=x[ids].copy(); oy=y[ids].copy()
    nx=ox+rng.normal(scale=.1,size=ox.shape); ny=oy+rng.normal(scale=.1,size=oy.shape)
    state.replace_rows(ox,nx,oy,ny); x[ids]=nx; y[ids]=ny
    np.testing.assert_allclose(state.gram,x.T@x,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(state.cross,x.T@y,rtol=1e-12,atol=1e-12)
