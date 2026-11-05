import numpy as np
from wirkung_math.dirty import compose_stencils, dirty_cone


def test_compose_stencils():
    a=np.array([1.,2.,1.]); b=np.array([1.,-1.])
    np.testing.assert_allclose(compose_stencils(a,b),np.convolve(a,b))


def test_dirty_cone():
    got=dirty_cone(np.array([10]),30,radius=2,layers=3)
    np.testing.assert_array_equal(got,np.arange(4,17))
