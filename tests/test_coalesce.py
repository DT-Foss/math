import numpy as np
from wirkung_math.coalesce import RowReplacementLog


def test_repeated_writes_collapse_to_first_old_last_new():
    log=RowReplacementLog()
    a=np.array([1.,2.]); b=np.array([3.,4.]); c=np.array([5.,6.])
    log.add(7,a,b); log.add(7,b,c)
    idx,old,new=log.arrays()
    np.testing.assert_array_equal(idx,np.array([7]))
    np.testing.assert_array_equal(old,np.array([a]))
    np.testing.assert_array_equal(new,np.array([c]))
    log.add(7,c,a)
    assert log.support==0
