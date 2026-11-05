import numpy as np
from wirkung_math.reduction import SegmentReductionTree


def test_sum_tree():
    x=list(range(100))
    t=SegmentReductionTree(x,lambda a,b:a+b,0)
    assert t.aggregate==sum(x)
    t.replace(7,1000); x[7]=1000
    assert t.aggregate==sum(x)
    assert t.range_reduce(10,40)==sum(x[10:40])


def test_matrix_product_tree_noncommutative():
    rng=np.random.default_rng(90)
    mats=[rng.normal(size=(2,2)) for _ in range(17)]
    eye=np.eye(2)
    t=SegmentReductionTree(mats,lambda a,b:a@b,eye)
    ref=eye
    for m in mats: ref=ref@m
    np.testing.assert_allclose(t.aggregate,ref,rtol=1e-12,atol=1e-12)
    mats[5]=rng.normal(size=(2,2)); t.replace(5,mats[5])
    ref=eye
    for m in mats: ref=ref@m
    np.testing.assert_allclose(t.aggregate,ref,rtol=1e-12,atol=1e-12)
