from wirkung_math.cacheless_local import LocalGraphDependency

def test_crop():
    g=LocalGraphDependency((1,)*12,16)
    assert g.input_crop_side(8)==56
