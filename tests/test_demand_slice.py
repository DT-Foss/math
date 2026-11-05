from wirkung_math.demand_slice import delta_demand_slice

def test_forward_dirty_intersects_backward_demand():
    edges=[("x","a"),("a","b"),("b","y1"),("a","c"),("c","y2"),("z","c")]
    s=delta_demand_slice(edges,changed=["x"],requested=["y2"])
    assert s.forward_dirty==frozenset({"x","a","b","c","y1","y2"})
    assert s.backward_demand==frozenset({"x","z","a","c","y2"})
    assert s.recompute==frozenset({"x","a","c","y2"})
