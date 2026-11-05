from wirkung_math.dirty2d import Rect
from wirkung_math.local_demand import local_chain_delta_demand,required_area_sum

def test_far_requested_region_needs_no_changed_work():
    p=local_chain_delta_demand(Rect(10,10,18,18),Rect(200,200,208,208),256,256,(1,)*12)
    assert required_area_sum(p)==0

def test_overlapping_demand_is_smaller_than_full_dirty_chain():
    p=local_chain_delta_demand(Rect(100,100,108,108),Rect(115,115,123,123),256,256,(1,)*12)
    assert required_area_sum(p)>0
    full=sum(x.forward_dirty.area for x in p[1:])
    assert required_area_sum(p)<full
