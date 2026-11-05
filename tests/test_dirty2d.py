import numpy as np
from wirkung_math.dirty2d import (
    AreaCostModel,DirtyGraphCompiler2D,Rect,schedule_rectangles_optimal,
)


def test_exact_2d_partition_merges_near_not_far():
    model=AreaCostModel(launch_ms=100.0,cell_ms=1.0)
    rects=[Rect(0,0,10,10),Rect(0,12,10,22),Rect(100,100,110,110)]
    tiles=schedule_rectangles_optimal(rects,model)
    assert tiles==[Rect(0,0,10,22),Rect(100,100,110,110)]


def test_2d_dirty_compiler_propagates_patch_geometry():
    model=AreaCostModel(launch_ms=1.0,cell_ms=.001)
    compiler=DirtyGraphCompiler2D(256,256,radius=1,layers=12,cost_model=model)
    pts=np.array([[100,100],[100,101],[101,100],[101,101]])
    plan=compiler.compile_points(pts)
    assert plan.source==(Rect(100,100,102,102),)
    assert plan.cones==(Rect(88,88,114,114),)
    assert plan.scheduled_area==26*26
