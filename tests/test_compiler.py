import numpy as np
from wirkung_math.compiler import DirtyGraphCompiler
from wirkung_math.dirty import GeometryCostModel, Interval


def test_dirty_graph_compiler_end_to_end():
    model=GeometryCostModel(launch_ms=4.0,cell_ms=.02)
    compiler=DirtyGraphCompiler(length=1000,radius=2,layers=4,cost_model=model)
    # First two components remain distinct after cone propagation but their
    # post-cone gap is cheap enough to fill; the third stays separate.
    plan=compiler.compile(np.array([100,101,260,261,800]))
    assert plan.source_components == (
        Interval(100,102),Interval(260,262),Interval(800,801)
    )
    assert plan.cone_components == (
        Interval(92,110),Interval(252,270),Interval(792,809)
    )
    assert plan.scheduled_tiles == (
        Interval(92,270),Interval(792,809)
    )
    assert plan.tile_count == 2
    assert plan.scheduled_cells == 195
