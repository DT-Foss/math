import numpy as np
from wirkung_math.temporal_census import measure_temporal_delta,effective_rank
from wirkung_math.temporal_delta_machine import (
    BeastDecodeBudget,DeltaPortfolio,LinearDeltaPlanner,LinearDeltaRoute,
    OrigamiDeltaMachineTarget,
)


def test_temporal_census_support_and_codes():
    prev=np.zeros(64);cur=prev.copy();cur[[1,2,40]]=np.array([1.0,-2.0,.5])
    pc=np.zeros(16,dtype=np.int8);cc=pc.copy();cc[[3,9]]=1
    r=measure_temporal_delta(prev,cur,block_size=16,previous_codes=pc,current_codes=cc)
    assert r.coordinate_support==3
    assert r.changed_blocks==2
    assert r.code_hamming==2
    assert r.top99_support<=3


def test_effective_rank_and_linear_route():
    rng=np.random.default_rng(230)
    A=rng.normal(size=(80,3));B=rng.normal(size=(64,3))
    assert effective_rank(A@B.T,.999)<=3
    planner=LinearDeltaPlanner(2560,2560)
    route=planner.choose(DeltaPortfolio(2560,coordinate_support=8,changed_blocks=2,block_size=32))
    assert route.route==LinearDeltaRoute.COORDINATE
    assert route.relative_to_full<.01


def test_beast_budget_target_projection():
    current=BeastDecodeBudget()
    assert abs(current.total_ms-134.13)<1e-9
    target=OrigamiDeltaMachineTarget()
    assert target.required_end_to_end_speedup>13
    projected=current.project(linear_speedup=20,normalized_speedup=20,io_speedup=20,other_speedup=3)
    assert projected.tokens_per_second>80
