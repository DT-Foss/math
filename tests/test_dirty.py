import numpy as np
from wirkung_math.dirty import (
    GeometryCostModel,
    Interval,
    compose_stencils,
    connected_components,
    dirty_cone,
    dirty_cone_components,
    schedule_components_optimal,
    update_global_sum,
    update_projected_global_sum,
)


def test_compose_stencils():
    a=np.array([1.,2.,1.]); b=np.array([1.,-1.])
    np.testing.assert_allclose(compose_stencils(a,b),np.convolve(a,b))


def test_dirty_cone():
    got=dirty_cone(np.array([10]),30,radius=2,layers=3)
    np.testing.assert_array_equal(got,np.arange(4,17))


def test_component_geometry_preserves_separation():
    c=connected_components(np.array([3,4,20,21,40]))
    assert c == [Interval(3,5),Interval(20,22),Interval(40,41)]
    cones=dirty_cone_components(np.array([10,100]),200,radius=4,layers=2)
    assert cones == [Interval(2,19),Interval(92,109)]


def test_measured_merge_crossover_model():
    model=GeometryCostModel.from_two_component_gap_fit(
        merged_intercept_ms=5.822425785524378,
        merged_ms_per_gap=0.02049972202198457,
        separate_ms=9.257104014977813,
    )
    assert abs(model.crossover_gap-167.54755141411061) < 1e-9
    assert model.merge_pair(Interval(0,16),Interval(144,160))  # gap 128
    assert not model.merge_pair(Interval(0,16),Interval(272,288))  # gap 256


def test_optimal_scheduler_merges_near_and_splits_far():
    model=GeometryCostModel(launch_ms=4.0,cell_ms=0.02)  # gap*=200
    components=[Interval(0,16),Interval(144,160),Interval(500,516)]
    assert schedule_components_optimal(components,model)==[
        Interval(0,160),Interval(500,516)
    ]


def test_global_reduction_delta_exact():
    rng=np.random.default_rng(186)
    x=rng.normal(size=(100,8))
    idx=np.array([2,7,33,80])
    old=x[idx].copy(); new=old+rng.normal(scale=.1,size=old.shape)
    old_sum=x.sum(axis=0)
    x2=x.copy(); x2[idx]=new
    np.testing.assert_allclose(update_global_sum(old_sum,old,new),x2.sum(axis=0),rtol=1e-12,atol=1e-12)


def test_projected_global_reduction_delta_exact():
    rng=np.random.default_rng(187)
    x=rng.normal(size=(100,8)); W=rng.normal(size=(8,3))
    idx=np.array([2,7,33,80])
    old=x[idx].copy(); new=old+rng.normal(scale=.1,size=old.shape)
    old_g=x.sum(axis=0)@W
    x2=x.copy(); x2[idx]=new
    got=update_projected_global_sum(old_g,old,new,W)
    ref=x2.sum(axis=0)@W
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
