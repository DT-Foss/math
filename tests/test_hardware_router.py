from wirkung_math.hardware_router import (
    ConsumerAwareKernelRouter, ConsumerRequirement, DirtyV100Calibration,
    KernelMaterializationCalibration,
)


def test_dirty_v100_crossover_matches_measured_bracket():
    r=DirtyV100Calibration()
    assert 354 < r.predicted_crossover_size < 359
    assert r.route(320)=="v100_full"
    assert r.route(384)=="cpu_delta"


def test_kernel_router_depends_on_consumer_representation():
    r=ConsumerAwareKernelRouter()
    # Full matrix: 8 edits already favor the V100.
    assert r.route(8,ConsumerRequirement.FULL_MATERIALIZATION)=="v100_full"
    # Same edit count, but a structured solve can consume rank<=2e directly.
    assert r.route(8,ConsumerRequirement.STRUCTURED_SOLVE)=="cpu_structured"


def test_materialized_rbf_edit_crossover():
    r=KernelMaterializationCalibration()
    assert 6.0 < r.predicted_crossover_edits < 7.5
