import numpy as np

from wirkung_math.quotient_benchmark import (
    make_observable_case, replay_observable_case,
    make_linear_chain_case, replay_linear_chain_case,
    make_finite_state_case, replay_finite_state_case, benchmark_finite_state_case,
)


def test_observable_benchmark_case_is_discovered_not_declared():
    case=make_observable_case(state_dim=32,observable_dim=3,input_dim=2,steps=20,seed=500)
    result=replay_observable_case(case)
    assert result["discovered_dimension"]==3
    assert result["full_dimension"]==32
    assert result["max_abs_error"] < 1e-9
    assert result["state_compression"] > 10


def test_linear_chain_benchmark_removes_dense_intermediate_exactly():
    case=make_linear_chain_case(
        input_dim=16,intermediate_dim=40,output_dim=12,
        quotient_dim=4,calls=10,seed=501,
    )
    result=replay_linear_chain_case(case)
    assert result["discovered_dimension"]==4
    assert result["dense_intermediate_dimension"]==40
    assert result["max_abs_error"] < 1e-9
    assert result["multiply_reduction"] > 1


def test_observable_benchmark_receipt_is_plain_json():
    import json
    case=make_observable_case(state_dim=16,observable_dim=2,input_dim=1,steps=4,seed=502)
    result=replay_observable_case(case)
    json.dumps(result)
    assert type(result["proof_passed"]) is bool


def test_finite_state_benchmark_discovers_behavioral_quotient():
    case=make_finite_state_case(
        physical_states=256,behavior_classes=8,actions=4,steps=100,seed=503
    )
    result=replay_finite_state_case(case)
    assert result["physical_states"]==256
    assert result["discovered_classes"]==8
    assert result["state_compression"]==32.0
    assert result["proof_passed"] is True
    assert result["trajectory_equal"] is True


def test_finite_state_benchmark_emits_timing_and_memory_receipt():
    case=make_finite_state_case(
        physical_states=128,behavior_classes=4,actions=2,steps=200,seed=504
    )
    result=benchmark_finite_state_case(case,repeats=2)
    assert result["proof_passed"] is True
    assert result["trajectory_equal"] is True
    assert result["full_ms"] > 0
    assert result["quotient_ms"] > 0
    assert result["table_byte_reduction"] > 1
