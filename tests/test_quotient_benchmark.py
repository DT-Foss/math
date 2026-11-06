import numpy as np

from wirkung_math.quotient_benchmark import (
    make_observable_case, replay_observable_case,
    make_linear_chain_case, replay_linear_chain_case,
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
