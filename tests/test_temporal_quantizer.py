import itertools
import numpy as np

from wirkung_math.temporal_quantizer import (
    TransitionRegularizedQuantizer,
    transition_count,
)


def brute_force(values, levels, penalty):
    best = None
    for path in itertools.product(range(len(levels)), repeat=len(values)):
        decoded = levels[np.asarray(path)]
        cost = float(np.square(values - decoded).sum())
        cost += penalty * sum(a != b for a, b in zip(path[:-1], path[1:]))
        if best is None or cost < best[0]:
            best = (cost, np.asarray(path, dtype=np.int64))
    return best


def test_viterbi_quantizer_matches_global_optimum():
    values = np.array([-0.8, -0.1, 0.2, 0.9], dtype=np.float64)
    levels = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    penalty = 0.3
    quantizer = TransitionRegularizedQuantizer(levels, transition_penalty=penalty)

    result = quantizer.quantize(values[:, None])
    expected_cost, expected_codes = brute_force(values, levels, penalty)

    np.testing.assert_array_equal(result.codes[:, 0], expected_codes)
    assert abs(result.objective - expected_cost) < 1e-12


def test_zero_penalty_is_nearest_level_quantization():
    values = np.array([[-0.9, 0.2], [-0.4, 0.8], [0.7, -0.1]])
    levels = np.array([-1.0, 0.0, 1.0])
    result = TransitionRegularizedQuantizer(levels, transition_penalty=0.0).quantize(values)
    expected = np.argmin(np.abs(values[..., None] - levels), axis=-1)
    np.testing.assert_array_equal(result.codes, expected)


def test_transition_penalty_reduces_code_events_with_bounded_distortion():
    values = np.array([
        [-0.51], [-0.49], [-0.52], [-0.48], [-0.53], [-0.47],
    ])
    levels = np.array([-1.0, 0.0, 1.0])
    nearest = TransitionRegularizedQuantizer(levels, transition_penalty=0.0).quantize(values)
    stable = TransitionRegularizedQuantizer(levels, transition_penalty=0.05).quantize(values)

    assert transition_count(stable.codes) < transition_count(nearest.codes)
    assert stable.mse <= nearest.mse + 0.05
