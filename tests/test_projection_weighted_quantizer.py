import itertools
import numpy as np

from wirkung_math.temporal_quantizer import TransitionRegularizedQuantizer


def weighted_bruteforce(values, levels, weights, penalty):
    time_steps, width = values.shape
    best_codes = np.empty((time_steps, width), dtype=np.int64)
    best_objective = 0.0
    for coordinate in range(width):
        best = None
        for path in itertools.product(range(len(levels)), repeat=time_steps):
            decoded = levels[np.asarray(path)]
            cost = weights[coordinate] * float(
                np.square(values[:, coordinate] - decoded).sum()
            )
            cost += penalty * sum(a != b for a, b in zip(path[:-1], path[1:]))
            if best is None or cost < best[0]:
                best = (cost, np.asarray(path, dtype=np.int64))
        best_objective += best[0]
        best_codes[:, coordinate] = best[1]
    return best_objective, best_codes


def test_projection_weighted_viterbi_matches_weighted_global_optimum():
    values = np.array([
        [-0.8, -0.2],
        [-0.1, 0.4],
        [0.7, 0.9],
    ])
    levels = np.array([-1.0, 0.0, 1.0])
    weights = np.array([10.0, 0.1])
    penalty = 0.3

    result = TransitionRegularizedQuantizer(
        levels,
        transition_penalty=penalty,
        distortion_weights=weights,
    ).quantize(values)
    expected_objective, expected_codes = weighted_bruteforce(
        values, levels, weights, penalty
    )

    np.testing.assert_array_equal(result.codes, expected_codes)
    assert abs(result.objective - expected_objective) < 1e-12


def test_weight_from_linear_operator_uses_column_energy():
    weight = np.array([
        [3.0, 0.0],
        [4.0, 2.0],
    ])
    got = TransitionRegularizedQuantizer.linear_sensitivity_weights(weight)
    np.testing.assert_allclose(got, np.array([25.0, 4.0]))
