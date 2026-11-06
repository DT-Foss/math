import numpy as np

from wirkung_math.atomic_response import (
    AtomicTransitionResponseCache,
    TemporalEventLinearState,
)
from wirkung_math.delta_operator import DenseDeltaLinearOperator


def test_atomic_response_is_exact_and_reused():
    rng = np.random.default_rng(301)
    weight = rng.normal(size=(12, 16))
    levels = np.array([-1.0, 0.0, 1.0])
    cache = AtomicTransitionResponseCache(
        DenseDeltaLinearOperator(weight), levels
    )

    first = cache.response(5, 0, 2)
    second = cache.response(5, 0, 2)

    np.testing.assert_allclose(first, weight[:, 5] * 2.0)
    np.testing.assert_allclose(second, first)
    assert cache.misses == 1
    assert cache.hits == 1


def test_temporal_event_linear_matches_full_quantized_linear():
    rng = np.random.default_rng(302)
    weight = rng.normal(size=(20, 24))
    levels = np.array([-1.0, 0.0, 1.0])
    codes = rng.integers(0, len(levels), size=24, dtype=np.int8)
    state = TemporalEventLinearState.from_codes(
        DenseDeltaLinearOperator(weight), codes, levels
    )

    new_codes = codes.copy()
    new_codes[[1, 7, 19]] = np.array([2, 0, 1], dtype=np.int8)
    got, stats = state.update_codes(new_codes)

    np.testing.assert_allclose(got, weight @ levels[new_codes])
    assert stats.events == np.count_nonzero(new_codes != codes)
    assert stats.misses == stats.events


def test_lru_capacity_bounds_compiled_responses():
    weight = np.eye(8)
    cache = AtomicTransitionResponseCache(
        DenseDeltaLinearOperator(weight),
        np.array([-1.0, 0.0, 1.0]),
        capacity=2,
    )
    cache.response(0, 0, 1)
    cache.response(1, 0, 1)
    cache.response(2, 0, 1)
    assert cache.size == 2
