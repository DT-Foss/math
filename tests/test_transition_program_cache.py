import numpy as np

from wirkung_math.atomic_response import (
    AtomicTransitionResponseCache,
    TransitionProgramCache,
)
from wirkung_math.delta_operator import DenseDeltaLinearOperator


def test_program_cache_promotes_repeated_transition_program():
    rng = np.random.default_rng(303)
    weight = rng.normal(size=(10, 12))
    levels = np.array([-1.0, 0.0, 1.0])
    atomic = AtomicTransitionResponseCache(
        DenseDeltaLinearOperator(weight), levels
    )
    programs = TransitionProgramCache(atomic, promote_after=2)

    old = np.zeros(12, dtype=np.int8)
    new = old.copy()
    new[[1, 5, 9]] = np.array([2, 1, 2], dtype=np.int8)

    first, first_stats = programs.response(old, new)
    second, second_stats = programs.response(old, new)
    third, third_stats = programs.response(old, new)

    expected = weight @ (levels[new] - levels[old])
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)
    np.testing.assert_allclose(third, expected)
    assert not first_stats.macro_hit
    assert not second_stats.macro_hit
    assert third_stats.macro_hit
    assert programs.macro_size == 1


def test_unique_programs_still_reuse_atomic_responses():
    weight = np.eye(6)
    levels = np.array([-1.0, 0.0, 1.0])
    atomic = AtomicTransitionResponseCache(
        DenseDeltaLinearOperator(weight), levels
    )
    programs = TransitionProgramCache(atomic, promote_after=10)

    old = np.zeros(6, dtype=np.int8)
    a = old.copy(); a[[1, 2]] = 2
    b = old.copy(); b[[1, 4]] = 2
    programs.response(old, a)
    _, stats = programs.response(old, b)

    # Direction 1 transition is reused despite the overall program being new.
    assert stats.atomic_hits == 1
    assert stats.atomic_misses == 1
