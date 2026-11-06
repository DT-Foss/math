import numpy as np

from wirkung_math.delta_operator import DenseDeltaLinearOperator
from wirkung_math.event_fanout import TemporalEventFanoutState


def test_event_fanout_updates_multiple_consumers_exactly():
    rng = np.random.default_rng(304)
    width = 32
    weights = {
        "q": rng.normal(size=(12, width)),
        "k": rng.normal(size=(8, width)),
        "v": rng.normal(size=(8, width)),
        "router": rng.normal(size=(5, width)),
    }
    operators = {
        name: DenseDeltaLinearOperator(weight)
        for name, weight in weights.items()
    }
    levels = np.array([-1.0, 0.0, 1.0])
    codes = rng.integers(0, len(levels), size=width, dtype=np.int8)
    state = TemporalEventFanoutState.from_codes(operators, codes, levels)

    new_codes = codes.copy()
    new_codes[[2, 9, 20]] = np.array([2, 0, 1], dtype=np.int8)
    got, stats = state.update_codes(new_codes)

    decoded = levels[new_codes]
    for name, weight in weights.items():
        np.testing.assert_allclose(got[name], weight @ decoded)
    assert stats.events == np.count_nonzero(new_codes != codes)


def test_fanout_atomic_instruction_is_reused_across_time():
    weight = np.eye(6)
    operators = {
        "a": DenseDeltaLinearOperator(weight),
        "b": DenseDeltaLinearOperator(2.0 * weight),
    }
    levels = np.array([-1.0, 0.0, 1.0])
    codes = np.zeros(6, dtype=np.int8)
    state = TemporalEventFanoutState.from_codes(operators, codes, levels)

    up = codes.copy(); up[3] = 2
    state.update_codes(up)
    state.update_codes(codes)
    _, stats = state.update_codes(up)

    # The repeated 0->2 event is served from the fused response cache.
    assert stats.cache_hits == 1
    assert stats.cache_misses == 0
