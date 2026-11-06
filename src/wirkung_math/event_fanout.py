from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from .delta_operator import DeltaLinearOperator


@dataclass(frozen=True)
class FanoutUpdateStats:
    events: int
    cache_hits: int
    cache_misses: int
    cache_size: int

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(1, self.cache_hits + self.cache_misses)


class FusedEventFanoutCache:
    """
    One compiled temporal instruction updates every linear consumer of a state.

    An event `(direction, old_code, new_code)` is translated once into the
    concatenated output response for Q/K/V/router or gate/up consumers. Reuse
    requires neither another model-page decode nor separate operator dispatches.
    """

    def __init__(
        self,
        operators: dict[str, DeltaLinearOperator],
        levels: np.ndarray,
        *,
        capacity: int | None = None,
    ) -> None:
        if not operators:
            raise ValueError("at least one operator is required")
        widths = {operator.input_width for operator in operators.values()}
        if len(widths) != 1:
            raise ValueError("all consumers must share an input width")
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive")

        self.operators = dict(operators)
        self.names = tuple(self.operators)
        self.input_width = widths.pop()
        self.levels = np.asarray(levels)
        if self.levels.ndim not in (1, 2):
            raise ValueError("levels must have shape [L] or [input_width,L]")
        if self.levels.ndim == 2 and self.levels.shape[0] != self.input_width:
            raise ValueError("per-direction codebook width mismatch")
        self.capacity = capacity

        self.slices: dict[str, slice] = {}
        start = 0
        for name, operator in self.operators.items():
            stop = start + operator.output_width
            self.slices[name] = slice(start, stop)
            start = stop
        self.output_width = start
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    def _level(self, direction: int, code: int):
        if direction < 0 or direction >= self.input_width:
            raise IndexError(direction)
        if self.levels.ndim == 1:
            return self.levels[code]
        return self.levels[direction, code]

    def response(self, direction: int, old_code: int, new_code: int) -> np.ndarray:
        key = (int(direction), int(old_code), int(new_code))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached

        delta = self._level(key[0], key[2]) - self._level(key[0], key[1])
        index = np.asarray([key[0]], dtype=np.int64)
        value = np.asarray([delta])
        response = np.concatenate(
            [
                np.asarray(operator.columns(index, value)).reshape(-1)
                for operator in self.operators.values()
            ]
        )
        self.misses += 1
        self._cache[key] = response
        self._cache.move_to_end(key)
        if self.capacity is not None and len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return response

    def split(self, response: np.ndarray) -> dict[str, np.ndarray]:
        value = np.asarray(response)
        if value.shape != (self.output_width,):
            raise ValueError("response shape mismatch")
        return {
            name: value[output_slice]
            for name, output_slice in self.slices.items()
        }


@dataclass
class TemporalEventFanoutState:
    codes: np.ndarray
    levels: np.ndarray
    outputs: dict[str, np.ndarray]
    response_cache: FusedEventFanoutCache

    @classmethod
    def from_codes(
        cls,
        operators: dict[str, DeltaLinearOperator],
        codes: np.ndarray,
        levels: np.ndarray,
        *,
        cache_capacity: int | None = None,
    ) -> "TemporalEventFanoutState":
        cache = FusedEventFanoutCache(
            operators, levels, capacity=cache_capacity
        )
        code_array = np.asarray(codes).copy()
        level_array = np.asarray(levels)
        if code_array.shape != (cache.input_width,):
            raise ValueError("code shape mismatch")
        if level_array.ndim == 1:
            decoded = level_array[code_array]
        else:
            decoded = level_array[np.arange(cache.input_width), code_array]
        outputs = {
            name: np.asarray(operator.full(decoded)).copy()
            for name, operator in operators.items()
        }
        return cls(code_array, level_array, outputs, cache)

    def update_codes(
        self, new_codes: np.ndarray
    ) -> tuple[dict[str, np.ndarray], FanoutUpdateStats]:
        incoming = np.asarray(new_codes)
        if incoming.shape != self.codes.shape:
            raise ValueError("code shape mismatch")
        changed = np.flatnonzero(incoming != self.codes)
        hit_before = self.response_cache.hits
        miss_before = self.response_cache.misses
        total = np.zeros(
            self.response_cache.output_width,
            dtype=np.result_type(self.levels, float),
        )
        for direction in changed:
            total += self.response_cache.response(
                int(direction),
                int(self.codes[direction]),
                int(incoming[direction]),
            )
        split = self.response_cache.split(total)
        for name, delta in split.items():
            self.outputs[name] = self.outputs[name] + delta
        self.codes = incoming.copy()
        return self.outputs, FanoutUpdateStats(
            events=int(len(changed)),
            cache_hits=self.response_cache.hits - hit_before,
            cache_misses=self.response_cache.misses - miss_before,
            cache_size=self.response_cache.size,
        )
