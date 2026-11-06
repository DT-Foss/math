from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from .delta_operator import DeltaLinearOperator


@dataclass(frozen=True)
class AtomicUpdateStats:
    events: int
    hits: int
    misses: int
    cache_size: int

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)


class AtomicTransitionResponseCache:
    """
    Exact response cache for quantized activation events.

    One event is `(input_direction, old_code, new_code)`. Its output response
    is compiled once through the model codec and reused without another weight
    decode/read. This is the instruction cache of the Temporal Event ISA.

    `levels` may be shared [L] or per-direction [input_width,L].
    """

    def __init__(
        self,
        operator: DeltaLinearOperator,
        levels: np.ndarray,
        *,
        capacity: int | None = None,
    ) -> None:
        level_array = np.asarray(levels)
        if level_array.ndim not in (1, 2):
            raise ValueError("levels must have shape [L] or [input_width,L]")
        if level_array.ndim == 2 and level_array.shape[0] != operator.input_width:
            raise ValueError("per-direction levels do not match operator input")
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive")
        self.operator = operator
        self.levels = level_array
        self.capacity = capacity
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    def _level(self, direction: int, code: int):
        if direction < 0 or direction >= self.operator.input_width:
            raise IndexError(direction)
        if self.levels.ndim == 1:
            if code < 0 or code >= len(self.levels):
                raise IndexError(code)
            return self.levels[code]
        if code < 0 or code >= self.levels.shape[1]:
            raise IndexError(code)
        return self.levels[direction, code]

    def response(self, direction: int, old_code: int, new_code: int) -> np.ndarray:
        key = (int(direction), int(old_code), int(new_code))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached

        delta = self._level(key[0], key[2]) - self._level(key[0], key[1])
        response = np.asarray(
            self.operator.columns(
                np.asarray([key[0]], dtype=np.int64),
                np.asarray([delta]),
            )
        ).copy()
        self.misses += 1
        self._cache[key] = response
        self._cache.move_to_end(key)
        if self.capacity is not None and len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return response

    def apply(self, old_codes: np.ndarray, new_codes: np.ndarray) -> tuple[np.ndarray, AtomicUpdateStats]:
        old = np.asarray(old_codes)
        new = np.asarray(new_codes)
        if old.shape != (self.operator.input_width,) or new.shape != old.shape:
            raise ValueError("code shape mismatch")
        changed = np.flatnonzero(old != new)
        hit_before = self.hits
        miss_before = self.misses
        total = np.zeros(self.operator.output_width, dtype=np.result_type(self.levels, float))
        for direction in changed:
            total += self.response(
                int(direction), int(old[direction]), int(new[direction])
            )
        return total, AtomicUpdateStats(
            events=int(len(changed)),
            hits=self.hits - hit_before,
            misses=self.misses - miss_before,
            cache_size=self.size,
        )


@dataclass
class TemporalEventLinearState:
    operator: DeltaLinearOperator
    codes: np.ndarray
    levels: np.ndarray
    output: np.ndarray
    response_cache: AtomicTransitionResponseCache

    @classmethod
    def from_codes(
        cls,
        operator: DeltaLinearOperator,
        codes: np.ndarray,
        levels: np.ndarray,
        *,
        cache_capacity: int | None = None,
    ) -> "TemporalEventLinearState":
        code_array = np.asarray(codes).copy()
        level_array = np.asarray(levels)
        if code_array.shape != (operator.input_width,):
            raise ValueError("code shape mismatch")
        if level_array.ndim == 1:
            decoded = level_array[code_array]
        elif level_array.ndim == 2 and level_array.shape[0] == operator.input_width:
            decoded = level_array[np.arange(operator.input_width), code_array]
        else:
            raise ValueError("level shape mismatch")
        cache = AtomicTransitionResponseCache(
            operator, level_array, capacity=cache_capacity
        )
        return cls(
            operator=operator,
            codes=code_array,
            levels=level_array,
            output=np.asarray(operator.full(decoded)).copy(),
            response_cache=cache,
        )

    def update_codes(self, new_codes: np.ndarray) -> tuple[np.ndarray, AtomicUpdateStats]:
        incoming = np.asarray(new_codes)
        delta, stats = self.response_cache.apply(self.codes, incoming)
        self.output = self.output + delta
        self.codes = incoming.copy()
        return self.output, stats
