from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def compose_stencils(*kernels: np.ndarray) -> np.ndarray:
    """Compose linear 1-D local operators into one equivalent stencil."""
    if not kernels:
        return np.array([1.0])
    out = np.asarray(kernels[0], dtype=float)
    for k in kernels[1:]:
        out = np.convolve(out, np.asarray(k, dtype=float))
    return out


def dirty_cone(indices: np.ndarray, length: int, radius: int, layers: int = 1) -> np.ndarray:
    """Exact dependency cone for a stack of radius-r local operators."""
    reach = radius * layers
    mask = np.zeros(length, dtype=bool)
    for i in np.asarray(indices, dtype=int):
        lo, hi = max(0, i - reach), min(length, i + reach + 1)
        mask[lo:hi] = True
    return np.flatnonzero(mask)


@dataclass(frozen=True, order=True)
class Interval:
    """Half-open 1-D dirty component [start, stop)."""
    start: int
    stop: int

    def __post_init__(self):
        if self.start < 0 or self.stop < self.start:
            raise ValueError("invalid interval")

    @property
    def width(self) -> int:
        return self.stop - self.start

    def gap_to(self, other: "Interval") -> int:
        return max(0, other.start - self.stop)

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.start, other.start), max(self.stop, other.stop))


def connected_components(indices: np.ndarray) -> list[Interval]:
    """Turn dirty point support into contiguous connected components."""
    x = np.unique(np.asarray(indices, dtype=np.int64))
    if len(x) == 0:
        return []
    out: list[Interval] = []
    start = prev = int(x[0])
    for value in x[1:]:
        value = int(value)
        if value != prev + 1:
            out.append(Interval(start, prev + 1))
            start = value
        prev = value
    out.append(Interval(start, prev + 1))
    return out


def coalesce_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping or touching intervals."""
    if not intervals:
        return []
    items = sorted(intervals)
    out = [items[0]]
    for cur in items[1:]:
        prev = out[-1]
        if cur.start <= prev.stop:
            out[-1] = Interval(prev.start, max(prev.stop, cur.stop))
        else:
            out.append(cur)
    return out


def expand_components(
    components: list[Interval], length: int, radius: int, layers: int = 1
) -> list[Interval]:
    """Propagate dirty components through a local graph and preserve disconnected cones."""
    reach = radius * layers
    expanded = [
        Interval(max(0, c.start - reach), min(length, c.stop + reach))
        for c in components
    ]
    return coalesce_intervals(expanded)


def dirty_cone_components(
    indices: np.ndarray, length: int, radius: int, layers: int = 1
) -> list[Interval]:
    return expand_components(connected_components(indices), length, radius, layers)


@dataclass(frozen=True)
class GeometryCostModel:
    """
    Geometry-aware tile scheduler.

    Up to a partition-independent constant:
      T(partition) = launch_ms * number_of_tiles
                   + cell_ms * sum(tile_hull_width).

    Two adjacent components should therefore merge iff:
      cell_ms * gap < launch_ms.
    """
    launch_ms: float
    cell_ms: float

    def __post_init__(self):
        if self.launch_ms < 0 or self.cell_ms <= 0:
            raise ValueError("costs must be non-negative and cell_ms > 0")

    @property
    def crossover_gap(self) -> float:
        return self.launch_ms / self.cell_ms

    @classmethod
    def from_two_component_gap_fit(
        cls, merged_intercept_ms: float, merged_ms_per_gap: float, separate_ms: float
    ) -> "GeometryCostModel":
        # In T_merge = intercept + c*g, the excess cost of a second tile is
        # T_separate - intercept. That is the marginal launch/scheduling cost.
        return cls(
            launch_ms=max(0.0, separate_ms - merged_intercept_ms),
            cell_ms=merged_ms_per_gap,
        )

    def merge_pair(self, left: Interval, right: Interval) -> bool:
        return self.cell_ms * left.gap_to(right) <= self.launch_ms

    def partition_cost(self, tiles: list[Interval]) -> float:
        return self.launch_ms * len(tiles) + self.cell_ms * sum(t.width for t in tiles)


def schedule_components_optimal(
    components: list[Interval], model: GeometryCostModel
) -> list[Interval]:
    """
    Globally optimal contiguous partition under the launch+processed-width model.

    Dynamic programming considers every possible last merged group, so scheduling
    does not depend on a greedy merge order.
    """
    c = coalesce_intervals(components)
    n = len(c)
    if n <= 1:
        return c

    dp = [0.0] + [float("inf")] * n
    prev = [-1] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            hull = Interval(c[i].start, c[j - 1].stop)
            cost = dp[i] + model.launch_ms + model.cell_ms * hull.width
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = i

    out: list[Interval] = []
    j = n
    while j > 0:
        i = prev[j]
        out.append(Interval(c[i].start, c[j - 1].stop))
        j = i
    return list(reversed(out))


def update_global_sum(
    old_sum: np.ndarray, old_dirty: np.ndarray, new_dirty: np.ndarray
) -> np.ndarray:
    """Exact cached global reduction update: S' = S - sum(old_D) + sum(new_D)."""
    if old_dirty.shape != new_dirty.shape:
        raise ValueError("dirty slices must have the same shape")
    return old_sum - old_dirty.sum(axis=0) + new_dirty.sum(axis=0)


def update_projected_global_sum(
    old_projected_sum: np.ndarray,
    old_dirty: np.ndarray,
    new_dirty: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """
    Exact cached narrow global projection update without touching clean rows.

    If G = sum_i x_i W, then
      G' = G + sum_{i in D}(x'_i-x_i)W.
    """
    delta = (new_dirty - old_dirty).sum(axis=0)
    return old_projected_sum + delta @ projection
