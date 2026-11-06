from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def transition_count(codes: np.ndarray) -> int:
    """Number of temporal code-change events over all coordinates."""
    c = np.asarray(codes)
    if c.ndim != 2:
        raise ValueError("codes must have shape [time, width]")
    if len(c) < 2:
        return 0
    return int(np.count_nonzero(c[1:] != c[:-1]))


@dataclass(frozen=True)
class TemporalQuantizationResult:
    codes: np.ndarray
    decoded: np.ndarray
    objective: float
    squared_error: float
    mse: float
    transitions: int

    @property
    def transition_fraction(self) -> float:
        opportunities = max(1, (self.codes.shape[0] - 1) * self.codes.shape[1])
        return self.transitions / opportunities


@dataclass(frozen=True)
class TransitionRegularizedQuantizer:
    """
    Globally optimal stateful quantizer for fixed scalar codebooks.

    For each coordinate independently, choose the complete code sequence that
    minimizes

        sum_t (x_t - level[code_t])^2
        + transition_penalty * 1[code_t != code_(t-1)].

    The decoded sequence defines the exact arithmetic consumed by the temporal
    runtime. Unlike hysteresis heuristics, this is the global Viterbi optimum
    for the specified distortion/compute objective.

    `levels` may be a shared vector [L] or one codebook per coordinate [D,L].
    `transition_penalty` may be a scalar or one penalty per coordinate [D].
    """
    levels: np.ndarray
    transition_penalty: float | np.ndarray

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels, dtype=np.float64)
        if levels.ndim not in (1, 2) or levels.shape[-1] < 1:
            raise ValueError("levels must have shape [L] or [width,L]")
        object.__setattr__(self, "levels", levels)

    def quantize(self, values: np.ndarray) -> TemporalQuantizationResult:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("values must have shape [time, width]")
        time_steps, width = x.shape
        if time_steps == 0:
            raise ValueError("at least one time step is required")

        if self.levels.ndim == 1:
            level_matrix = np.broadcast_to(self.levels, (width, len(self.levels)))
        else:
            if self.levels.shape[0] != width:
                raise ValueError("per-coordinate codebook width mismatch")
            level_matrix = self.levels
        code_count = level_matrix.shape[1]

        penalty = np.asarray(self.transition_penalty, dtype=np.float64)
        if penalty.ndim == 0:
            penalties = np.full(width, float(penalty), dtype=np.float64)
        elif penalty.shape == (width,):
            penalties = penalty
        else:
            raise ValueError("transition_penalty must be scalar or [width]")
        if np.any(penalties < 0):
            raise ValueError("transition penalties must be non-negative")

        codes = np.empty((time_steps, width), dtype=np.int64)
        objective = 0.0

        # Coordinates are independent under the additive distortion/event cost.
        for coordinate in range(width):
            levels = level_matrix[coordinate]
            emission = np.square(x[:, coordinate, None] - levels[None, :])
            dp = emission[0].copy()
            back = np.empty((time_steps, code_count), dtype=np.int64)
            back[0] = np.arange(code_count)
            p = penalties[coordinate]

            for step in range(1, time_steps):
                # transition[previous_code, current_code]
                transition = dp[:, None] + p
                transition[np.arange(code_count), np.arange(code_count)] = dp
                predecessor = np.argmin(transition, axis=0)
                dp = emission[step] + transition[predecessor, np.arange(code_count)]
                back[step] = predecessor

            code = int(np.argmin(dp))
            objective += float(dp[code])
            codes[-1, coordinate] = code
            for step in range(time_steps - 1, 0, -1):
                code = int(back[step, code])
                codes[step - 1, coordinate] = code

        decoded = np.take_along_axis(
            level_matrix[None, :, :], codes[:, :, None], axis=2
        )[:, :, 0]
        squared_error = float(np.square(x - decoded).sum())
        transitions = transition_count(codes)
        return TemporalQuantizationResult(
            codes=codes,
            decoded=decoded,
            objective=objective,
            squared_error=squared_error,
            mse=squared_error / x.size,
            transitions=transitions,
        )


def transition_distortion_frontier(
    values: np.ndarray,
    levels: np.ndarray,
    penalties: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[TemporalQuantizationResult, ...]:
    """Evaluate a complete distortion-versus-compute frontier."""
    return tuple(
        TransitionRegularizedQuantizer(levels, float(penalty)).quantize(values)
        for penalty in penalties
    )
