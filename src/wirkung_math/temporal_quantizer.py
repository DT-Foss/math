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

        sum_t distortion_weight_j * (x_tj - level_j[code_tj])^2
        + transition_penalty_j * 1[code_tj != code_(t-1)j].

    The decoded sequence defines the arithmetic consumed by the temporal
    runtime. Unlike hysteresis heuristics, this is the global Viterbi optimum
    for the specified downstream-distortion/compute objective.

    `levels` may be a shared vector [L] or one codebook per coordinate [D,L].
    Penalties and distortion weights may be scalars or one value per coordinate.
    """
    levels: np.ndarray
    transition_penalty: float | np.ndarray
    distortion_weights: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels, dtype=np.float64)
        if levels.ndim not in (1, 2) or levels.shape[-1] < 1:
            raise ValueError("levels must have shape [L] or [width,L]")
        object.__setattr__(self, "levels", levels)

    @staticmethod
    def linear_sensitivity_weights(weight: np.ndarray) -> np.ndarray:
        """
        Diagonal downstream metric induced by y = W x.

        The exact projection-error energy is e^T W^T W e. Its diagonal is the
        squared norm of each input-direction response and gives a separable
        Viterbi objective while preserving the real consumer's sensitivity.
        """
        matrix = np.asarray(weight, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("weight must be a matrix")
        return np.square(matrix).sum(axis=0)

    @staticmethod
    def _coordinate_values(
        value: float | np.ndarray,
        width: int,
        *,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            result = np.full(width, float(array), dtype=np.float64)
        elif array.shape == (width,):
            result = array
        else:
            raise ValueError(f"{name} must be scalar or [width]")
        if np.any(result < 0):
            raise ValueError(f"{name} values must be non-negative")
        return result

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

        penalties = self._coordinate_values(
            self.transition_penalty, width, name="transition_penalty"
        )
        distortion_weights = self._coordinate_values(
            self.distortion_weights, width, name="distortion_weights"
        )

        codes = np.empty((time_steps, width), dtype=np.int64)
        objective = 0.0

        # Coordinates are independent under this separable consumer metric.
        for coordinate in range(width):
            levels = level_matrix[coordinate]
            emission = distortion_weights[coordinate] * np.square(
                x[:, coordinate, None] - levels[None, :]
            )
            dp = emission[0].copy()
            back = np.empty((time_steps, code_count), dtype=np.int64)
            back[0] = np.arange(code_count)
            penalty = penalties[coordinate]

            for step in range(1, time_steps):
                # transition[previous_code, current_code]
                transition = np.broadcast_to(
                    dp[:, None] + penalty, (code_count, code_count)
                ).copy()
                transition[np.arange(code_count), np.arange(code_count)] = dp
                predecessor = np.argmin(transition, axis=0)
                dp = emission[step] + transition[
                    predecessor, np.arange(code_count)
                ]
                back[step] = predecessor

            code = int(np.argmin(dp))
            objective += float(dp[code])
            codes[-1, coordinate] = code
            for step in range(time_steps - 1, 0, -1):
                code = int(back[step, code])
                codes[step - 1, coordinate] = code

        decoded = np.take_along_axis(
            np.broadcast_to(
                level_matrix[None, :, :],
                (time_steps, width, code_count),
            ),
            codes[:, :, None],
            axis=2,
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
    *,
    distortion_weights: float | np.ndarray = 1.0,
) -> tuple[TemporalQuantizationResult, ...]:
    """Evaluate a complete downstream-distortion-versus-compute frontier."""
    return tuple(
        TransitionRegularizedQuantizer(
            levels,
            float(penalty),
            distortion_weights=distortion_weights,
        ).quantize(values)
        for penalty in penalties
    )
