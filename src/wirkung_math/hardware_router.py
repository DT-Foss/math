from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


class ConsumerRequirement(str, Enum):
    FULL_MATERIALIZATION = "full_materialization"
    STRUCTURED_SOLVE = "structured_solve"
    STRUCTURED_STATE = "structured_state"


@dataclass(frozen=True)
class DirtyV100Calibration:
    """
    Calibrated CPU-delta vs resident V100/cuDNN router for the measured
    12-layer, channels=16, patch=8 local residual graph.

    Small-area V100 model is fitted on 256, 320, 384, 512:
      T_gpu(N) = intercept_ms + ms_per_pixel * N^2.
    CPU dirty time is nearly flat over the same sweep.
    """
    gpu_intercept_ms: float = 0.150518844
    gpu_ms_per_pixel: float = 1.22620038e-5
    cpu_delta_ms: float = 1.723580137
    measured_lower_gpu_wins: int = 320
    measured_upper_cpu_wins: int = 384

    @property
    def predicted_crossover_size(self) -> float:
        return math.sqrt((self.cpu_delta_ms-self.gpu_intercept_ms)/self.gpu_ms_per_pixel)

    def gpu_ms(self, size: int) -> float:
        return self.gpu_intercept_ms+self.gpu_ms_per_pixel*(size*size)

    def route(self, size: int) -> str:
        return "cpu_delta" if self.cpu_delta_ms < self.gpu_ms(size) else "v100_full"


@dataclass(frozen=True)
class KernelMaterializationCalibration:
    """
    Resident V100 full RBF matrix versus CPU changed-row/column materialization,
    measured at n=8192,d=64.

    The edit crossover is interpolated between e=1 and e=8. This route is only
    for consumers that truly require the full dense matrix.
    """
    edit1_cpu_ms: float = 1.48
    edit1_gpu_ms: float = 5.54
    edit8_cpu_ms: float = 6.46
    edit8_gpu_ms: float = 5.56

    @property
    def predicted_crossover_edits(self) -> float:
        # Linear interpolation of D(e)=T_gpu-T_cpu between the two measured points.
        d1=self.edit1_gpu_ms-self.edit1_cpu_ms
        d8=self.edit8_gpu_ms-self.edit8_cpu_ms
        return 1.0+7.0*d1/(d1-d8)

    def route(self, edits: int) -> str:
        return "cpu_delta" if edits < self.predicted_crossover_edits else "v100_full"


@dataclass(frozen=True)
class KernelSolveCalibration:
    """
    RBF -> 8-RHS solve at n=4096. The structured CPU path never materializes
    the kernel delta: row/column update -> rank<=2e -> Woodbury solve.
    """
    speedup_e1: float = 16.57
    speedup_e4: float = 11.51
    speedup_e16: float = 4.54

    def observed_cpu_wins(self, edits: int) -> bool:
        return edits in {1,4,16}

    @property
    def linearized_ratio_crossover_edits(self) -> float:
        # Fit CPU/GPU ratio = a*e+b through the three supplied measurements.
        # Stored coefficients are deterministic from those measurements.
        a=0.01079076
        b=0.04696312
        return (1.0-b)/a


@dataclass(frozen=True)
class ConsumerAwareKernelRouter:
    full: KernelMaterializationCalibration = KernelMaterializationCalibration()
    solve: KernelSolveCalibration = KernelSolveCalibration()

    def route(self, edits: int, requirement: ConsumerRequirement) -> str:
        if requirement == ConsumerRequirement.FULL_MATERIALIZATION:
            return self.full.route(edits)
        if requirement in {ConsumerRequirement.STRUCTURED_SOLVE, ConsumerRequirement.STRUCTURED_STATE}:
            # Measured CPU wins through e=16; extrapolated ratio crossover is ~88
            # for this n/d/RHS configuration. The consumer contract permits
            # retaining the rank factor instead of materializing n^2 state.
            return "cpu_structured" if edits < self.solve.linearized_ratio_crossover_edits else "v100_full"
        raise ValueError(requirement)
