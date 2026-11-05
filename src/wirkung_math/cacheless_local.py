from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LocalGraphDependency:
    """
    Geometry and state footprint for a local feed-forward graph.

    For total receptive radius R and an input edit patch of side p:
      final dirty output side = p + 2R
      input crop sufficient to reconstruct that dirty output = p + 4R

    This permits exact old/new local recomputation from the input crop without
    storing every intermediate full activation, provided the graph is strictly
    local and the requested consumer only needs the dirty output / an
    incrementally repairable downstream statistic.
    """
    radius_per_layer: tuple[int,...]
    channels: int
    bytes_per_value: int = 4

    @property
    def layers(self) -> int:
        return len(self.radius_per_layer)

    @property
    def total_radius(self) -> int:
        return int(sum(self.radius_per_layer))

    def dirty_output_side(self, patch_side: int) -> int:
        return int(patch_side + 2*self.total_radius)

    def input_crop_side(self, patch_side: int) -> int:
        return int(patch_side + 4*self.total_radius)

    def input_crop_bytes(self, patch_side: int) -> int:
        side=self.input_crop_side(patch_side)
        return side*side*self.channels*self.bytes_per_value

    def full_input_bytes(self, height: int, width: int) -> int:
        return height*width*self.channels*self.bytes_per_value

    def full_activation_cache_bytes(self, height: int, width: int) -> int:
        # input + one state after every local layer
        return (self.layers+1)*height*width*self.channels*self.bytes_per_value

    def cache_to_crop_ratio(self, height: int, width: int, patch_side: int) -> float:
        return self.full_activation_cache_bytes(height,width)/self.input_crop_bytes(patch_side)

    def transfer_ms(self, byte_count: int, bandwidth_GBps: float) -> float:
        return byte_count/(bandwidth_GBps*1e9)*1e3


@dataclass(frozen=True)
class LocalResidencyChoice:
    name: str
    setup_ms: float
    update_ms: float

    def total_ms(self, updates: int) -> float:
        return self.setup_ms + updates*self.update_ms


@dataclass(frozen=True)
class LocalResidencyPlanner:
    """
    Time-horizon planner for repeated local updates.

    Typical modes:
      GPU full each update
      eager CPU activation-cache handoff
      cacheless CPU dual-crop reconstruction
      demand-paged CPU state/crop fetch
    """
    choices: tuple[LocalResidencyChoice,...]

    def best(self, updates: int) -> LocalResidencyChoice:
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        return min(self.choices,key=lambda c:c.total_ms(updates))

    @staticmethod
    def break_even(a: LocalResidencyChoice,b: LocalResidencyChoice) -> float|None:
        # a.setup + H*a.update = b.setup + H*b.update
        denom=a.update_ms-b.update_ms
        if abs(denom)<1e-15:
            return None
        return (b.setup_ms-a.setup_ms)/denom
