from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .dirty import (
    GeometryCostModel,
    Interval,
    connected_components,
    expand_components,
    schedule_components_optimal,
)


@dataclass(frozen=True)
class DirtyPlan:
    source_components: tuple[Interval, ...]
    cone_components: tuple[Interval, ...]
    scheduled_tiles: tuple[Interval, ...]
    predicted_cost_ms: float

    @property
    def source_cells(self) -> int:
        return sum(x.width for x in self.source_components)

    @property
    def cone_cells(self) -> int:
        return sum(x.width for x in self.cone_components)

    @property
    def scheduled_cells(self) -> int:
        return sum(x.width for x in self.scheduled_tiles)

    @property
    def tile_count(self) -> int:
        return len(self.scheduled_tiles)

    def to_dict(self) -> dict:
        return {
            "source_components":[asdict(x) for x in self.source_components],
            "cone_components":[asdict(x) for x in self.cone_components],
            "scheduled_tiles":[asdict(x) for x in self.scheduled_tiles],
            "source_cells":self.source_cells,
            "cone_cells":self.cone_cells,
            "scheduled_cells":self.scheduled_cells,
            "tile_count":self.tile_count,
            "predicted_cost_ms":self.predicted_cost_ms,
        }


@dataclass(frozen=True)
class DirtyGraphCompiler:
    """
    Compile sparse support into executable local tiles.

    1. Recover connected source components.
    2. Propagate each component through the local receptive field.
    3. Coalesce cones that physically overlap.
    4. Choose the globally cheapest remaining tile partition under the
       calibrated launch-vs-extra-area model.
    """
    length: int
    radius: int
    layers: int
    cost_model: GeometryCostModel

    def compile(self, dirty_indices: np.ndarray) -> DirtyPlan:
        source=connected_components(dirty_indices)
        cones=expand_components(source,self.length,self.radius,self.layers)
        tiles=schedule_components_optimal(cones,self.cost_model)
        return DirtyPlan(
            tuple(source),
            tuple(cones),
            tuple(tiles),
            self.cost_model.partition_cost(tiles),
        )
