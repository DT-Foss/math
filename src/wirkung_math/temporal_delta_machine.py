from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


class LinearDeltaRoute(str,Enum):
    COORDINATE="coordinate_delta"
    BLOCK="block_delta"
    BASIS="basis_delta"
    BATCH_LOWRANK="batch_lowrank"
    FULL="full"


@dataclass(frozen=True)
class DeltaPortfolio:
    width:int
    coordinate_support:int|None=None
    changed_blocks:int|None=None
    block_size:int|None=None
    basis_support:int|None=None
    basis_residual_support:int|None=None
    batch_size:int|None=None
    batch_rank:int|None=None


@dataclass(frozen=True)
class LinearRouteCost:
    route:LinearDeltaRoute
    multiply_count:int
    relative_to_full:float


@dataclass(frozen=True)
class LinearDeltaPlanner:
    input_width:int
    output_width:int
    fixed_overhead_equivalent_multiplies:int=0

    @property
    def full_multiplies(self)->int:
        return self.input_width*self.output_width

    def candidates(self,p:DeltaPortfolio)->tuple[LinearRouteCost,...]:
        full=self.full_multiplies
        rows=[LinearRouteCost(LinearDeltaRoute.FULL,full,1.0)]
        overhead=self.fixed_overhead_equivalent_multiplies
        if p.coordinate_support is not None:
            cost=self.output_width*p.coordinate_support+overhead
            rows.append(LinearRouteCost(LinearDeltaRoute.COORDINATE,cost,cost/full))
        if p.changed_blocks is not None and p.block_size is not None:
            cost=self.output_width*p.changed_blocks*p.block_size+overhead
            rows.append(LinearRouteCost(LinearDeltaRoute.BLOCK,cost,cost/full))
        if p.basis_support is not None:
            residual=p.basis_residual_support or 0
            cost=self.output_width*(p.basis_support+residual)+overhead
            rows.append(LinearRouteCost(LinearDeltaRoute.BASIS,cost,cost/full))
        if p.batch_size is not None and p.batch_rank is not None and p.batch_size>0:
            # Compare a full Bxdin by dinxdout pass to rank-r factor propagation.
            full_batch=p.batch_size*self.input_width*self.output_width
            cost=(p.batch_rank*self.input_width*self.output_width+
                  p.batch_size*p.batch_rank*self.output_width+overhead)
            rows.append(LinearRouteCost(
                LinearDeltaRoute.BATCH_LOWRANK,cost,cost/full_batch
            ))
        return tuple(rows)

    def choose(self,p:DeltaPortfolio)->LinearRouteCost:
        return min(self.candidates(p),key=lambda x:x.multiply_count)


@dataclass(frozen=True)
class BeastDecodeBudget:
    """Measured Beast ctx512 operation census, milliseconds per token."""
    linear_forms_ms:float=75.75
    normalized_reductions_ms:float=26.24
    io_ms:float=19.48
    other_ms:float=12.66

    @property
    def total_ms(self)->float:
        return self.linear_forms_ms+self.normalized_reductions_ms+self.io_ms+self.other_ms

    @property
    def tokens_per_second(self)->float:
        return 1000.0/self.total_ms

    def project(
        self,
        *,
        linear_speedup:float,
        normalized_speedup:float,
        io_speedup:float,
        other_speedup:float=1.0,
    )->"ProjectedDecodeBudget":
        values=(linear_speedup,normalized_speedup,io_speedup,other_speedup)
        if any(v<=0 for v in values):
            raise ValueError("speedups must be positive")
        return ProjectedDecodeBudget(
            self.linear_forms_ms/linear_speedup,
            self.normalized_reductions_ms/normalized_speedup,
            self.io_ms/io_speedup,
            self.other_ms/other_speedup,
        )


@dataclass(frozen=True)
class ProjectedDecodeBudget:
    linear_forms_ms:float
    normalized_reductions_ms:float
    io_ms:float
    other_ms:float

    @property
    def total_ms(self)->float:
        return self.linear_forms_ms+self.normalized_reductions_ms+self.io_ms+self.other_ms

    @property
    def tokens_per_second(self)->float:
        return 1000.0/self.total_ms

    def reaches(self,target_tokens_per_second:float)->bool:
        return self.tokens_per_second>=target_tokens_per_second


@dataclass(frozen=True)
class OrigamiDeltaMachineTarget:
    """
    System-level target for Beast/Origami.

    The runtime must attack all three measured dominant classes together:
    temporal linear forms, compiled normalized reductions, and direct codec I/O.
    """
    target_tokens_per_second:float=100.0
    current:BeastDecodeBudget=BeastDecodeBudget()

    @property
    def target_ms_per_token(self)->float:
        return 1000.0/self.target_tokens_per_second

    @property
    def required_end_to_end_speedup(self)->float:
        return self.current.total_ms/self.target_ms_per_token
