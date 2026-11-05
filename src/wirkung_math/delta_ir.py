from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Structure(str,Enum):
    SPARSE_ROWS="sparse_rows"
    SPARSE_COLS="sparse_cols"
    LOW_RANK="low_rank"
    ROW_COLUMN="row_column"
    DIRTY_GEOMETRY="dirty_geometry"
    ADDITIVE_SUMMARY="additive_summary"
    RESPONSE_BASIS="response_basis"
    ROW_SCALE="row_scale"
    DENSE="dense"


@dataclass(frozen=True)
class DeltaDescriptor:
    """
    Structural type of a state transition.

    Operators transform descriptors; the compiler is allowed to switch
    representation instead of forcing one structure to survive every operator.
    """
    shape: tuple[int,...]
    structures: frozenset[Structure]
    changed_rows: int|None=None
    changed_cols: int|None=None
    rank_bound: int|None=None
    components: int|None=None
    metadata: dict=field(default_factory=dict)

    def through_elementwise(self) -> "DeltaDescriptor":
        keep=set(self.structures)
        keep.discard(Structure.LOW_RANK)
        if not ({Structure.SPARSE_ROWS,Structure.DIRTY_GEOMETRY}&keep):
            keep.add(Structure.DENSE)
        return DeltaDescriptor(
            self.shape,frozenset(keep),self.changed_rows,self.changed_cols,
            None,self.components,{**self.metadata,"operator":"elementwise"},
        )

    def through_pairwise_symmetric(self, changed_points: int) -> "DeltaDescriptor":
        n=self.shape[0]
        return DeltaDescriptor(
            (n,n),
            frozenset({Structure.ROW_COLUMN,Structure.LOW_RANK}),
            changed_rows=changed_points,
            changed_cols=changed_points,
            rank_bound=2*changed_points,
            metadata={**self.metadata,"operator":"pairwise_symmetric"},
        )

    def through_global_additive_reduction(self, output_shape: tuple[int,...]) -> "DeltaDescriptor":
        return DeltaDescriptor(
            output_shape,
            frozenset({Structure.ADDITIVE_SUMMARY}),
            metadata={**self.metadata,"operator":"global_additive_reduction"},
        )

    def through_fft(self) -> "DeltaDescriptor":
        return DeltaDescriptor(
            self.shape,
            frozenset({Structure.RESPONSE_BASIS}),
            changed_rows=self.changed_rows,
            metadata={**self.metadata,"operator":"fft"},
        )
