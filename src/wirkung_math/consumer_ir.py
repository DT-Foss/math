from __future__ import annotations
from dataclasses import dataclass,field
from enum import IntEnum
from .demand_slice import delta_demand_slice


class EvidenceLevel(IntEnum):
    HEURISTIC=0
    VALIDATED=1
    PROVEN=2


@dataclass(frozen=True)
class ProgramNode:
    name:str
    family:str
    inputs:tuple[str,...]=()
    metadata:dict=field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRoute:
    name:str
    source:str
    target:str
    setup_ms:float
    update_ms:float
    query_ms:float
    state_bytes:int
    evidence:EvidenceLevel
    device:str="cpu"
    metadata:dict=field(default_factory=dict)

    def total_ms(self,updates:int,queries_per_update:float=1.0)->float:
        return float(self.setup_ms+updates*(self.update_ms+queries_per_update*self.query_ms))


@dataclass(frozen=True)
class CompilationPolicy:
    exact_only:bool=True
    allow_validated:bool=False

    def allows(self,route:CandidateRoute)->bool:
        if self.exact_only:
            return route.evidence==EvidenceLevel.PROVEN
        if route.evidence==EvidenceLevel.HEURISTIC:
            return False
        if route.evidence==EvidenceLevel.VALIDATED and not self.allow_validated:
            return False
        return True

    def filter(self,routes:tuple[CandidateRoute,...])->tuple[CandidateRoute,...]:
        return tuple(r for r in routes if self.allows(r))


@dataclass(frozen=True)
class ConsumerProgram:
    nodes:tuple[ProgramNode,...]
    changed:frozenset[str]
    requested:frozenset[str]

    def __post_init__(self)->None:
        names=[n.name for n in self.nodes]
        if len(names)!=len(set(names)):
            raise ValueError("duplicate node name")
        known=set(names)
        for node in self.nodes:
            for dep in node.inputs:
                if dep not in known:
                    raise ValueError(f"missing dependency {dep!r} for node {node.name!r}")
        for label,items in (("changed",self.changed),("requested",self.requested)):
            missing=set(items)-known
            if missing:
                raise ValueError(f"{label} nodes not found: {sorted(missing)!r}")

    @property
    def node_map(self)->dict[str,ProgramNode]:
        return {n.name:n for n in self.nodes}

    @property
    def edges(self)->tuple[tuple[str,str],...]:
        return tuple((dep,node.name) for node in self.nodes for dep in node.inputs)

    def required_nodes(self)->frozenset[str]:
        return delta_demand_slice(
            self.edges,self.changed,self.requested
        ).recompute
