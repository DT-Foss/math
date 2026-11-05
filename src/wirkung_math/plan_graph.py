from __future__ import annotations
from dataclasses import dataclass
import heapq

@dataclass(frozen=True)
class PlanTransition:
    source:str
    target:str
    cost_ms:float
    action:str

@dataclass(frozen=True)
class PlanResult:
    cost_ms:float
    states:tuple[str,...]
    actions:tuple[str,...]

class RepresentationPlanGraph:
    """Shortest-path planner over representation + device states."""
    def __init__(self):
        self._adj:dict[str,list[PlanTransition]]={}

    def add(self,source:str,target:str,cost_ms:float,action:str)->None:
        if cost_ms<0:
            raise ValueError("cost must be nonnegative")
        self._adj.setdefault(source,[]).append(
            PlanTransition(source,target,float(cost_ms),action)
        )

    def shortest(self,start:str,goal:str)->PlanResult:
        q=[(0.0,start,(),(start,))]
        best={start:0.0}
        while q:
            cost,node,actions,states=heapq.heappop(q)
            if cost!=best.get(node):
                continue
            if node==goal:
                return PlanResult(cost,states,actions)
            for edge in self._adj.get(node,()):
                nc=cost+edge.cost_ms
                if nc<best.get(edge.target,float("inf")):
                    best[edge.target]=nc
                    heapq.heappush(q,(nc,edge.target,actions+(edge.action,),states+(edge.target,)))
        raise ValueError(f"no path from {start!r} to {goal!r}")
