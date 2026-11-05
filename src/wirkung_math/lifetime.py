from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRegime:
    name:str
    setup_ms:float
    update_ms:float
    query_ms:float=0.0

    def total_ms(self,updates:int,queries_per_update:float=1.0)->float:
        return self.setup_ms + updates*(self.update_ms+queries_per_update*self.query_ms)


@dataclass(frozen=True)
class HorizonPlanner:
    """Choose the cheapest execution/residency regime over a known horizon."""
    regimes:tuple[ExecutionRegime,...]

    def choose(self,updates:int,queries_per_update:float=1.0)->ExecutionRegime:
        return min(self.regimes,key=lambda r:r.total_ms(updates,queries_per_update))

    def ranking(self,updates:int,queries_per_update:float=1.0)->list[tuple[str,float]]:
        return sorted(
            ((r.name,r.total_ms(updates,queries_per_update)) for r in self.regimes),
            key=lambda x:x[1],
        )

    @staticmethod
    def pairwise_break_even(a:ExecutionRegime,b:ExecutionRegime,queries_per_update:float=1.0)->float|None:
        ca=a.update_ms+queries_per_update*a.query_ms
        cb=b.update_ms+queries_per_update*b.query_ms
        denom=ca-cb
        if abs(denom)<1e-15:return None
        return (b.setup_ms-a.setup_ms)/denom
