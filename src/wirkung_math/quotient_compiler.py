from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .consumer_ir import (
    CandidateRoute,CompilationPolicy,ConsumerProgram,EvidenceLevel
)
from .linear_quotient import LinearConsumerQuotient
from .plan_graph import RepresentationPlanGraph,PlanResult


@dataclass(frozen=True)
class CompilationResult:
    required_nodes:frozenset[str]
    accepted:tuple[CandidateRoute,...]
    rejected:tuple[CandidateRoute,...]
    selected:tuple[CandidateRoute,...]
    plan:PlanResult

    @property
    def selected_route_names(self)->tuple[str,...]:
        return tuple(r.name for r in self.selected)

    @property
    def total_ms(self)->float:
        return float(self.plan.cost_ms)


class QuotientCompiler:
    def __init__(self,policy:CompilationPolicy|None=None):
        self.policy=policy or CompilationPolicy()

    def compile(
        self,
        program:ConsumerProgram,
        routes:tuple[CandidateRoute,...],
        *,
        start:str,
        goal:str,
        updates:int,
        queries_per_update:float=1.0,
    )->CompilationResult:
        if updates<0 or queries_per_update<0:
            raise ValueError("invalid horizon")
        names=[r.name for r in routes]
        if len(names)!=len(set(names)):
            raise ValueError("candidate route names must be unique")
        accepted=self.policy.filter(routes)
        allowed_ids={id(r) for r in accepted}
        rejected=tuple(r for r in routes if id(r) not in allowed_ids)
        graph=RepresentationPlanGraph()
        by_name={}
        for route in accepted:
            by_name[route.name]=route
            graph.add(
                route.source,route.target,
                route.total_ms(updates,queries_per_update),
                route.name,
            )
        try:
            plan=graph.shortest(start,goal)
        except ValueError as exc:
            raise ValueError(
                f"no semantically allowed plan from {start!r} to {goal!r}"
            ) from exc
        selected=tuple(by_name[name] for name in plan.actions)
        return CompilationResult(
            required_nodes=program.required_nodes(),
            accepted=accepted,rejected=rejected,selected=selected,plan=plan,
        )

    def compile_linear_chain(
        self,
        *,
        name:str,
        program:ConsumerProgram,
        upstream:np.ndarray,
        consumer:np.ndarray,
        full_ms:float,
        quotient_ms:float|None=None,
        updates:int=1,
        queries_per_update:float=1.0,
        device:str="cpu",
    )->CompilationResult:
        A=np.asarray(upstream,dtype=np.float64)
        C=np.asarray(consumer,dtype=np.float64)
        if A.ndim!=2 or C.ndim!=2 or C.shape[1]!=A.shape[0]:
            raise ValueError("linear chain shape mismatch")
        if full_ms<0 or quotient_ms is not None and quotient_ms<0:
            raise ValueError("cost must be nonnegative")
        quotient=LinearConsumerQuotient.fit(C)
        full_ops=int(A.size+C.size)
        quotient_ops=int(
            quotient.dimension*A.shape[1]
            + C.shape[0]*quotient.dimension
        )
        if quotient_ms is None:
            quotient_ms=(
                0.0 if full_ops==0 else
                float(full_ms)*quotient_ops/full_ops
            )
            cost_source="operation_model"
        else:
            cost_source="measured"

        routes=(
            CandidateRoute(
                f"{name}:full","start","goal",0.0,float(full_ms),0.0,
                int(A.shape[0]*8),EvidenceLevel.PROVEN,device,
                {
                    "representation":"dense_intermediate",
                    "runtime_multiplies":full_ops,
                    "cost_source":"measured",
                },
            ),
            CandidateRoute(
                f"{name}:consumer_quotient","start","goal",0.0,
                float(quotient_ms),0.0,int(quotient.dimension*8),
                EvidenceLevel.PROVEN,device,
                {
                    "representation":"consumer_row_space",
                    "quotient_dimension":quotient.dimension,
                    "dense_intermediate_dimension":int(A.shape[0]),
                    "runtime_multiplies":quotient_ops,
                    "full_runtime_multiplies":full_ops,
                    "reconstruction_residual":quotient.reconstruction_residual,
                    "cost_source":cost_source,
                },
            ),
        )
        return self.compile(
            program,routes,start="start",goal="goal",updates=updates,
            queries_per_update=queries_per_update,
        )
