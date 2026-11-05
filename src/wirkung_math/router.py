from __future__ import annotations
from dataclasses import dataclass, asdict
from .lowrank import crossover_rank


@dataclass(frozen=True)
class RouteDecision:
    route: str
    direct_sparse_multiplies: int
    sparse_lowrank_multiplies: int | None
    full_multiplies: int
    crossover_rank: float | None

    def to_dict(self):
        return asdict(self)


def route_linear_delta(n: int, edits: int, width: int, rank: int | None = None) -> RouteDecision:
    """Operation-count router for full vs sparse vs sparse×low-rank value updates."""
    full = n * n * width
    sparse = n * edits * width
    if rank is None:
        return RouteDecision("sparse" if sparse < full else "full", sparse, None, full, None)
    slr = n * rank * (edits + width)
    route = min((full, "full"), (sparse, "sparse"), (slr, "sparse_lowrank"))[1]
    return RouteDecision(route, sparse, slr, full, crossover_rank(edits, width))
