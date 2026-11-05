from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Hashable,Iterable


Node=Hashable


@dataclass(frozen=True)
class DemandSlice:
    changed:frozenset[Node]
    requested:frozenset[Node]
    forward_dirty:frozenset[Node]
    backward_demand:frozenset[Node]
    recompute:frozenset[Node]


def _reach(starts:Iterable[Node],adj:dict[Node,set[Node]])->set[Node]:
    seen=set(starts)
    q=deque(seen)
    while q:
        x=q.popleft()
        for y in adj.get(x,()):
            if y not in seen:
                seen.add(y);q.append(y)
    return seen


def delta_demand_slice(
    edges:Iterable[tuple[Node,Node]],
    changed:Iterable[Node],
    requested:Iterable[Node],
)->DemandSlice:
    """
    Necessary raw recomputation set before algebraic rewrite replacement:

      R = Desc(changed) intersect Anc(requested).

    This unifies forward change propagation with backward demand slicing.
    """
    fwd:dict[Node,set[Node]]={}
    rev:dict[Node,set[Node]]={}
    for a,b in edges:
        fwd.setdefault(a,set()).add(b)
        rev.setdefault(b,set()).add(a)
        fwd.setdefault(b,set()); rev.setdefault(a,set())
    c=set(changed);r=set(requested)
    dirty=_reach(c,fwd)
    demand=_reach(r,rev)
    return DemandSlice(
        frozenset(c),frozenset(r),frozenset(dirty),frozenset(demand),
        frozenset(dirty & demand),
    )
