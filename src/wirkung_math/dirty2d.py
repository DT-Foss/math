from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


@dataclass(frozen=True, order=True)
class Rect:
    """Half-open 2-D rectangle [y0,y1) × [x0,x1)."""
    y0:int; x0:int; y1:int; x1:int

    def __post_init__(self):
        if min(self.y0,self.x0)<0 or self.y1<self.y0 or self.x1<self.x0:
            raise ValueError("invalid rectangle")

    @property
    def area(self)->int:
        return (self.y1-self.y0)*(self.x1-self.x0)

    def hull(self,other:"Rect")->"Rect":
        return Rect(min(self.y0,other.y0),min(self.x0,other.x0),
                    max(self.y1,other.y1),max(self.x1,other.x1))

    def expand(self,radius:int,height:int,width:int)->"Rect":
        return Rect(max(0,self.y0-radius),max(0,self.x0-radius),
                    min(height,self.y1+radius),min(width,self.x1+radius))

    def touches(self,other:"Rect")->bool:
        return not (self.y1 < other.y0 or other.y1 < self.y0 or
                    self.x1 < other.x0 or other.x1 < self.x0)


@dataclass(frozen=True)
class AreaCostModel:
    launch_ms:float
    cell_ms:float

    def tile_cost(self,r:Rect)->float:
        return self.launch_ms+self.cell_ms*r.area

    def partition_cost(self,tiles:list[Rect]|tuple[Rect,...])->float:
        return sum(self.tile_cost(r) for r in tiles)

    def merge_saves(self,a:Rect,b:Rect)->float:
        """Positive means the merged bounding tile is cheaper."""
        return self.tile_cost(a)+self.tile_cost(b)-self.tile_cost(a.hull(b))


def point_components_2d(points:np.ndarray,connectivity:int=8)->list[Rect]:
    """Exact connected components of sparse integer (y,x) dirty points."""
    pts=np.asarray(points,dtype=np.int64)
    if pts.size==0:
        return []
    if pts.ndim!=2 or pts.shape[1]!=2:
        raise ValueError("points must be N×2")
    if connectivity not in (4,8):
        raise ValueError("connectivity must be 4 or 8")
    remaining={tuple(map(int,p)) for p in pts}
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    out=[]
    while remaining:
        seed=remaining.pop(); stack=[seed]
        ys=[seed[0]]; xs=[seed[1]]
        while stack:
            y,x=stack.pop()
            for dy,dx in dirs:
                q=(y+dy,x+dx)
                if q in remaining:
                    remaining.remove(q); stack.append(q)
                    ys.append(q[0]); xs.append(q[1])
        out.append(Rect(min(ys),min(xs),max(ys)+1,max(xs)+1))
    return sorted(out)


def expand_components_2d(
    components:list[Rect],height:int,width:int,radius:int,layers:int=1
)->list[Rect]:
    reach=radius*layers
    return [r.expand(reach,height,width) for r in components]


def _subset_hulls(rects:list[Rect])->list[Rect|None]:
    n=len(rects); hulls:[Rect|None]=[None]*(1<<n)
    for mask in range(1,1<<n):
        bit=mask & -mask; i=bit.bit_length()-1
        rest=mask^bit
        hulls[mask]=rects[i] if rest==0 else hulls[rest].hull(rects[i])  # type: ignore[union-attr]
    return hulls


def schedule_rectangles_optimal(
    rects:list[Rect],model:AreaCostModel,exact_limit:int=12
)->list[Rect]:
    """
    Minimum-cost partition into bounding tiles.

    For <= exact_limit components, subset DP searches every set partition:
      cost(group)=launch_ms + cell_ms*area(hull(group)).
    For larger sets, a deterministic best-saving agglomerative fallback is used.
    """
    rects=list(rects)
    n=len(rects)
    if n<=1:return rects
    if n>exact_limit:
        cur=rects
        while True:
            best=None
            for i in range(len(cur)):
                for j in range(i+1,len(cur)):
                    saving=model.merge_saves(cur[i],cur[j])
                    if saving>0 and (best is None or saving>best[0]):
                        best=(saving,i,j)
            if best is None: return cur
            _,i,j=best
            merged=cur[i].hull(cur[j])
            cur=[r for k,r in enumerate(cur) if k not in (i,j)]+[merged]
            cur.sort()

    hulls=_subset_hulls(rects)
    cost=[0.0]*(1<<n)
    for mask in range(1,1<<n):
        cost[mask]=model.tile_cost(hulls[mask])  # type: ignore[arg-type]

    @lru_cache(None)
    def solve(mask:int):
        if mask==0:return (0.0,())
        first=mask & -mask
        best=(math.inf,())
        sub=mask
        while sub:
            if sub & first:
                tail_cost,tail_groups=solve(mask^sub)
                total=cost[sub]+tail_cost
                if total<best[0]:
                    best=(total,(sub,)+tail_groups)
            sub=(sub-1)&mask
        return best

    _,groups=solve((1<<n)-1)
    return sorted([hulls[g] for g in groups if hulls[g] is not None])  # type: ignore[list-item]


@dataclass(frozen=True)
class DirtyPlan2D:
    source:tuple[Rect,...]
    cones:tuple[Rect,...]
    tiles:tuple[Rect,...]
    predicted_cost_ms:float

    @property
    def scheduled_area(self)->int:
        return sum(r.area for r in self.tiles)


@dataclass(frozen=True)
class DirtyGraphCompiler2D:
    height:int
    width:int
    radius:int
    layers:int
    cost_model:AreaCostModel
    connectivity:int=8

    def compile_points(self,points:np.ndarray)->DirtyPlan2D:
        source=point_components_2d(points,self.connectivity)
        cones=expand_components_2d(source,self.height,self.width,self.radius,self.layers)
        tiles=schedule_rectangles_optimal(cones,self.cost_model)
        return DirtyPlan2D(tuple(source),tuple(cones),tuple(tiles),self.cost_model.partition_cost(tiles))
